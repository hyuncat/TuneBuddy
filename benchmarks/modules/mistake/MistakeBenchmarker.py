from __future__ import annotations

import sys
import time
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeAlias

import numpy as np

def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


_BOOTSTRAP_ROOT = _bootstrap_repo_root()
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from benchmarks.paths import REPO_ROOT, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()
ROOT = REPO_ROOT

from app_logic.Alignment import Mistake  # noqa: E402
from app_logic.NoteData import Note, NoteData  # noqa: E402
from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from benchmarks.modules.mistake.MistakeInjector import MistakeInjector, TruthEvent  # noqa: E402
from benchmarks.note.NoteBenchmarker import NoteBenchmarker, OneInstrumentScoreData  # noqa: E402
from benchmarks.modules.pitch.PitchBenchmarker import PathLike, PitchBenchmarker  # noqa: E402

MistakeMode: TypeAlias = Literal["symbolic", "audio"]
MistakeCounts: TypeAlias = dict[str, tuple[int, int, int]]


class MistakeScore(TypedDict, total=False):
    counts: MistakeCounts
    pitch_detector_compute_time: float
    pitch_smoother_compute_time: float
    pitch_compute_time: float
    note_compute_time: float
    mistake_detection_compute_time: float
    mistake_check_compute_time: float



class MistakeBenchmarker(NoteBenchmarker):
    MISTAKE_DB_OUTPUT_ALIASES = {
        "wohlfahrt": "wolfhart",
        "wolfhart": "wolfhart",
    }

    def __init__(
        self,
        onset_tolerance: float = 0.05,
    ) -> None:
        super().__init__(onset_tolerance=onset_tolerance)

    def mistake_db_dataset_dir(self, dataset: str) -> Path:
        """Output corpus directory under benchmarks/datasets/mistake-db.

        The source corpus is spelled ``wohlfahrt`` in violin-etudes. The user-
        facing benchmark DB folder is kept as ``wolfhart`` to match the requested
        layout.
        """
        output_dataset = self.MISTAKE_DB_OUTPUT_ALIASES.get(dataset, dataset)
        return self.MISTAKE_DIR / output_dataset

    @staticmethod
    def dataset_from_etude_midi(midi_path: PathLike) -> str:
        return PitchBenchmarker.dataset_name_for_midi(midi_path)

    def mistake_db_paths(self, dataset: str, track_id: str) -> dict[str, Path]:
        dataset_dir = self.mistake_db_dataset_dir(dataset)
        return {
            "dataset": dataset_dir,
            "audio": dataset_dir / "audio" / f"{track_id}.wav",
            "midi": dataset_dir / "midi" / f"{track_id}.mid",
            "pitch_data": self.pitch_cache_path(dataset_dir, track_id),
            "note_data": self.note_cache_path(dataset_dir, track_id),
            "truth": dataset_dir / "truth" / f"{track_id}.truth.json",
        }

    @staticmethod
    def _jsonable_truth(truth: Sequence[TruthEvent]) -> list[dict[str, Any]]:
        return [
            {key: (int(value) if isinstance(value, np.integer) else value)
             for key, value in event.items()}
            for event in truth
        ]

    def write_truth_data(
        self,
        truth_path: PathLike,
        *,
        dataset: str,
        source_midi: Path,
        track_id: str,
        seed: int,
        truth: Sequence[TruthEvent],
        injector: MistakeInjector,
        pitch_timing: dict[str, float] | None = None,
        note_timing: dict[str, float] | None = None,
    ) -> Path:
        truth_path = Path(truth_path)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dataset": dataset,
            "track_id": track_id,
            "seed": int(seed),
            "source_midi": str(source_midi),
            "truth": self._jsonable_truth(truth),
            "injector": injector.last_metadata,
            "pitch_timing": pitch_timing or {},
            "note_timing": note_timing or {},
        }
        with open(truth_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return truth_path

    def generate_mistake_db_track(
        self,
        midi_path: PathLike,
        dataset: str,
        seed: int,
        injector: MistakeInjector | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Generate one mistake-db item and its analysis caches.

        Writes:
          - <mistake-db>/<dataset>/audio/<track>_seed<seed>.wav
          - <mistake-db>/<dataset>/pitch_data/<track>_seed<seed>.pitch.pkl.xz
          - <mistake-db>/<dataset>/note_data/<track>_seed<seed>.note.json

        A generated MIDI and truth JSON are also written for reproducibility.
        The cached note data is PELT-L2 detector output after the same first/last
        boundary trim used by NoteBenchmarker.
        """
        midi_path = Path(midi_path)
        injector = injector or MistakeInjector(out_dir=self.MISTAKE_DIR)
        score_data = OneInstrumentScoreData(midi_path)
        reference_notes = score_data.note_data
        performance_notes, truth = injector.inject(
            reference_notes,
            np.random.default_rng(seed),
        )

        track_id = f"{midi_path.stem}_seed{seed}"
        paths = self.mistake_db_paths(dataset, track_id)
        for key in ("audio", "midi", "pitch_data", "note_data", "truth"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)

        if force or not paths["midi"].exists():
            self.notedata_to_pm(performance_notes).write(str(paths["midi"]))
        audio_path = self.synth_midi(paths["midi"], out_dir=paths["audio"].parent, force=force)

        config = self.config_for(*self.range_from_midi(score_data.midi_numbers))
        recording = self.recording_for(
            config,
            score_data=OneInstrumentScoreData(midi_path),
        )
        recording.audio_data = AudioData(
            audio_filepath=str(audio_path),
            config=recording.config,
        )

        pitch_timing = self.load_or_detect_pitches(
            recording,
            cache_path=paths["pitch_data"],
            smooth=True,
            write_cache=True,
        )

        note_timing: dict[str, float]
        if paths["note_data"].exists() and not force:
            recording.note_data, note_metadata = self.load_note_data(paths["note_data"])
            note_timing = {
                "note_compute_time": float(note_metadata.get("note_compute_time", 0.0)),
            }
        else:
            recording.reset_analysis()
            _, note_compute_time = self.detect_recording_notes_timed(recording)
            self._trim_boundary_notes(recording.note_data, performance_notes)
            note_timing = {"note_compute_time": note_compute_time}
            self.save_note_data(
                recording.note_data,
                paths["note_data"],
                metadata={
                    **note_timing,
                    "model": "l2",
                    "method": "recording.detect_notes",
                    "trimmed_boundaries": True,
                    "trim_reference": "generated_performance_midi",
                },
            )

        self.write_truth_data(
            paths["truth"],
            dataset=dataset,
            source_midi=midi_path,
            track_id=track_id,
            seed=seed,
            truth=truth,
            injector=injector,
            pitch_timing=pitch_timing,
            note_timing=note_timing,
        )
        return {
            "dataset": dataset,
            "track_id": track_id,
            "seed": int(seed),
            "source_midi": str(midi_path),
            "audio": str(audio_path),
            "midi": str(paths["midi"]),
            "pitch_data": str(paths["pitch_data"]),
            "note_data": str(paths["note_data"]),
            "truth": str(paths["truth"]),
            "truth_events": len(truth),
            **pitch_timing,
            **note_timing,
        }

    def build_mistake_db(
        self,
        datasets: Sequence[str] = ("kayser", "wohlfahrt"),
        seeds: Iterable[int] = range(6),
        injector: MistakeInjector | None = None,
        max_tracks: int | None = None,
        force: bool = False,
        verbose: bool = True,
        write_manifest: bool = True,
    ) -> pd.DataFrame:
        """Build the Polytune-style mistake database for the etude corpora."""
        import pandas as pd

        injector = injector or MistakeInjector(out_dir=self.MISTAKE_DIR)
        seed_values = list(seeds)
        rows: list[dict[str, Any]] = []
        for dataset in datasets:
            tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
            for track_index, (title, midi_path) in enumerate(tracks, start=1):
                for seed in seed_values:
                    row = self.generate_mistake_db_track(
                        midi_path,
                        dataset=dataset,
                        seed=int(seed),
                        injector=injector,
                        force=force,
                    )
                    rows.append(row)
                    if verbose:
                        print(
                            f"[mistake-db/{dataset}] "
                            f"{track_index}/{len(tracks)} seed={seed} {title[:40]}"
                        )

        df = pd.DataFrame(rows)
        if write_manifest:
            manifest_path = self.MISTAKE_DIR / "manifest.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(manifest_path, index=False)
        return df

    def analyze_recording(
        self,
        recording,
        method: str = "pelt",
        model: str = "l2",
        do_correction: bool = False,
        detect_timing: bool = True,
        update_distances: bool = True,
        truncate: bool = False,
        note_cache_path: PathLike | None = None,
        trim_reference: NoteData | None = None,
        onset_aware: bool = True,
    ) -> dict[str, float]:
        recording.reset_analysis()
        if note_cache_path is not None and Path(note_cache_path).exists():
            # Reuse cached production notes. The boundary trim below is idempotent,
            # so both old raw caches and new trimmed mistake-db caches are safe.
            recording.note_data, metadata = self.load_note_data(note_cache_path)
            note_compute_time = float(metadata.get("note_compute_time", 0.0))
        else:
            del method, model
            _, note_compute_time = self.detect_recording_notes_timed(recording)
            if trim_reference is not None:
                # same boundary trim as the note benchmark: clamp the first/last
                # detected notes to the synthesized MIDI's true durations (synth
                # attack/release swell) BEFORE onset-resize / string-edit
                # alignment and before saving the mistake-db note cache.
                self._trim_boundary_notes(recording.note_data, trim_reference)
            if note_cache_path is not None:
                self.save_note_data(
                    recording.note_data,
                    note_cache_path,
                    metadata={
                        "note_compute_time": note_compute_time,
                        "model": "l2",
                        "method": "recording.detect_notes",
                        "trimmed_boundaries": trim_reference is not None,
                    },
                )
        if trim_reference is not None and note_cache_path is not None and Path(note_cache_path).exists():
            self._trim_boundary_notes(recording.note_data, trim_reference)
        if recording.note_data.times:
            recording.resize_score(to_span="note")

        mistake_start = time.perf_counter()
        recording.detect_mistakes(onset_aware=onset_aware)
        if detect_timing:
            recording.mistake_detector.detect_timing_mistakes()
        mistake_detection_compute_time = time.perf_counter() - mistake_start

        mistake_check_compute_time = 0.0
        if do_correction:
            check_start = time.perf_counter()
            recording.mistake_checker.mistake_correction_loop()
            recording.reindex_mistakes()
            mistake_check_compute_time = time.perf_counter() - check_start
            if detect_timing:
                timing_start = time.perf_counter()
                recording.mistake_detector.detect_timing_mistakes()
                mistake_detection_compute_time += time.perf_counter() - timing_start

        if update_distances:
            recording.update_alignment_distances()
        if truncate:
            recording.trim_end(mark_unsaved=False)

        return {
            "note_compute_time": note_compute_time,
            "mistake_detection_compute_time": mistake_detection_compute_time,
            "mistake_check_compute_time": mistake_check_compute_time,
        }

    def bench_mistake_track(
        self,
        midi_path: PathLike,
        injector: MistakeInjector,
        seeds: Iterable[int] = range(6),
        mode: MistakeMode = "symbolic",
        max_sec: float | None = None,
        correct_symbolic: bool = False,
        onset_aware: bool = True,
        canonical: bool = True,
    ) -> pd.DataFrame:
        return self.aggregate(
            self._mistake_scores(
                midi_path,
                injector,
                seeds,
                mode,
                max_sec,
                correct_symbolic,
                onset_aware,
                canonical,
            ),
            canonical=canonical,
        )

    def bench_mistake_dataset(
        self,
        dataset: str,
        injector: MistakeInjector,
        seeds: Iterable[int] = range(6),
        mode: MistakeMode = "symbolic",
        max_tracks: int | None = None,
        max_sec: float | None = None,
        correct_symbolic: bool = False,
        onset_aware: bool = True,
        canonical: bool = True,
        verbose: bool = True,
        write: bool = False,
    ) -> pd.DataFrame:
        tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
        scores: list[MistakeScore] = []
        for i, (title, midi_path) in enumerate(tracks):
            scores += self._mistake_scores(
                midi_path,
                injector,
                seeds,
                mode,
                max_sec,
                correct_symbolic,
                onset_aware,
                canonical,
            )
            if verbose:
                print(f"[{dataset}/{mode}] {i+1}/{len(tracks)} {title[:32]}")
        df = self.aggregate(scores, canonical=canonical)
        if write:
            self.write_dataset_result(df, "mistakes", dataset)
        return df

    def clean_render_mistakes(
        self,
        midi_path: PathLike,
        max_sec: float | None = None,
    ) -> dict[str, float | int | str | bool]:
        midi_path = Path(midi_path)
        score_data = OneInstrumentScoreData(midi_path)
        reference_notes = score_data.note_data
        config = self.config_for(*self.range_from_midi(score_data.midi_numbers))
        recording = self.recording_for(config, score_data=score_data)
        recording.audio_data = AudioData(
            audio_filepath=str(self.synth_midi(midi_path)),
            config=recording.config,
        )
        if max_sec:
            recording.audio_data.end_index = min(
                recording.audio_data.end_index,
                int(max_sec * config.sr),
            )
        pitch_timing = self.load_or_detect_pitches(
            recording,
            cache_path=self.pitch_cache_path(
                self.etude_corpus_dir_for_midi(midi_path),
                midi_path.stem,
            ),
            smooth=True,
            write_cache=True,
        )
        analyze_timing = self.analyze_recording(
            recording,
            note_cache_path=self.note_cache_path(
                self.etude_corpus_dir_for_midi(midi_path),
                midi_path.stem,
            ),
            trim_reference=reference_notes,
        )
        reference_note_count = sum(
            1
            for note_time in reference_notes.times
            if max_sec is None or note_time < max_sec
        )
        return dict(
            track=midi_path.stem,
            spurious=len(recording.alignment.pitch_mistakes),
            timing_spurious=len(recording.alignment.timing_mistakes),
            **{"Reference Notes": reference_note_count},
            **pitch_timing,
            **analyze_timing,
        )

    def _mistake_scores(
        self,
        midi_path: PathLike,
        injector: MistakeInjector,
        seeds: Iterable[int],
        mode: MistakeMode,
        max_sec: float | None,
        correct_symbolic: bool = False,
        onset_aware: bool = True,
        canonical: bool = True,
    ) -> list[MistakeScore]:
        midi_path = Path(midi_path)
        score_data = OneInstrumentScoreData(midi_path)
        reference_notes = score_data.note_data
        scores: list[MistakeScore] = []
        if mode == "symbolic":
            recording = self.recording_for(
                self.config_for(196, 3000),
                score_data=score_data,
            )
            for seed in seeds:
                performance_notes, truth = injector.inject(
                    reference_notes,
                    np.random.default_rng(seed),
                )
                recording.note_data = performance_notes
                mistake_detection_start = time.perf_counter()
                recording.detect_mistakes(onset_aware=onset_aware)
                mistake_detection_compute_time = time.perf_counter() - mistake_detection_start
                mistake_check_compute_time = 0.0
                if correct_symbolic:
                    mistake_start = time.perf_counter()
                    recording.mistake_checker.mistake_correction_loop()
                    recording.reindex_mistakes()
                    mistake_check_compute_time = time.perf_counter() - mistake_start
                # PolyTune labels no timing mistakes, so the symbolic benchmark
                # scores pitch mistakes (substitution/deletion/insertion) only.
                detected_mistakes = recording.alignment.pitch_mistakes
                scores.append(
                    MistakeScore(
                        counts=self.score_symbolic(
                            detected_mistakes,
                            truth,
                            pairs=recording.alignment.pairs,
                            canonical=canonical,
                        ),
                        mistake_detection_compute_time=mistake_detection_compute_time,
                        mistake_check_compute_time=mistake_check_compute_time,
                    )
                )
            return scores

        if mode == "audio":
            pitch_range = self.range_from_midi(score_data.midi_numbers)
            dataset = self.dataset_from_etude_midi(midi_path)
            for seed in seeds:
                performance_notes, truth = injector.inject(
                    reference_notes,
                    np.random.default_rng(seed),
                )
                performance_name = f"{midi_path.stem}_seed{seed}"
                paths = self.mistake_db_paths(dataset, performance_name)
                for key in ("audio", "midi", "pitch_data", "note_data", "truth"):
                    paths[key].parent.mkdir(parents=True, exist_ok=True)
                if not paths["midi"].exists():
                    self.notedata_to_pm(performance_notes).write(str(paths["midi"]))
                audio_path = self.synth_midi(
                    paths["midi"],
                    out_dir=paths["audio"].parent,
                    force=False,
                )
                if max_sec:
                    truth = [event for event in truth if event["time"] < max_sec]
                config = self.config_for(*pitch_range)
                recording = self.recording_for(
                    config,
                    score_data=OneInstrumentScoreData(midi_path),
                )
                recording.audio_data = AudioData(
                    audio_filepath=str(audio_path),
                    config=recording.config,
                )
                if max_sec:
                    recording.audio_data.end_index = min(
                        recording.audio_data.end_index,
                        int(max_sec * config.sr),
                    )
                pitch_timing = self.load_or_detect_pitches(
                    recording,
                    cache_path=paths["pitch_data"],
                    smooth=True,
                    write_cache=True,
                )
                analyze_timing = self.analyze_recording(
                    recording,
                    note_cache_path=paths["note_data"],
                    trim_reference=performance_notes,
                    detect_timing=False,
                    onset_aware=onset_aware,
                )
                self.write_truth_data(
                    paths["truth"],
                    dataset=dataset,
                    source_midi=midi_path,
                    track_id=performance_name,
                    seed=int(seed),
                    truth=truth,
                    injector=injector,
                    pitch_timing=pitch_timing,
                    note_timing={
                        "note_compute_time": analyze_timing.get(
                            "note_compute_time",
                            0.0,
                        )
                    },
                )
                detected_mistakes = recording.alignment.pitch_mistakes
                scores.append(
                    MistakeScore(
                        counts=self.score_onset(
                            detected_mistakes,
                            truth,
                            onset_tolerance=0.1,
                            pairs=recording.alignment.pairs,
                            canonical=canonical,
                        ),
                        **pitch_timing,
                        **analyze_timing,
                    )
                )
            return scores

        raise ValueError(f"unknown mistake mode: {mode!r}")

    @staticmethod
    def _match_onsets(
        detected_times: Sequence[float],
        truth_times: Sequence[float],
        onset_tolerance: float,
    ) -> tuple[int, int, int]:
        used_truth_events = [False] * len(truth_times)
        true_positives = 0
        for detected_time in sorted(detected_times):
            for truth_index, truth_time in enumerate(truth_times):
                if (
                    not used_truth_events[truth_index]
                    and abs(detected_time - truth_time) <= onset_tolerance
                ):
                    used_truth_events[truth_index] = True
                    true_positives += 1
                    break
        return (
            true_positives,
            len(detected_times) - true_positives,
            len(truth_times) - true_positives,
        )

    @classmethod
    def score_correct_alignment(
        cls,
        pairs: Sequence[tuple[Note | None, Note | None]] | None,
        mistakes: Sequence[Mistake],
        truth: Sequence[TruthEvent],
    ) -> tuple[int, int, int]:
        if pairs is None:
            return (0, 0, 0)

        truth_incorrect_ids = {
            int(event["score_note_id"])
            for event in truth
            if "score_note_id" in event
        }
        detected_incorrect_ids = {
            int(mistake.midi_note.id)
            for mistake in mistakes
            if mistake.type != "insertion" and mistake.midi_note is not None
        }

        all_score_ids: set[int] = set()
        predicted_correct_ids: set[int] = set()
        for user_note, score_note in pairs:
            if score_note is None:
                continue
            score_id = int(score_note.id)
            all_score_ids.add(score_id)
            if user_note is None or score_id in detected_incorrect_ids:
                continue

            source_score_id = getattr(
                user_note,
                "source_score_id",
                _SOURCE_SCORE_ID_UNSET,
            )
            if source_score_id is not _SOURCE_SCORE_ID_UNSET:
                if source_score_id is None or int(source_score_id) != score_id:
                    continue

            predicted_correct_ids.add(score_id)

        truth_correct_ids = all_score_ids - truth_incorrect_ids
        true_positives = len(predicted_correct_ids & truth_correct_ids)
        false_positives = len(predicted_correct_ids - truth_correct_ids)
        false_negatives = len(truth_correct_ids - predicted_correct_ids)
        return true_positives, false_positives, false_negatives

    @classmethod
    def score_symbolic(
        cls,
        mistakes: Sequence[Mistake],
        truth: Sequence[TruthEvent],
        onset_tolerance: float = 0.06,
        pairs: Sequence[tuple[Note | None, Note | None]] | None = None,
        canonical: bool = True,
    ) -> MistakeCounts:
        """Score substitution/deletion/insertion (+correct) for the symbolic bench.

        `canonical` maps substitution losslessly onto PolyTune's missed/extra space:
        a wrong note is the intended score note MISSED (a deletion) plus the played
        pitch being EXTRA (an insertion). Folding both truth and detected
        substitutions into the deletion/insertion tallies makes the score invariant
        to whether a wrong note is represented as one substitution or as a
        deletion+insertion (and stops a detector substitution on a screwup-2/3 extra
        note from being a pure false positive). The `substitution` row is still
        reported as the direct sub-vs-sub agreement, but it is informational under
        `canonical` (OVERALL sums deletion+insertion only — see aggregate)."""
        truth_substitution_ids = {
            event["score_note_id"] for event in truth if event["type"] == "substitution"
        }
        truth_substitution_times = [
            event["time"] for event in truth if event["type"] == "substitution"
        ]
        truth_deletion_ids = {
            event["score_note_id"] for event in truth if event["type"] == "deletion"
        }
        truth_insertion_times = [
            event["time"] for event in truth if event["type"] == "insertion"
        ]
        detected_substitution_ids = [
            mistake.midi_note.id
            for mistake in mistakes
            if mistake.type == "substitution"
        ]
        detected_substitution_times = [
            mistake.user_note.start_time
            for mistake in mistakes
            if mistake.type == "substitution"
        ]
        detected_deletion_ids = [
            mistake.midi_note.id for mistake in mistakes if mistake.type == "deletion"
        ]
        detected_insertion_times = [
            mistake.user_note.start_time
            for mistake in mistakes
            if mistake.type == "insertion"
        ]

        # deletion (== "missed") and insertion (== "extra"): fold the substitution
        # halves in under canonical scoring.
        deletion_truth = set(truth_deletion_ids)
        deletion_detected = list(detected_deletion_ids)
        insertion_truth_times = list(truth_insertion_times)
        insertion_detected_times = list(detected_insertion_times)
        if canonical:
            deletion_truth |= truth_substitution_ids
            deletion_detected += detected_substitution_ids
            insertion_truth_times += truth_substitution_times
            insertion_detected_times += detected_substitution_times

        counts_by_type: MistakeCounts = {}
        # substitution row: direct agreement (informational under canonical).
        sub_tp = len(set(detected_substitution_ids) & truth_substitution_ids)
        counts_by_type["substitution"] = (
            sub_tp,
            len(detected_substitution_ids) - sub_tp,
            len(truth_substitution_ids) - sub_tp,
        )
        del_tp = len(set(deletion_detected) & deletion_truth)
        counts_by_type["deletion"] = (
            del_tp,
            len(deletion_detected) - del_tp,
            len(deletion_truth) - del_tp,
        )
        counts_by_type["insertion"] = cls._match_onsets(
            insertion_detected_times,
            sorted(insertion_truth_times),
            onset_tolerance,
        )
        # No early/late/short/long: PolyTune treats timing/duration jitter as
        # unlabeled realism, so there is no timing truth to score against.
        if pairs is not None:
            counts_by_type["correct"] = cls.score_correct_alignment(
                pairs,
                mistakes,
                truth,
            )
        return counts_by_type

    @classmethod
    def score_onset(
        cls,
        mistakes: Sequence[Mistake],
        truth: Sequence[TruthEvent],
        onset_tolerance: float = 0.1,
        pairs: Sequence[tuple[Note | None, Note | None]] | None = None,
        canonical: bool = True,
    ) -> MistakeCounts:
        """Onset-matched scoring for the audio bench. `canonical` folds substitution
        into missed/extra exactly as score_symbolic does: the substitution's score
        onset joins the deletion (missed) times, its played onset joins the insertion
        (extra) times. The `substitution` row stays as the direct agreement
        (informational under canonical)."""
        def truth_times(mistake_type: str) -> list[float]:
            return [e["time"] for e in truth if e["type"] == mistake_type]

        def detected_times(mistake_type: str, attr: str) -> list[float]:
            return [
                getattr(mistake, attr).start_time
                for mistake in mistakes
                if mistake.type == mistake_type and getattr(mistake, attr) is not None
            ]

        substitution_truth = truth_times("substitution")
        substitution_played = detected_times("substitution", "user_note")
        counts_by_type: MistakeCounts = {}
        # substitution row: direct agreement on the played onset (informational
        # under canonical).
        counts_by_type["substitution"] = cls._match_onsets(
            substitution_played, substitution_truth, onset_tolerance
        )

        deletion_truth = truth_times("deletion")
        deletion_detected = detected_times("deletion", "midi_note")
        insertion_truth = truth_times("insertion")
        insertion_detected = detected_times("insertion", "user_note")
        if canonical:
            # missed half -> score onset; extra half -> played onset.
            deletion_truth += substitution_truth
            deletion_detected += detected_times("substitution", "midi_note")
            insertion_truth += substitution_truth
            insertion_detected += substitution_played

        counts_by_type["deletion"] = cls._match_onsets(
            deletion_detected, deletion_truth, onset_tolerance
        )
        counts_by_type["insertion"] = cls._match_onsets(
            insertion_detected, insertion_truth, onset_tolerance
        )
        if pairs is not None:
            counts_by_type["correct"] = cls.score_correct_alignment(
                pairs,
                mistakes,
                truth,
            )
        return counts_by_type

    @staticmethod
    def aggregate(scores: Sequence[MistakeScore], canonical: bool = True) -> pd.DataFrame:
        import pandas as pd

        mistake_types = ("substitution", "deletion", "insertion")
        # Under canonical scoring substitution is already folded into deletion
        # (missed) + insertion (extra), so it is reported as an informational row
        # but excluded from OVERALL to avoid double-counting.
        overall_types = ("deletion", "insertion") if canonical else mistake_types
        include_correct = any("correct" in score["counts"] for score in scores)
        row_types = (*mistake_types, "correct") if include_correct else mistake_types
        aggregate_counts = {mistake_type: [0, 0, 0] for mistake_type in row_types}
        for score in scores:
            counts = score["counts"]
            for mistake_type in row_types:
                aggregate_counts[mistake_type] = [
                    aggregate_value + score_value
                    for aggregate_value, score_value in zip(
                        aggregate_counts[mistake_type],
                        counts.get(mistake_type, (0, 0, 0)),
                    )
                ]

        def precision_recall_f_measure(
            true_positive_count: int,
            false_positive_count: int,
            false_negative_count: int,
        ) -> dict[str, float]:
            precision = (
                true_positive_count / (true_positive_count + false_positive_count)
                if true_positive_count + false_positive_count
                else (1.0 if false_negative_count == 0 else 0.0)
            )
            recall = (
                true_positive_count / (true_positive_count + false_negative_count)
                if true_positive_count + false_negative_count
                else 1.0
            )
            f_measure = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            return {
                "Precision": precision,
                "Recall": recall,
                "F-measure": f_measure,
            }

        timing_keys = [
            "pitch_detector_compute_time",
            "pitch_smoother_compute_time",
            "pitch_compute_time",
            "note_compute_time",
            "mistake_detection_compute_time",
            "mistake_check_compute_time",
        ]
        timing = {
            key: (
                float(np.mean([float(score.get(key, 0.0)) for score in scores]))
                if scores
                else 0.0
            )
            for key in timing_keys
        }
        rows: dict[str, dict[str, float]] = {
            mistake_type: {
                **precision_recall_f_measure(*aggregate_counts[mistake_type]),
                **timing,
            }
            for mistake_type in row_types
        }
        overall_counts = [
            sum(
                aggregate_counts[mistake_type][count_index]
                for mistake_type in overall_types
            )
            for count_index in range(3)
        ]
        rows["OVERALL"] = {
            **precision_recall_f_measure(*overall_counts),
            **timing,
        }
        return pd.DataFrame(rows).T
