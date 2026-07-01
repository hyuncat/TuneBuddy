from __future__ import annotations

import json
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import mir_eval
import numpy as np
import numpy.typing as npt
import pretty_midi

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

from app_logic.NoteData import Note, NoteData  # noqa: E402
from app_logic.midi.ScoreData import ScoreData  # noqa: E402
from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from app_logic.user.ds.Recording import Recording  # noqa: E402
from benchmarks.modules.note.NoteDetectionBaselines import (  # noqa: E402
    BenchmarkNoteDetector,
    clone_note_data,
)
from benchmarks.modules.pitch.PitchBenchmarker import PathLike, PitchBenchmarker  # noqa: E402

NoteMethodConfig: TypeAlias = dict[str, Any]
NOTE_CACHE_VERSION: int = 1


def _expanded_ruptures_methods() -> dict[str, NoteMethodConfig]:
    bases: list[tuple[str, NoteMethodConfig]] = [
        ("pelt-l2", dict(method="ruptures", ruptures_algorithm="pelt", model="l2")),
        ("pelt-l1", dict(method="ruptures", ruptures_algorithm="pelt", model="l1")),
        ("pelt-rbf", dict(method="ruptures", ruptures_algorithm="pelt", model="rbf")),
        (
            "pelt-normal",
            dict(
                method="ruptures",
                ruptures_algorithm="pelt",
                model="normal",
                features=("pitch", "delta_pitch", "volume", "voiced_prob"),
            ),
        ),
        ("pelt-rank", dict(method="ruptures", ruptures_algorithm="pelt", model="rank")),
        ("kernelcpd-rbf", dict(method="ruptures", ruptures_algorithm="kernelcpd", model="rbf")),
        ("window-l2", dict(method="ruptures", ruptures_algorithm="window", model="l2")),
        ("window-rbf", dict(method="ruptures", ruptures_algorithm="window", model="rbf")),
        ("bottomup-l2", dict(method="ruptures", ruptures_algorithm="bottomup", model="l2")),
        ("bottomup-rbf", dict(method="ruptures", ruptures_algorithm="bottomup", model="rbf")),
        (
            "dynp-l2-oracle",
            dict(
                method="ruptures",
                ruptures_algorithm="dynp",
                model="l2",
                oracle_note_count=True,
            ),
        ),
    ]

    methods: dict[str, NoteMethodConfig] = {}
    for base_label, base_config in bases:
        for exclude_transitions in (False, True):
            for refined in (False, True):
                transition_label = "exclude-transitions" if exclude_transitions else "keep-transitions"
                onset_label = "onset-refined" if refined else "raw"
                label = f"{base_label}__{transition_label}__{onset_label}"
                config = dict(base_config)
                config.update(
                    exclude_transitions=exclude_transitions,
                    refined_with_onsets=refined,
                    benchmark_group="ruptures",
                    base_method=base_label,
                    current_notedetector=(
                        base_label == "pelt-l2"
                        and exclude_transitions
                        and not refined
                    ),
                )
                methods[label] = config
    return methods


def _competitor_methods() -> dict[str, NoteMethodConfig]:
    return {
        "crepe-notes": dict(
            method="crepe_notes",
            refined_with_onsets=False,
            external_baseline=True,
            uses_onsets=True,
            benchmark_group="external",
            base_method="crepe-notes",
        ),
        "onset-only": dict(
            method="onset_only",
            exclude_transitions=False,
            refined_with_onsets=False,
            uses_onsets=True,
            benchmark_group="onset",
            base_method="onset-only",
        ),
        "onset-pitch-hybrid": dict(
            method="onset_pitch_hybrid",
            exclude_transitions=False,
            refined_with_onsets=False,
            uses_onsets=True,
            benchmark_group="competitor",
            base_method="onset-pitch-hybrid",
        ),
        "basic-pitch": dict(
            method="basic_pitch",
            refined_with_onsets=False,
            external_baseline=True,
            benchmark_group="external",
            base_method="basic-pitch",
        ),
        "tony-pyin": dict(
            method="tony_pyin",
            refined_with_onsets=False,
            external_baseline=True,
            benchmark_group="external",
            base_method="tony-pyin",
        ),
    }


def _default_note_methods(
    all_methods: dict[str, NoteMethodConfig],
) -> dict[str, NoteMethodConfig]:
    """Practical default set for interactive benchmark runs.

    The full matrix intentionally includes expensive oracle/kernel/external
    baselines. Keep the default useful enough to compare the main families while
    avoiding methods that can take minutes per 20k-frame etude.
    """
    labels = [
        "pelt-l2__keep-transitions__raw",
        "pelt-l2__keep-transitions__onset-refined",
        "pelt-l2__exclude-transitions__raw",
        "pelt-l2__exclude-transitions__onset-refined",
        "pelt-l1__exclude-transitions__raw",
        "pelt-l1__exclude-transitions__onset-refined",
        "pelt-rank__exclude-transitions__raw",
        "pelt-rank__exclude-transitions__onset-refined",
        "bottomup-l2__exclude-transitions__raw",
        "bottomup-l2__exclude-transitions__onset-refined",
        "crepe-notes",
        "onset-only",
        "onset-pitch-hybrid",
        "basic-pitch",
        "tony-pyin",
    ]
    return {label: all_methods[label] for label in labels}


_ALL_NOTE_METHODS = {
    **_expanded_ruptures_methods(),
    **_competitor_methods(),
}


class NoteBenchmarker(PitchBenchmarker):
    ETUDE_DATASETS: ClassVar[list[str]] = ["kayser", "wohlfahrt"]
    ETUDE_DATASET_ALIASES: ClassVar[dict[str, str]] = {
        "wolfhart": "wohlfahrt",
    }

    ALL_NOTE_METHODS: ClassVar[dict[str, NoteMethodConfig]] = _ALL_NOTE_METHODS
    NOTE_METHODS: ClassVar[dict[str, NoteMethodConfig]] = _default_note_methods(
        _ALL_NOTE_METHODS
    )
    OPTIONAL_NOTE_METHODS: ClassVar[dict[str, NoteMethodConfig]] = {
        "mt3": dict(
            method="mt3",
            refined_with_onsets=False,
            external_baseline=True,
            benchmark_group="external",
            base_method="mt3",
        ),
    }

    def __init__(self, onset_tolerance: float = 0.05) -> None:
        super().__init__()
        self.onset_tolerance = onset_tolerance  # mir-eval specific onset tolerance for note evals

    @staticmethod
    def notedata_to_intervals(note_data: NoteData, config, drop_rests: bool=True
                              ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert NoteData to mir_eval-style intervals and frequencies."""
        notes = note_data.read(i=0, j=len(note_data.times)) if note_data.times else []
        rows, frequencies_hz = [], []
        for note in notes:
            if drop_rests and (not note.midi_num or note.midi_num[0] == -1):
                continue
            if note.end_time <= note.start_time:
                continue
            rows.append([note.start_time, note.end_time])
            frequencies_hz.append(config.midi_to_freq(note.midi_num[0]))
        if not rows:
            return np.zeros((0, 2)), np.zeros((0,))
        return np.asarray(rows, dtype=float), np.asarray(frequencies_hz, dtype=float)

    def iter_etudes(self, dataset: str) -> Iterator[tuple[str, Path]]:
        """Returns iterable of (title, midi_path) for each etude in the given dataset."""
        source_dataset = self.ETUDE_DATASET_ALIASES.get(dataset, dataset)
        dataset_dir = self.DATASETS / "violin-etudes" / source_dataset
        midi_dir = dataset_dir / "midi"
        search_dir = midi_dir if midi_dir.exists() else dataset_dir
        for mid in sorted(search_dir.glob("*.mid")):
            yield mid.stem, mid

    @staticmethod
    def etude_corpus_dir_for_midi(midi_path: PathLike) -> Path:
        return PitchBenchmarker.corpus_dir_for_midi(midi_path)

    @staticmethod
    def detect_notes(
        recording: Recording, method: str = "pelt",
        model: str = "l2", do_transitions: bool = True,
        jump: int | None = None,
        refine_with_onsets: bool | None = None,
        onset_data=None,
        **kwargs: Any,
    ) -> NoteData:
        method_config = dict(kwargs)
        method_config["method"] = method
        if method == "pelt":
            method_config["method"] = "ruptures"
            method_config.setdefault("ruptures_algorithm", "pelt")
        method_config.setdefault("model", model)
        if do_transitions is not None:
            method_config.setdefault("exclude_transitions", bool(do_transitions))
        if jump is not None:
            method_config.setdefault("jump", jump)
        if refine_with_onsets is not None:
            method_config["refined_with_onsets"] = bool(refine_with_onsets)

        notes = BenchmarkNoteDetector(recording).detect(
            onset_data=onset_data,
            **method_config,
        )
        recording.note_data = notes
        return notes

    @staticmethod
    def prepare_for_note_detection(
        recording: Recording, 
        resize_score_to_pitch: bool=True,
        detect_transitions: bool=True,
    ) -> Recording:
        if detect_transitions:
            recording.detect_transitions()
        if resize_score_to_pitch: # sets min_note_length from this
            recording.resize_score(
                to_span="pitch",
                include_transitions=not detect_transitions,
            )
        return recording

    def detect_notes_timed(
        self, recording: Recording, method: str="pelt", 
        model: str="l2", do_transitions: bool=True,
        jump: int | None = None,
        refine_with_onsets: bool | None = None,
        onset_data=None,
        **kwargs: Any,
    ) -> tuple[NoteData, float]:
        start = time.perf_counter()
        notes = self.detect_notes(
            recording,
            method=method,
            model=model,
            do_transitions=do_transitions,
            jump=jump,
            refine_with_onsets=refine_with_onsets,
            onset_data=onset_data,
            **kwargs,
        )
        return notes, time.perf_counter() - start

    def analyze_recording(
        self, recording: Recording,  method: str="pelt",
        model: str="l2", truncate: bool=False,
        refine_with_onsets: bool | None = None,
        **_unused,
    ) -> dict[str, float]:
        recording.reset_analysis()
        self.prepare_for_note_detection(
            recording,
            resize_score_to_pitch=True,
            detect_transitions=True,
        )
        _, note_compute_time = self.detect_notes_timed(
            recording,
            method=method,
            model=model,
            do_transitions=False,
            refine_with_onsets=refine_with_onsets,
        )
        if recording.note_data.times:
            recording.resize_score(to_span="note")
        if truncate:
            recording.trim_end(mark_unsaved=False)

        return {
            "note_compute_time": note_compute_time,
        }

    @staticmethod
    def _latency_offset(
        reference_onsets: npt.NDArray[np.float64],
        estimated_onsets: npt.NDArray[np.float64],
    ) -> float:
        """Robust constant detector latency: the median signed gap from each
        estimated onset to its NEAREST reference onset. This is a pure
        translation (no scale), so it can't drift interior notes — it only
        absorbs a fixed pitch/note-detector lag. Outlier onsets wash out in the
        median."""
        ref = np.sort(np.asarray(reference_onsets, dtype=float))
        est = np.asarray(estimated_onsets, dtype=float)
        if ref.size == 0 or est.size == 0:
            return 0.0
        if ref.size == 1:
            return float(ref[0] - np.median(est))
        idx = np.clip(np.searchsorted(ref, est), 1, ref.size - 1)
        left, right = ref[idx - 1], ref[idx]
        nearest = np.where(np.abs(est - left) <= np.abs(est - right), left, right)
        return float(np.median(nearest - est))

    @staticmethod
    def _trim_boundary_notes(detected: NoteData, reference: NoteData) -> None:
        """NoteData form of `_trim_boundaries`: in-place clamp the FIRST/LAST
        VOICED detected notes to the reference (MIDI) note DURATIONS, anchoring
        the reliable inner edge and trimming the swelled outer one (synth
        attack/release). Used by the mistake pipeline, which aligns/checks the
        NoteData directly (the note benchmark trims the mir_eval intervals).
          - first note: start := end - ref_first_duration
          - last note:  end   := start + ref_last_duration
        No global shift — the detected take keeps its absolute time so resize /
        mistake detection still anchor it against the score."""
        def voiced(nd: NoteData) -> list[Note]:
            notes = nd.read(i=0, j=len(nd.times)) if nd.times else []
            return [n for n in notes if n.midi_num and n.midi_num[0] != -1]

        det_voiced, ref_voiced = voiced(detected), voiced(reference)
        if not det_voiced or not ref_voiced:
            return
        det_voiced[0].start_time = det_voiced[0].end_time - (
            ref_voiced[0].end_time - ref_voiced[0].start_time
        )
        det_voiced[-1].end_time = det_voiced[-1].start_time + (
            ref_voiced[-1].end_time - ref_voiced[-1].start_time
        )
        # the first note's start_time (its dict key) changed -> re-key the whole
        # NoteData so the sorted-times index stays in sync.
        all_notes = detected.read(i=0, j=len(detected.times))
        detected.load_data({n.start_time: n for n in all_notes})

    # --- PELT-L2 note cache (mirrors the pitch cache: <corpus>/note_data/) ---
    def note_cache_path(self, corpus_dir: PathLike, track_id: str) -> Path:
        safe_track_id = track_id.replace("/", "_")
        return Path(corpus_dir) / "note_data" / f"{safe_track_id}.note.json"

    @staticmethod
    def _note_to_payload(note: Note) -> dict[str, Any]:
        return {
            "id": int(note.id),
            "start_time": float(note.start_time),
            "end_time": float(note.end_time),
            "midi_num": [float(m) for m in note.midi_num],
            "velocity": (None if note.velocity is None else int(note.velocity)),
            "instrument": note.instrument,
        }

    @staticmethod
    def _note_from_payload(payload: dict[str, Any]) -> Note:
        return Note(
            i=int(payload["id"]),
            start_time=float(payload["start_time"]),
            end_time=float(payload["end_time"]),
            midi_num=[float(m) for m in payload["midi_num"]],
            velocity=payload.get("velocity"),
            instrument=payload.get("instrument"),
        )

    def save_note_data(
        self, note_data: NoteData, cache_path: PathLike, metadata: dict[str, Any]=None,
    ) -> Path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        notes = note_data.read(i=0, j=len(note_data.times)) if note_data.times else []
        payload = {
            "version": NOTE_CACHE_VERSION,
            "metadata": metadata or {},
            "notes": [self._note_to_payload(n) for n in notes],
        }
        with open(cache_path, "w") as fh:
            json.dump(payload, fh)
        return cache_path

    def load_note_data(self, cache_path: PathLike) -> tuple[NoteData, dict[str, Any]]:
        with open(cache_path) as fh:
            payload = json.load(fh)
        if payload.get("version") != NOTE_CACHE_VERSION:
            raise ValueError(f"Unsupported note cache version: {payload.get('version')}")
        note_data = NoteData()
        for note_payload in payload.get("notes", []):
            note_data.write_note(self._note_from_payload(note_payload))
        return note_data, dict(payload.get("metadata") or {})

    def load_or_detect_notes(
        self, recording: Recording, cache_path: PathLike,
        model: str="l2", write_cache: bool=True,
    ) -> tuple[NoteData, float]:
        """Load cached PELT notes if present, else detect (PELT) and cache the
        RAW notes. Caching is decoupled from trimming: callers apply
        `_trim_boundary_notes` after, so the cache holds the honest detector
        output. Returns (notes, note_compute_time) — the cached compute time on a
        hit, so timing reporting stays meaningful."""
        cache_path = Path(cache_path)
        if cache_path.exists():
            recording.note_data, metadata = self.load_note_data(cache_path)
            return recording.note_data, float(metadata.get("note_compute_time", 0.0))
        notes, note_compute_time = self.detect_notes_timed(
            recording, model=model, do_transitions=False,
        )
        if write_cache:
            self.save_note_data(
                notes, cache_path,
                metadata={"note_compute_time": note_compute_time, "model": model},
            )
        return notes, note_compute_time

    @staticmethod
    def _trim_boundaries(
        est_intervals: npt.NDArray[np.float64],
        ref_intervals: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Force the FIRST/LAST detected notes to the reference (MIDI) note
        DURATIONS, anchoring the reliable inner edge and trimming the outer one.
        The synth's attack/release swells those two notes past their true length
        even with reverb/chorus off, and they have no neighbour to mask them.
          - first note: keep its END, pull its START to end - ref_first_duration
          - last note:  keep its START, pull its END to start + ref_last_duration
        Pure per-note trim (interior notes untouched); the caller's pad then
        slides everything up so all start times are >= 0. Returns a COPY so the
        shared estimated array is never mutated."""
        if est_intervals.shape[0] == 0 or ref_intervals.shape[0] == 0:
            return est_intervals
        out = est_intervals.copy()
        first_end, last_start = out[0, 1], out[-1, 0]  # capture before mutating
        out[0, 0] = first_end - (ref_intervals[0, 1] - ref_intervals[0, 0])
        out[-1, 1] = last_start + (ref_intervals[-1, 1] - ref_intervals[-1, 0])
        return out

    @staticmethod
    def _normalized_method_name(method_config: NoteMethodConfig) -> str:
        method = method_config.get("method", "pelt")
        if method == "pelt":
            return "ruptures"
        if method == "basic-pitch":
            return "basic_pitch"
        if method == "tony-pyin":
            return "tony_pyin"
        if method == "crepe-notes":
            return "crepe_notes"
        return str(method)

    @classmethod
    def _method_uses_onsets(cls, method_config: NoteMethodConfig) -> bool:
        method = cls._normalized_method_name(method_config)
        return bool(
            method_config.get("uses_onsets", False)
            or method_config.get("refined_with_onsets", False)
            or method_config.get("refine_with_onsets", False)
            or method in {"crepe_notes", "onset_only", "onset_pitch_hybrid"}
        )

    @classmethod
    def _method_needs_attune_onsets(cls, method_config: NoteMethodConfig) -> bool:
        method = cls._normalized_method_name(method_config)
        return bool(
            method_config.get("uses_attune_onsets", False)
            or method_config.get("refined_with_onsets", False)
            or method_config.get("refine_with_onsets", False)
            or method in {"onset_only", "onset_pitch_hybrid"}
        )

    @classmethod
    def _base_detection_config(cls, method_config: NoteMethodConfig) -> NoteMethodConfig:
        config = dict(method_config)
        for key in (
            "benchmark_group",
            "base_method",
            "current_notedetector",
            "external_baseline",
            "uses_onsets",
            "uses_attune_onsets",
        ):
            config.pop(key, None)
        config["method"] = cls._normalized_method_name(config)
        if config["method"] == "ruptures":
            config.setdefault("ruptures_algorithm", "pelt")
        config["refined_with_onsets"] = False
        config.pop("refine_with_onsets", None)
        return config

    @staticmethod
    def _jsonable_method_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return [NoteBenchmarker._jsonable_method_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): NoteBenchmarker._jsonable_method_value(v)
                for k, v in sorted(value.items())
            }
        return repr(value)

    @classmethod
    def _method_cache_key(cls, method_config: NoteMethodConfig) -> str:
        base = cls._base_detection_config(method_config)
        payload = {
            key: cls._jsonable_method_value(value)
            for key, value in sorted(base.items())
            if key != "onset_data"
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def _method_row_metadata(
        cls,
        label: str,
        method_config: NoteMethodConfig,
    ) -> dict[str, Any]:
        method = cls._normalized_method_name(method_config)
        return {
            "method_label": label,
            "detector_family": method,
            "benchmark_group": method_config.get("benchmark_group", method),
            "base_method": method_config.get("base_method", label),
            "ruptures_algorithm": method_config.get(
                "ruptures_algorithm",
                "pelt" if method == "ruptures" else None,
            ),
            "ruptures_model": method_config.get("model"),
            "ruptures_cost": method_config.get("cost", method_config.get("model")),
            "refined_with_onsets": bool(
                method_config.get(
                    "refined_with_onsets",
                    method_config.get("refine_with_onsets", False),
                )
            ),
            "transition_excluding": (
                bool(method_config.get("exclude_transitions"))
                if "exclude_transitions" in method_config
                else None
            ),
            "oracle_note_count": bool(method_config.get("oracle_note_count", False)),
            "external_baseline": bool(method_config.get("external_baseline", False)),
            "uses_onsets": cls._method_uses_onsets(method_config),
            "current_notedetector": bool(method_config.get("current_notedetector", False)),
        }

    def _detect_notes_for_method(
        self,
        recording: Recording,
        method_config: NoteMethodConfig,
        method_cache: dict[str, tuple[NoteData, float]],
        onset_data=None,
    ) -> tuple[NoteData, dict[str, float]]:
        cache_key = self._method_cache_key(method_config)
        if cache_key not in method_cache:
            base_config = self._base_detection_config(method_config)
            base_notes, base_time = self.detect_notes_timed(
                recording,
                onset_data=onset_data,
                **base_config,
            )
            method_cache[cache_key] = (clone_note_data(base_notes), base_time)

        base_notes, base_time = method_cache[cache_key]
        notes = clone_note_data(base_notes)
        refinement_time = 0.0
        if bool(
            method_config.get(
                "refined_with_onsets",
                method_config.get("refine_with_onsets", False),
            )
        ):
            start = time.perf_counter()
            notes = BenchmarkNoteDetector(recording).refine_with_onsets(
                notes,
                onset_data=onset_data,
            )
            refinement_time = time.perf_counter() - start

        recording.note_data = notes
        return notes, {
            "base_note_compute_time": base_time,
            "onset_refinement_compute_time": refinement_time,
            "note_compute_time": base_time + refinement_time,
        }

    @staticmethod
    def _failed_method_row(error: Exception) -> dict[str, Any]:
        return {
            "Precision": np.nan,
            "Recall": np.nan,
            "F-measure": np.nan,
            "Average Overlap Ratio": np.nan,
            "Estimated Notes": 0,
            "note_compute_time": np.nan,
            "base_note_compute_time": np.nan,
            "onset_refinement_compute_time": np.nan,
            "error": f"{type(error).__name__}: {error}",
        }

    def bench_note_track(
        self, midi_path: PathLike, onset_tolerance: float=None,
        methods: dict[str, NoteMethodConfig]=None,
        align: str="identity", latency_align: bool=True,
        trim_boundaries: bool=True, write_note_cache: bool=True,
        refine_with_onsets: bool=False,
        progress: bool=False,
    ) -> dict[str, Any]:
        """Benchmark note detection on one synthesized etude.

        `align` controls how the detected notes are placed against the reference:
          - "identity" (default): the WAV is rendered FROM the reference MIDI, so
            audio-time already EQUALS MIDI-time and the true alignment is the
            identity. No tempo rescale runs (it would only re-fit the timebase
            from two noisy detected endpoints and inject drift). With
            `latency_align` we shift the detected notes by a single constant to
            absorb detector lag. This isolates note-DETECTION quality.
          - "resize": runs the app's resize_score(to_span="note") pipeline
            (perform.py::analyze) per method to score the full alignment-aware
            flow as the app actually runs it.
        `trim_boundaries` clamps the first/last detected notes to the reference
        MIDI durations (see _trim_boundaries) to neutralize the synth's
        attack/release swell on those two notes.
        `write_note_cache` auto-saves the RAW PELT-L2 notes to
        <corpus>/note_data/<stem>.note.json so the mistake benchmarker can reuse
        them instead of re-running PELT.
        """
        effective_onset_tolerance = (
            self.onset_tolerance if onset_tolerance is None else onset_tolerance
        )
        effective_methods = self._effective_note_methods(methods, refine_with_onsets)
        # create the Recording with the prereq data from the etude MIDI
        midi_path = Path(midi_path)
        score_data = OneInstrumentScoreData(midi_path)
        config = self.config_for(*self.range_from_midi(score_data.midi_numbers))
        recording = self.recording_for(config, score_data=score_data)
        wav_path = self.synth_midi(midi_path) # synthesize the MIDI to a WAV for pitch detection
        recording.audio_data = AudioData(
            audio_filepath=str(wav_path),
            config=recording.config,
        )
        recording.audio_filepath = wav_path
        pitch_timing = self.load_or_detect_pitches(
            recording,
            cache_path=self.pitch_cache_path(
                self.etude_corpus_dir_for_midi(midi_path),
                midi_path.stem,
            ),
            smooth=True, write_cache=True,
        )
        # The pitch pre-resize rescales the score to the take, so it only belongs
        # in the alignment-aware "resize" mode. Identity keeps the score at its
        # MIDI tempo (the synth timebase) and its config-seeded min_note_length.
        self.prepare_for_note_detection(
            recording,
            resize_score_to_pitch=(align == "resize"),
            detect_transitions=True,
        )

        def read_reference():
            return self.notedata_to_intervals(
                recording.score_data.clipped_note_data(
                    channel=recording.active_instrument
                ),
                recording.config,
            )

        # Reference note count is invariant under resize (it only retimes notes).
        reference_intervals, reference_pitches = read_reference()
        out: dict[str, Any] = {"_reference_notes": len(reference_intervals)}
        method_cache: dict[str, tuple[NoteData, float]] = {}
        shared_onset_data = None
        shared_onset_time = 0.0

        def method_onsets(method_config: NoteMethodConfig):
            nonlocal shared_onset_data, shared_onset_time
            if not self._method_needs_attune_onsets(method_config):
                return None
            if shared_onset_data is None:
                start = time.perf_counter()
                shared_onset_data = recording.note_detector._detect_spectral_onsets()
                shared_onset_time = time.perf_counter() - start
            return shared_onset_data

        for label, method_config in effective_methods.items():
            method_start = time.perf_counter()
            if progress:
                print(f"    note method: {label}", flush=True)
            row_base = {
                **self._method_row_metadata(label, method_config),
                **pitch_timing,
            }
            onset_data = method_onsets(method_config)
            try:
                notes, note_timing = self._detect_notes_for_method(
                    recording,
                    method_config,
                    method_cache=method_cache,
                    onset_data=onset_data,
                )
            except Exception as exc:  # noqa: BLE001 -- keep one baseline from killing a run
                out[label] = {
                    **row_base,
                    **self._failed_method_row(exc),
                    "onset_compute_time": shared_onset_time if onset_data is not None else 0.0,
                }
                if progress:
                    print(
                        f"      error after {time.perf_counter() - method_start:.1f}s: {exc}",
                        flush=True,
                    )
                continue

            note_compute_time = note_timing["note_compute_time"]
            # auto-cache the RAW production PELT-L2 notes (before any resize
            # shifts them) so the mistake benchmarker can skip re-detecting.
            if (
                write_note_cache
                and row_base["current_notedetector"]
                and method_config.get("model") == "l2"
            ):
                self.save_note_data(
                    notes,
                    self.note_cache_path(
                        self.etude_corpus_dir_for_midi(midi_path),
                        midi_path.stem,
                    ),
                    metadata={"note_compute_time": note_compute_time, "model": "l2"},
                )
            if align == "resize":
                # Re-align the score to the take's detected NOTE onsets, mirroring
                # perform.py::analyze's second resize. The score is the reference,
                # so re-read it AFTER the resize (note count is unchanged).
                if notes.times:
                    recording.resize_score(to_span="note")
                reference_intervals, reference_pitches = read_reference()
            estimated_intervals, estimated_pitches = self.notedata_to_intervals(
                notes,
                recording.config,
            )
            if len(estimated_intervals) == 0:
                out[label] = {
                    **row_base,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F-measure": 0.0,
                    "Average Overlap Ratio": 0.0,
                    "Estimated Notes": 0,
                    "onset_compute_time": shared_onset_time if onset_data is not None else 0.0,
                    **note_timing,
                    "error": None,
                }
                if progress:
                    print(
                        f"      done in {time.perf_counter() - method_start:.1f}s "
                        f"(0 notes)",
                        flush=True,
                    )
                continue
            # Local copies for the eval so the shared reference (read once in
            # identity mode) is never mutated across the method loop.
            ref_iv, ref_pi = reference_intervals, reference_pitches
            est_iv, est_pi = estimated_intervals, estimated_pitches
            if align == "identity" and latency_align:
                # one constant shift (translation, not scale) to absorb detector lag
                est_iv = est_iv + self._latency_offset(ref_iv[:, 0], est_iv[:, 0])
            if trim_boundaries:
                # clamp first/last detected notes to the MIDI durations (synth
                # attack/release overshoots them); pad below shifts starts >= 0.
                est_iv = self._trim_boundaries(est_iv, ref_iv)
            # detector lag makes the latency offset NEGATIVE, which can push the
            # first onset below 0; mir_eval forbids negative interval times. Matching
            # and overlap are translation-invariant, so slide BOTH sides up by a
            # common pad — the scores are unchanged. (ref_iv may be empty.)
            mins = [0.0, float(est_iv.min())]
            if ref_iv.size:
                mins.append(float(ref_iv.min()))
            pad = -min(mins)
            if pad:
                ref_iv = ref_iv + pad
                est_iv = est_iv + pad
            precision, recall, f_measure, average_overlap_ratio = (
                mir_eval.transcription.precision_recall_f1_overlap(
                    ref_iv,
                    ref_pi,
                    est_iv,
                    est_pi,
                    onset_tolerance=effective_onset_tolerance,
                    offset_ratio=None,
                )
            )
            out[label] = {
                **row_base,
                "Precision": precision,
                "Recall": recall,
                "F-measure": f_measure,
                "Average Overlap Ratio": average_overlap_ratio,
                "Estimated Notes": len(est_iv),
                "onset_compute_time": shared_onset_time if onset_data is not None else 0.0,
                **note_timing,
                "error": None,
            }
            if progress:
                print(
                    f"      done in {time.perf_counter() - method_start:.1f}s "
                    f"(F={f_measure:.3f}, notes={len(est_iv)})",
                    flush=True,
                )
        return out

    def bench_note_dataset(
        self,
        dataset: str,
        max_tracks: int | None = None,
        onset_tolerance: float | None = None,
        methods: dict[str, NoteMethodConfig] = None,
        refine_with_onsets: bool = False,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        write_note_cache: bool = True,
        verbose: bool = True,
        write: bool = False,
    ) -> pd.DataFrame:
        import pandas as pd

        effective_methods = self._effective_note_methods(methods, refine_with_onsets)
        tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
        rows = []
        for i, (title, midi_path) in enumerate(tracks):
            track_result = self.bench_note_track(
                midi_path,
                onset_tolerance=onset_tolerance,
                methods=effective_methods,
                refine_with_onsets=False,
                align=align,
                latency_align=latency_align,
                trim_boundaries=trim_boundaries,
                write_note_cache=write_note_cache,
                progress=verbose,
            )
            reference_note_count = track_result.pop("_reference_notes")
            for label, method_result in track_result.items():
                row = {
                    "dataset": dataset,
                    "track": title,
                    "method": label,
                    "Reference Notes": reference_note_count,
                }
                row.update(method_result)
                rows.append(row)
            if verbose:
                print(
                    f"[{dataset}] {i+1}/{len(tracks)} {title[:32]:32s} "
                    f"reference={reference_note_count}"
                )
        df = pd.DataFrame(rows)
        if write:
            self.write_dataset_result(df, "note", dataset, index=False)
        return df

    @staticmethod
    def _methods_refine_with_onsets(methods: dict[str, NoteMethodConfig]) -> bool:
        return any(
            bool(
                method_config.get(
                    "refined_with_onsets",
                    method_config.get("refine_with_onsets", False),
                )
            )
            for method_config in methods.values()
        )

    def _effective_note_methods(
        self,
        methods: dict[str, NoteMethodConfig] | None,
        refine_with_onsets: bool,
    ) -> dict[str, NoteMethodConfig]:
        effective = {
            label: dict(method_config)
            for label, method_config in (methods or self.NOTE_METHODS).items()
        }
        if refine_with_onsets and methods is not None:
            for method_config in effective.values():
                method_config["refined_with_onsets"] = True
                method_config.pop("refine_with_onsets", None)
        return effective


class OneInstrumentScoreData(ScoreData):
    """ScoreData wrapper that flattens all different channels to a single instrument."""

    def __init__(
        self,
        midi_path: PathLike,
        collapse_simultaneous: bool = True,
    ) -> None:
        source_path = Path(midi_path)
        note_data = self.pretty_midi_to_notedata(
            pretty_midi.PrettyMIDI(str(source_path)),
            collapse_simultaneous=collapse_simultaneous,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            normalized_path = Path(temp_dir) / source_path.name
            PitchBenchmarker.notedata_to_pm(note_data).write(str(normalized_path))
            super().__init__(normalized_path)
        self.filepath = source_path
        self.title = source_path.stem

    @property
    def note_data(self) -> NoteData:
        return self.clipped_note_data(channel=self.active_instrument)

    @property
    def midi_numbers(self) -> list[int]:
        return self.notedata_midi_numbers(self.note_data)

    @staticmethod
    def pretty_midi_to_notedata(
        pretty_midi_data: pretty_midi.PrettyMIDI,
        collapse_simultaneous: bool = True,
    ) -> NoteData:
        raw_notes = sorted(
            (
                pretty_midi_note
                for instrument in pretty_midi_data.instruments
                for pretty_midi_note in instrument.notes
            ),
            key=lambda pretty_midi_note: (
                pretty_midi_note.start,
                -pretty_midi_note.pitch,
            ),
        )
        note_data, note_index, last_start_time = NoteData(), 0, None
        for pretty_midi_note in raw_notes:
            if (
                collapse_simultaneous
                and last_start_time is not None
                and abs(pretty_midi_note.start - last_start_time) < 1e-3
            ):
                continue
            note_data.write_note(
                Note(
                    i=note_index,
                    start_time=float(pretty_midi_note.start),
                    end_time=float(pretty_midi_note.end),
                    midi_num=[int(pretty_midi_note.pitch)],
                    velocity=int(pretty_midi_note.velocity),
                )
            )
            last_start_time = pretty_midi_note.start
            note_index += 1
        return note_data

    @staticmethod
    def notedata_midi_numbers(note_data: NoteData) -> list[int]:
        return [
            note_data.data[t].midi_num[0]
            for t in note_data.times
            if note_data.data[t].midi_num and note_data.data[t].midi_num[0] != -1
        ]
