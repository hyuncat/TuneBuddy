from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

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

from benchmarks.paths import ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()

from app_logic.NoteData import Note, NoteData  # noqa: E402
from app_logic.midi.ScoreData import ScoreData  # noqa: E402
from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from app_logic.user.ds.PitchData import Pitch  # noqa: E402
from app_logic.user.ds.Recording import Recording  # noqa: E402
from benchmarks.modules.pitch.PitchBenchmarker import PathLike, PitchBenchmarker  # noqa: E402
from benchmarks.note.algorithms.BasicPitchDetector import BasicPitchDetector  # noqa: E402
from benchmarks.note.algorithms.CrepeNotesDetector import CrepeNotesDetector  # noqa: E402
from benchmarks.note.algorithms.RupturesDetector import RupturesDetector  # noqa: E402
from benchmarks.note.algorithms.TonyDetector import TonyDetector  # noqa: E402

NOTE_CACHE_VERSION: int = 1


class NoteBenchmarker(PitchBenchmarker):
    ETUDE_DATASETS: ClassVar[list[str]] = ["kayser", "wohlfahrt"]
    ETUDE_DATASET_ALIASES: ClassVar[dict[str, str]] = {"wolfhart": "wohlfahrt"}

    #: label -> (RupturesDetector method, kwargs). The bench flags
    #: (exclude_transitions / split_runs / refine_with_onsets /
    #: drop_transition_notes) shape these rows only; "slope-aware windowed" is
    #: detect_windowed with explicit transition exclusion, not a separate
    #: detector. Penalties are cost-aware (see RupturesDetector._penalty); a
    #: kwargs "penalty_scale" here would pin a per-method tuned scale, and the
    #: CLI --penalty-scale multiplies on top for sweeps.
    RUPTURES_METHODS: ClassVar[dict[str, tuple[str, dict[str, Any]]]] = {
        "pelt-l1": ("detect_pelt", {"model": "l1"}),
        "pelt-l2": ("detect_pelt", {"model": "l2"}),  # production configuration
        "pelt-rbf": ("detect_pelt", {"model": "rbf"}),
        "kernelcpd-linear": ("detect_kernel", {"kernel": "linear"}),
        "kernelcpd-gaussian": ("detect_kernel", {"kernel": "rbf"}),
        "kernelcpd-cosine": ("detect_kernel", {"kernel": "cosine"}),
        "bottomup": ("detect_bottom_up", {"model": "l2"}),
        "window": ("detect_windowed", {"model": "l2"}),
    }
    EXTERNAL_METHODS: ClassVar[dict[str, type]] = {
        "basic-pitch": BasicPitchDetector,
        "tony": TonyDetector,
        "crepe-notes": CrepeNotesDetector,
    }
    ALL_METHODS: ClassVar[list[str]] = [*RUPTURES_METHODS, *EXTERNAL_METHODS]

    REALTIME_COL: ClassVar[str] = "Audio(s)/Compute(s)"
    COMPUTE_COL: ClassVar[str] = "Compute Time(s)"
    SUMMARY_METRICS: ClassVar[list[str]] = [
        "Accuracy", "Precision", "Recall", "F-measure",
        "Average Overlap Ratio", "False Positive Rate",
    ]
    #: columns kept (and their order) in the note result CSVs. Pitch-detection
    #: time is intentionally absent — only the note detector's own cost is scored.
    NOTE_RESULT_COLUMNS: ClassVar[list[str]] = [
        "Track ID", "Split", "Track", "Ensemble", "Instrument", "Voice",
        "Method", "Accuracy", "Precision", "Recall", "F-measure",
        "Average Overlap Ratio", REALTIME_COL, COMPUTE_COL,
        "Reference Notes", "Estimated Notes", "False Positives",
        "False Positive Rate",
        "exclude_transitions", "split_runs", "refine_with_onsets",
        "drop_transition_notes", "penalty_scale", "error",
    ]

    def __init__(self, onset_tolerance: float = 0.05) -> None:
        super().__init__()
        self.onset_tolerance = onset_tolerance  # mir_eval onset match tolerance (s)

    # ---------------------------------------------------------------- corpora
    def iter_etudes(self, dataset: str) -> Iterator[tuple[str, Path]]:
        source_dataset = self.ETUDE_DATASET_ALIASES.get(dataset, dataset)
        dataset_dir = self.DATASETS / "violin-etudes" / source_dataset
        midi_dir = dataset_dir / "midi"
        search_dir = midi_dir if midi_dir.exists() else dataset_dir
        for mid in sorted(search_dir.glob("*.mid")):
            yield mid.stem, mid

    @staticmethod
    def etude_corpus_dir_for_midi(midi_path: PathLike) -> Path:
        return PitchBenchmarker.corpus_dir_for_midi(midi_path)

    # ----------------------------------------------------------- method runs
    @staticmethod
    def pitch_runs(
        recording: Recording, exclude_transitions: bool, split_runs: bool,
    ) -> list[list[Pitch]]:
        """Pre-split the pitch track for the ruptures detectors.

        Transition (slide) frames are dropped from the signal when excluded —
        that alone makes the windowed method slope-aware. Without run splitting
        the whole track (voiced or not) is one signal, so the change-point
        search has to find silence boundaries itself."""
        pitches = recording.pitch_data.data
        if exclude_transitions:
            recording.transition_detector.detect_transitions(pitches)
        if split_runs:
            return recording.note_detector.get_pitch_runs(
                pitches,
                include_transitions=not exclude_transitions,
            )
        run = [
            p for p in pitches
            if p is not None and (not exclude_transitions or not p.is_transition)
        ]
        return [run] if run else []

    def detect_method_notes(
        self, recording: Recording, method: str, *,
        exclude_transitions: bool = False, split_runs: bool = True,
        refine_with_onsets: bool = False,
        drop_transition_notes: bool = False,
        penalty_scale: float = 1.0,
        runs: list[list[Pitch]] | None = None,
        onset_times: list[float] | None = None,
    ) -> tuple[NoteData, float]:
        """Run one method -> (notes, compute seconds). ``runs``/``onset_times``
        let bench_note_track share the prep across the ruptures rows (their
        build time is charged there)."""
        start = time.perf_counter()
        if method in self.EXTERNAL_METHODS:
            notes = self.EXTERNAL_METHODS[method](recording).detect()
        else:
            func, kwargs = self.RUPTURES_METHODS[method]
            kwargs = dict(kwargs)
            # a sweep's scale multiplies any per-method tuned scale in the matrix
            scale = penalty_scale * float(kwargs.pop("penalty_scale", 1.0))
            if runs is None:
                runs = self.pitch_runs(recording, exclude_transitions, split_runs)
            notes = getattr(RupturesDetector(recording), func)(
                runs, penalty_scale=scale, **kwargs,
            )
            if refine_with_onsets:
                if onset_times is None:
                    onset_times = recording.onset_detector.detect().times
                notes = recording.note_detector.refine_with_onsets(notes, onset_times)
            if drop_transition_notes:
                notes = self.drop_mostly_transition_notes(notes, recording.pitch_data)
        recording.note_data = notes
        return notes, time.perf_counter() - start

    @staticmethod
    def drop_mostly_transition_notes(
        note_data: NoteData, pitch_data, max_transition_frac: float = 0.5,
    ) -> NoteData:
        """Final pass: segmentation keeps transition (slide) frames as boundary
        evidence, so a long slide can still come out as its own segment — drop
        notes whose voiced frames are mostly flagged is_transition."""
        notes = note_data.read(i=0, j=len(note_data.times)) if note_data.times else []
        out = NoteData()
        for note in notes:
            frames = pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=True,
            )
            transitions = sum(1 for p in frames if p.is_transition)
            if frames and transitions / len(frames) > max_transition_frac:
                continue
            note.id = len(out.times)
            out.write_note(note)
        return out

    @staticmethod
    def detect_recording_notes(recording: Recording) -> NoteData:
        """Run the production note pipeline exactly as the app does."""
        recording.detect_notes()
        return recording.note_data

    def detect_recording_notes_timed(self, recording: Recording) -> tuple[NoteData, float]:
        start = time.perf_counter()
        notes = self.detect_recording_notes(recording)
        return notes, time.perf_counter() - start

    # ------------------------------------------------------------- evaluation
    @staticmethod
    def notedata_to_intervals(
        note_data: NoteData, config, drop_rests: bool = True,
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

    @staticmethod
    def _latency_offset(
        reference_onsets: npt.NDArray[np.float64],
        estimated_onsets: npt.NDArray[np.float64],
    ) -> float:
        """Robust constant detector latency: the median signed gap from each
        estimated onset to its NEAREST reference onset. A pure translation (no
        scale), so it can't drift interior notes — it only absorbs a fixed
        detector lag; outlier onsets wash out in the median."""
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
    def _trim_boundaries(
        est_intervals: npt.NDArray[np.float64],
        ref_intervals: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Clamp the FIRST/LAST detected notes to the reference note DURATIONS,
        anchoring the reliable inner edge: the synth's attack/release swells
        those two notes past their true length and they have no neighbour to
        mask them. Returns a COPY so shared arrays are never mutated."""
        if est_intervals.shape[0] == 0 or ref_intervals.shape[0] == 0:
            return est_intervals
        out = est_intervals.copy()
        first_end, last_start = out[0, 1], out[-1, 0]  # capture before mutating
        out[0, 0] = first_end - (ref_intervals[0, 1] - ref_intervals[0, 0])
        out[-1, 1] = last_start + (ref_intervals[-1, 1] - ref_intervals[-1, 0])
        return out

    @staticmethod
    def _trim_boundary_notes(detected: NoteData, reference: NoteData) -> None:
        """NoteData form of `_trim_boundaries` for the mistake pipeline, which
        aligns/checks the NoteData directly. In-place; no global shift, so the
        detected take keeps its absolute time."""
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
        # the first note's start_time (its dict key) changed -> re-key so the
        # sorted-times index stays in sync
        all_notes = detected.read(i=0, j=len(detected.times))
        detected.load_data({n.start_time: n for n in all_notes})

    def _interval_metrics(
        self,
        ref_iv: npt.NDArray[np.float64], ref_pi: npt.NDArray[np.float64],
        est_iv: npt.NDArray[np.float64], est_pi: npt.NDArray[np.float64],
        *, align: str, latency_align: bool, trim_boundaries: bool,
        onset_tolerance: float,
    ) -> dict[str, Any]:
        """Constant latency shift + boundary trim + non-negative pad, then
        onset-based note matching (offsets ignored)."""
        if len(est_iv) == 0 or len(ref_iv) == 0:
            return self._metrics_row(tp=0, n_est=len(est_iv), n_ref=len(ref_iv))
        if align == "identity" and latency_align:
            # one constant shift (translation, not scale) to absorb detector lag
            est_iv = est_iv + self._latency_offset(ref_iv[:, 0], est_iv[:, 0])
        if trim_boundaries:
            est_iv = self._trim_boundaries(est_iv, ref_iv)
        # the latency shift can push onsets below 0 and mir_eval forbids
        # negative times; matching is translation-invariant, so slide both up
        pad = -min(0.0, float(est_iv.min()), float(ref_iv.min()))
        if pad:
            ref_iv, est_iv = ref_iv + pad, est_iv + pad
        matching = mir_eval.transcription.match_notes(
            ref_iv, ref_pi, est_iv, est_pi,
            onset_tolerance=onset_tolerance, offset_ratio=None,
        )
        aor = (
            float(mir_eval.transcription.average_overlap_ratio(ref_iv, est_iv, matching))
            if matching else 0.0
        )
        return self._metrics_row(
            tp=len(matching), n_est=len(est_iv), n_ref=len(ref_iv), aor=aor,
        )

    @staticmethod
    def _metrics_row(tp: int, n_est: int, n_ref: int, aor: float = 0.0) -> dict[str, Any]:
        fp, fn = n_est - tp, n_ref - tp
        precision = tp / n_est if n_est else 0.0
        recall = tp / n_ref if n_ref else 0.0
        return {
            # MIREX-style overall accuracy: TP / (TP + FP + FN)
            "Accuracy": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "Precision": precision,
            "Recall": recall,
            "F-measure": mir_eval.util.f_measure(precision, recall),
            "Average Overlap Ratio": aor,
            "Reference Notes": n_ref,
            "Estimated Notes": n_est,
            "False Positives": fp,
            # per reference note, so track length doesn't dominate the average
            "False Positive Rate": fp / n_ref if n_ref else float("nan"),
        }

    def _timing_columns(self, compute_time: float, audio_seconds: float) -> dict[str, float]:
        realtime = (
            audio_seconds / compute_time
            if compute_time and compute_time > 0 and np.isfinite(audio_seconds)
            else float("nan")
        )
        return {self.REALTIME_COL: float(realtime), self.COMPUTE_COL: float(compute_time)}

    def _flag_columns(
        self, method: str, *, exclude_transitions: bool, refine_with_onsets: bool,
        split_runs: bool = True, drop_transition_notes: bool = False,
        penalty_scale: float = 1.0,
    ) -> dict[str, Any]:
        if method in self.EXTERNAL_METHODS:  # bench flags shape ruptures rows only
            return {
                "exclude_transitions": None, "split_runs": None,
                "refine_with_onsets": None, "drop_transition_notes": None,
                "penalty_scale": None,
            }
        return {
            "exclude_transitions": exclude_transitions,
            "split_runs": split_runs,
            "refine_with_onsets": refine_with_onsets,
            "drop_transition_notes": drop_transition_notes,
            "penalty_scale": float(penalty_scale),
        }

    def _failed_row(self, base: dict[str, Any], n_ref: int, audio_seconds: float,
                    exc: Exception) -> dict[str, Any]:
        return {
            **base,
            **{metric: np.nan for metric in self.SUMMARY_METRICS},
            "Reference Notes": n_ref,
            "Estimated Notes": 0,
            **self._timing_columns(float("nan"), audio_seconds),
            "error": f"{type(exc).__name__}: {exc}",
        }

    # --------------------------------------------------------- track scoring
    def _reference_intervals(
        self, recording: Recording,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.notedata_to_intervals(
            recording.score_data.clipped_note_data(channel=recording.active_instrument),
            recording.config,
        )

    @staticmethod
    def prepare_for_note_detection(
        recording: Recording, resize_score_to_pitch: bool = True,
    ) -> Recording:
        recording.transition_detector.detect_transitions(recording.pitch_data.data)
        if resize_score_to_pitch:  # sets the tempo the min-note heuristics read
            recording.resize_score(to_span="pitch", include_transitions=False)
        recording.update_min_note_length()
        return recording

    def _prepare_note_recording(
        self,
        primary_path: PathLike,
        reference_path: PathLike,
        align: str = "identity",
        needs_audio: bool = True,
    ) -> tuple[Recording, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Build a Recording with cached pitches + score reference for one etude.

        The etude WAV is synthesized from the reference MIDI; pitch data comes
        from the shared pyin_smooth cache, so pitch detection time is NOT
        charged to the note benchmark. CocoNoteBenchmarker overrides this with
        its own audio/reference/pitch-cache loading (``needs_audio`` lets it
        skip the waveform for pure change-point methods; the etude path always
        has audio from synthesis)."""
        midi_path = Path(reference_path)
        score_data = OneInstrumentScoreData(midi_path)
        config = self.config_for(*self.range_from_midi(score_data.midi_numbers))
        recording = self.recording_for(config, score_data=score_data)
        wav_path = self.synth_midi(midi_path)
        recording.audio_data = AudioData(audio_filepath=str(wav_path), config=recording.config)
        recording.audio_filepath = wav_path
        self.load_or_detect_pitches(
            recording,
            cache_path=self.pitch_cache_path(
                self.etude_corpus_dir_for_midi(midi_path), midi_path.stem,
            ),
            smooth=True, write_cache=True, verbose=self.algorithm_verbose,
        )
        # the pitch pre-resize rescales the score to the take, so it only
        # belongs in the alignment-aware "resize" mode; identity keeps the
        # score at its MIDI tempo (the synth timebase)
        self.prepare_for_note_detection(recording, resize_score_to_pitch=(align == "resize"))
        ref_iv, ref_pi = self._reference_intervals(recording)
        return recording, ref_iv, ref_pi, self._audio_seconds(wav_path)

    def _audio_seconds(self, wav_path: PathLike | None) -> float:
        if wav_path is None:
            return float("nan")
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            return float(info.frames) / float(info.samplerate)
        except Exception:  # noqa: BLE001
            return float("nan")

    # --------------------------------------------------- per-method note cache
    def note_cache_dir(self, primary_path: PathLike, reference_path: PathLike) -> Path:
        """Corpus directory the detected-notes caches live under (note_data/)."""
        return self.etude_corpus_dir_for_midi(reference_path)

    def note_cache_track_id(self, primary_path: PathLike) -> str:
        """Track identity for the cache filename — MUST be unique within
        note_cache_dir (etude midi stems are; coco wav stems like "1_violin"
        repeat across tracks, so CocoNoteBenchmarker overrides this)."""
        return Path(primary_path).stem

    def method_note_cache_path(
        self, primary_path: PathLike, reference_path: PathLike, method: str, *,
        align: str, exclude_transitions: bool, split_runs: bool,
        refine_with_onsets: bool, drop_transition_notes: bool,
        penalty_scale: float,
    ) -> Path:
        """Cache key = method + everything that shapes its output, so a flag
        change can never serve stale notes. External methods key on the method
        alone (the bench flags don't shape them)."""
        tokens = [self.note_cache_track_id(primary_path), method]
        if method not in self.EXTERNAL_METHODS:
            if align != "identity":
                tokens.append(align)
            if exclude_transitions:
                tokens.append("noslides")
            if not split_runs:
                tokens.append("nosplit")
            if refine_with_onsets:
                tokens.append("onsets")
            if drop_transition_notes:
                tokens.append("droptrans")
            if penalty_scale != 1.0:
                tokens.append(f"pen{penalty_scale:g}")
        return self.note_cache_path(
            self.note_cache_dir(primary_path, reference_path), ".".join(tokens),
        )

    def _read_note_cache(self, cache_path: Path | None) -> tuple[NoteData, float] | None:
        if cache_path is None or not Path(cache_path).exists():
            return None
        try:
            notes, metadata = self.load_note_data(cache_path)
        except Exception:  # noqa: BLE001 -- corrupt/partial caches get recomputed
            return None
        return notes, float(metadata.get("compute_seconds", float("nan")))

    def _write_note_cache(
        self, cache_path: Path | None, notes: NoteData, compute_time: float,
        method: str, flags: dict[str, Any],
    ) -> None:
        if cache_path is None:
            return
        self.save_note_data(notes, cache_path, metadata={
            "method": method,
            "compute_seconds": float(compute_time),
            # the key analyze_recording (mistake benchmark) reads
            "note_compute_time": float(compute_time),
            **flags,
        })

    def score_note_track(
        self,
        primary_path: PathLike,
        reference_path: PathLike,
        method: str,
        *,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        onset_tolerance: float | None = None,
        exclude_transitions: bool = False,
        split_runs: bool = True,
        refine_with_onsets: bool = False,
        drop_transition_notes: bool = False,
        penalty_scale: float = 1.0,
        use_note_cache: bool = True,
        refresh_note_cache: bool = False,
    ) -> dict[str, Any]:
        """Detect + score ONE method on ONE track -> one flat row (the unit the
        parallel runner distributes).

        `align` controls how detected notes are placed against the reference:
          - "identity" (default): audio-time already equals MIDI-time (the WAV
            is rendered from the reference MIDI), so only a constant latency
            shift is applied — isolates note-DETECTION quality.
          - "resize": runs the app's resize_score(to_span="note") flow to score
            the alignment-aware pipeline as the app runs it.

        Detected notes are cached per method+flags (`method_note_cache_path`);
        a cache hit skips detection (the row keeps the ORIGINAL compute time)
        and, for external methods, the waveform load too."""
        onset_tolerance = self.onset_tolerance if onset_tolerance is None else onset_tolerance
        detect_flags: dict[str, Any] = {
            "exclude_transitions": exclude_transitions,
            "split_runs": split_runs,
            "refine_with_onsets": refine_with_onsets,
            "drop_transition_notes": drop_transition_notes,
            "penalty_scale": penalty_scale,
        }
        cache_path = (
            self.method_note_cache_path(
                primary_path, reference_path, method, align=align, **detect_flags,
            )
            if use_note_cache and self.use_cache
            else None
        )
        cached = None if refresh_note_cache else self._read_note_cache(cache_path)
        needs_audio = cached is None and (
            method in self.EXTERNAL_METHODS or refine_with_onsets
        )
        recording, ref_iv, ref_pi, audio_seconds = self._prepare_note_recording(
            primary_path, reference_path, align, needs_audio=needs_audio,
        )
        base = {
            "Method": method,
            **self._flag_columns(method, **detect_flags),
        }
        try:
            if cached is not None:
                notes, compute_time = cached
                recording.note_data = notes
            else:
                notes, compute_time = self.detect_method_notes(
                    recording, method, **detect_flags,
                )
                self._write_note_cache(
                    cache_path, notes, compute_time, method,
                    {"align": align, **detect_flags},
                )
        except Exception as exc:  # noqa: BLE001 -- isolate a bad method/track
            return self._failed_row(base, len(ref_iv), audio_seconds, exc)

        if align == "resize" and notes.times:
            # re-align the score to the detected notes, mirroring the app's
            # second resize; the score is the reference, so re-read it after
            recording.resize_score(to_span="note")
            ref_iv, ref_pi = self._reference_intervals(recording)
        est_iv, est_pi = self.notedata_to_intervals(notes, recording.config)
        return {
            **base,
            **self._interval_metrics(
                ref_iv, ref_pi, est_iv, est_pi,
                align=align, latency_align=latency_align,
                trim_boundaries=trim_boundaries, onset_tolerance=onset_tolerance,
            ),
            **self._timing_columns(compute_time, audio_seconds),
            "error": None,
        }

    def bench_note_track(
        self,
        midi_path: PathLike,
        methods: list[str] | None = None,
        *,
        exclude_transitions: bool = False,
        split_runs: bool = True,
        refine_with_onsets: bool = False,
        drop_transition_notes: bool = False,
        penalty_scale: float = 1.0,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        onset_tolerance: float | None = None,
        use_note_cache: bool = True,
        refresh_note_cache: bool = False,
        progress: bool = False,
    ) -> list[dict[str, Any]]:
        """All methods on one synthesized etude; one row per method.

        The bench flags shape only the ruptures rows; run splitting and
        onsets are prepared once and shared across them (their cost is added to
        each row so timings match a standalone run). Cached notes are reused
        per method+flags, in which case the prep is skipped too."""
        onset_tolerance = self.onset_tolerance if onset_tolerance is None else onset_tolerance
        methods = list(methods or self.ALL_METHODS)
        detect_flags: dict[str, Any] = {
            "exclude_transitions": exclude_transitions,
            "split_runs": split_runs,
            "refine_with_onsets": refine_with_onsets,
            "drop_transition_notes": drop_transition_notes,
            "penalty_scale": penalty_scale,
        }
        recording, ref_iv, ref_pi, audio_seconds = self._prepare_note_recording(
            midi_path, midi_path, align, needs_audio=True,
        )
        track_id = Path(midi_path).stem
        runs, onset_times, prep_time = None, None, 0.0
        rows: list[dict[str, Any]] = []
        for method in methods:
            if progress:
                print(f"    note method: {method}", flush=True)
            base = {
                "Track ID": track_id,
                "Method": method,
                **self._flag_columns(method, **detect_flags),
            }
            cache_path = (
                self.method_note_cache_path(
                    midi_path, midi_path, method, align=align, **detect_flags,
                )
                if use_note_cache and self.use_cache
                else None
            )
            try:
                cached = None if refresh_note_cache else self._read_note_cache(cache_path)
                if cached is not None:
                    notes, compute_time = cached
                    recording.note_data = notes
                else:
                    if method in self.RUPTURES_METHODS and runs is None:
                        start = time.perf_counter()
                        runs = self.pitch_runs(recording, exclude_transitions, split_runs)
                        if refine_with_onsets:
                            onset_times = recording.onset_detector.detect().times
                        prep_time = time.perf_counter() - start
                    notes, compute_time = self.detect_method_notes(
                        recording, method,
                        runs=runs, onset_times=onset_times, **detect_flags,
                    )
                    if method in self.RUPTURES_METHODS:
                        compute_time += prep_time
                    self._write_note_cache(
                        cache_path, notes, compute_time, method,
                        {"align": align, **detect_flags},
                    )
            except Exception as exc:  # noqa: BLE001 -- keep one method from killing the track
                rows.append(self._failed_row(base, len(ref_iv), audio_seconds, exc))
                continue
            if align == "resize":
                if notes.times:
                    recording.resize_score(to_span="note")
                ref_iv, ref_pi = self._reference_intervals(recording)
            est_iv, est_pi = self.notedata_to_intervals(notes, recording.config)
            rows.append({
                **base,
                **self._interval_metrics(
                    ref_iv, ref_pi, est_iv, est_pi,
                    align=align, latency_align=latency_align,
                    trim_boundaries=trim_boundaries, onset_tolerance=onset_tolerance,
                ),
                **self._timing_columns(compute_time, audio_seconds),
                "error": None,
            })
        return rows

    def bench_note_dataset(
        self,
        dataset: str,
        max_tracks: int | None = None,
        methods: list[str] | None = None,
        *,
        exclude_transitions: bool = False,
        split_runs: bool = True,
        refine_with_onsets: bool = False,
        drop_transition_notes: bool = False,
        penalty_scale: float = 1.0,
        align: str = "identity",
        latency_align: bool = True,
        trim_boundaries: bool = True,
        onset_tolerance: float | None = None,
        use_note_cache: bool = True,
        refresh_note_cache: bool = False,
        verbose: bool = True,
        write: bool = False,
    ) -> "pd.DataFrame":
        import pandas as pd

        tracks = self._limit(list(self.iter_etudes(dataset)), max_tracks)
        rows: list[dict[str, Any]] = []
        for i, (title, midi_path) in enumerate(tracks):
            track_rows = self.bench_note_track(
                midi_path, methods,
                exclude_transitions=exclude_transitions,
                split_runs=split_runs,
                refine_with_onsets=refine_with_onsets,
                drop_transition_notes=drop_transition_notes,
                penalty_scale=penalty_scale,
                align=align, latency_align=latency_align,
                trim_boundaries=trim_boundaries,
                onset_tolerance=onset_tolerance,
                use_note_cache=use_note_cache,
                refresh_note_cache=refresh_note_cache,
                progress=verbose,
            )
            for row in track_rows:
                row["dataset"] = dataset
            rows.extend(track_rows)
            if verbose:
                print(f"[{dataset}] {i + 1}/{len(tracks)} {title[:40]}")
        df = pd.DataFrame(rows)
        if write:
            self.write_dataset_result(self.display_note_columns(df), "note", dataset, index=False)
        return df

    # ----------------------------------------------------------- results IO
    def note_result_csv_path(self, method: str, dataset: str) -> Path:
        safe_dataset = dataset.replace("/", "_")
        safe_method = method.replace("/", "_")
        return self.RESULTS / "note" / "raw_outputs" / safe_method / f"{safe_dataset}.csv"

    @property
    def note_summary_csv_path(self) -> Path:
        return self.RESULTS / "note" / "note_benchmarks.csv"

    def display_note_columns(self, df: "pd.DataFrame") -> "pd.DataFrame":
        out = df.rename(columns={
            "track_id": "Track ID", "method": "Method", "split": "Split",
            "track": "Track", "ensemble": "Ensemble", "instrument": "Instrument",
            "voice": "Voice",
        })
        # f0_voice is a pitch-benchmark key (0-based f0-pickle column) with no
        # meaning against the stems_midi reference
        out = out.drop(columns=["dataset", "f0_voice"], errors="ignore")
        ordered = [c for c in self.NOTE_RESULT_COLUMNS if c in out.columns]
        return out[ordered + [c for c in out.columns if c not in ordered]]

    def write_note_result(self, df: "pd.DataFrame", method: str, dataset: str) -> Path:
        out_path = self.note_result_csv_path(method, dataset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.display_note_columns(df).to_csv(out_path, index=False)
        return out_path

    # --------------- production-note cache (used by the mistake benchmarker)
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
        self, note_data: NoteData, cache_path: PathLike, metadata: dict[str, Any] = None,
    ) -> Path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        notes = note_data.read(i=0, j=len(note_data.times)) if note_data.times else []
        payload = {
            "version": NOTE_CACHE_VERSION,
            "metadata": metadata or {},
            "notes": [self._note_to_payload(n) for n in notes],
        }
        # tmp + replace so a crash mid-write can't leave a truncated cache
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        with open(tmp_path, "w") as fh:
            json.dump(payload, fh)
        tmp_path.replace(cache_path)
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
