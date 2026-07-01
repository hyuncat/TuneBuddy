from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pretty_midi
import ruptures as rpt

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording


class _BasicPitchOptionalBackendFilter(logging.Filter):
    _PREFIXES = (
        "Coremltools is not installed.",
        "tflite-runtime is not installed.",
        "onnxruntime is not installed.",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(
            record.getMessage().startswith(prefix)
            for prefix in self._PREFIXES
        )


def copy_note(note: Note, note_id: int | None = None) -> Note:
    """Return a detached Note copy for benchmark post-processing."""
    return Note(
        i=note.id if note_id is None else note_id,
        start_time=float(note.start_time),
        end_time=float(note.end_time),
        midi_num=list(note.midi_num),
        velocity=note.velocity,
        instrument=note.instrument,
    )


def clone_note_data(note_data: NoteData) -> NoteData:
    out = NoteData()
    for idx, note in enumerate(note_data.read(i=0, j=len(note_data.times))):
        out.write_note(copy_note(note, note_id=idx))
    return out


class BenchmarkNoteDetector:
    """Benchmark-only adapters around Attune's pitch data.

    The production detector remains the source of common thresholds and helper
    logic. This class adds comparison methods without changing the app path.
    """

    RUPTURES_ALGORITHMS = {"pelt", "kernelcpd", "bottomup", "window", "dynp"}
    _basic_pitch_unavailable_error: str | None = None
    _crepe_notes_unavailable_error: str | None = None
    _tony_pyin_unavailable_error: str | None = None

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.detector = recording.note_detector
        self.config = recording.config

    # ------------------------------------------------------------------ public
    def detect(
        self,
        method: str = "ruptures",
        refined_with_onsets: bool | None = None,
        refine_with_onsets: bool | None = None,
        onset_data=None,
        **kwargs: Any,
    ) -> NoteData:
        """Dispatch one benchmark method.

        `refine_with_onsets` is accepted as a compatibility alias; the benchmark
        output column uses `refined_with_onsets`.
        """
        if refined_with_onsets is None:
            refined_with_onsets = bool(refine_with_onsets)

        method = self._normalize_method(method)
        if method == "ruptures":
            notes = self.detect_ruptures(**kwargs)
        elif method == "crepe_notes":
            notes = self.detect_crepe_notes(onset_data=onset_data, **kwargs)
        elif method == "onset_only":
            notes = self.detect_onset_only(onset_data=onset_data, **kwargs)
        elif method == "onset_pitch_hybrid":
            notes = self.detect_onset_only(
                onset_data=onset_data,
                merge_adjacent=True,
                **kwargs,
            )
        elif method == "basic_pitch":
            notes = self.detect_basic_pitch(**kwargs)
        elif method == "tony_pyin":
            notes = self.detect_tony_pyin(**kwargs)
        elif method == "mt3":
            raise NotImplementedError(
                "MT3 is intentionally a placeholder: it needs a separate heavy "
                "model/runtime setup and is not part of the default benchmark."
            )
        else:
            raise ValueError(f"unknown note benchmark method: {method!r}")

        if refined_with_onsets:
            notes = self.refine_with_onsets(notes, onset_data=onset_data)
        return notes

    def refine_with_onsets(self, note_data: NoteData, onset_data=None) -> NoteData:
        onset_data = self._onset_data(onset_data)
        return self.detector.refine_with_spectral_onsets(
            clone_note_data(note_data),
            self.recording.pitch_data,
            onset_data,
        )

    # ------------------------------------------------------------ ruptures core
    def detect_ruptures(
        self,
        ruptures_algorithm: str = "pelt",
        algorithm: str | None = None,
        model: str = "l2",
        cost: str | None = None,
        pen: float | None = None,
        jump: int | None = None,
        width: int | None = None,
        n_bkps: int | None = None,
        oracle_note_count: bool = False,
        exclude_transitions: bool = True,
        do_transitions: bool | None = None,
        features: Sequence[str] | None = None,
        standardize_features: bool | None = None,
        merge_adjacent: bool = True,
        **_unused: Any,
    ) -> NoteData:
        if algorithm is not None:
            ruptures_algorithm = algorithm
        ruptures_algorithm = ruptures_algorithm.lower()
        if ruptures_algorithm not in self.RUPTURES_ALGORITHMS:
            raise ValueError(f"unknown ruptures algorithm: {ruptures_algorithm!r}")
        if do_transitions is not None:
            exclude_transitions = bool(do_transitions)
        self._prepare_transition_flags(exclude_transitions)

        min_size = self.detector._pelt_min_size_from_score()
        runs = self.detector._pelt_runs(
            self.recording.pitch_data,
            min_gap_frames=min_size,
        )
        if not runs:
            return NoteData()

        penalty = self.detector._pelt_penalty(min_size) if pen is None else float(pen)
        pelt_jump = self.detector._pelt_jump(jump)
        feature_names = tuple(features or self._default_features(cost or model))
        if standardize_features is None:
            standardize_features = len(feature_names) > 1

        budgets = self._run_breakpoint_budgets(
            runs,
            min_size=min_size,
            n_bkps=n_bkps,
            oracle_note_count=oracle_note_count,
            algorithm=ruptures_algorithm,
        )

        bkps_by_run: list[list[int]] = []
        for run, run_n_bkps in zip(runs, budgets):
            signal = self._feature_matrix(
                run,
                feature_names=feature_names,
                standardize=bool(standardize_features),
            )
            bkps_by_run.append(
                self._predict_bkps(
                    signal,
                    algorithm=ruptures_algorithm,
                    model=(cost or model),
                    min_size=min_size,
                    jump=pelt_jump,
                    penalty=penalty,
                    width=width,
                    n_bkps=run_n_bkps,
                )
            )

        return self._notes_from_run_breakpoints(
            runs,
            bkps_by_run,
            merge_adjacent=merge_adjacent,
        )

    def _predict_bkps(
        self,
        signal: np.ndarray,
        algorithm: str,
        model: str,
        min_size: int,
        jump: int,
        penalty: float,
        width: int | None,
        n_bkps: int | None,
    ) -> list[int]:
        n_frames = len(signal)
        if n_frames < 2 * min_size:
            return [n_frames]

        try:
            if algorithm == "kernelcpd":
                kernel = "linear" if model == "l2" else model
                algo = rpt.KernelCPD(
                    kernel=kernel,
                    min_size=min_size,
                    jump=jump,
                ).fit(signal)
                bkps = (
                    algo.predict(n_bkps=int(n_bkps))
                    if n_bkps is not None
                    else algo.predict(pen=penalty)
                )
            elif algorithm == "bottomup":
                algo = rpt.BottomUp(
                    model=model,
                    min_size=min_size,
                    jump=jump,
                ).fit(signal)
                bkps = (
                    algo.predict(n_bkps=int(n_bkps))
                    if n_bkps is not None
                    else algo.predict(pen=penalty)
                )
            elif algorithm == "window":
                window_width = self._window_width(width, min_size, n_frames)
                algo = rpt.Window(
                    width=window_width,
                    model=model,
                    min_size=min_size,
                    jump=jump,
                ).fit(signal)
                bkps = (
                    algo.predict(n_bkps=int(n_bkps))
                    if n_bkps is not None
                    else algo.predict(pen=penalty)
                )
            elif algorithm == "dynp":
                if not n_bkps:
                    return [n_frames]
                algo = rpt.Dynp(
                    model=model,
                    min_size=min_size,
                    jump=jump,
                ).fit(signal)
                bkps = algo.predict(n_bkps=int(n_bkps))
            else:
                algo = rpt.Pelt(
                    model=model,
                    min_size=min_size,
                    jump=jump,
                ).fit(signal)
                bkps = algo.predict(pen=penalty)
        except (rpt.exceptions.BadSegmentationParameters, ValueError):
            return [n_frames]

        return self._sanitize_bkps(bkps, n_frames)

    @staticmethod
    def _sanitize_bkps(bkps: Sequence[int], n_frames: int) -> list[int]:
        clean = sorted({int(b) for b in bkps if 0 < int(b) <= n_frames})
        if not clean or clean[-1] != n_frames:
            clean.append(n_frames)
        return clean

    @staticmethod
    def _window_width(width: int | None, min_size: int, n_frames: int) -> int:
        if n_frames <= 2:
            return 2
        candidate = int(width or max(2 * min_size, 5 * min_size))
        candidate = max(2, min(candidate, n_frames - 1))
        return candidate

    def _run_breakpoint_budgets(
        self,
        runs: Sequence[Sequence[Pitch]],
        min_size: int,
        n_bkps: int | None,
        oracle_note_count: bool,
        algorithm: str,
    ) -> list[int | None]:
        if algorithm != "dynp" and n_bkps is None:
            return [None] * len(runs)

        if oracle_note_count and n_bkps is None:
            expected_notes = self._expected_score_note_count()
            total_budget = max(0, expected_notes - len(runs))
        else:
            total_budget = max(0, int(n_bkps or 0))
        if total_budget <= 0:
            return [0] * len(runs)

        lengths = np.asarray([len(run) for run in runs], dtype=float)
        weights = lengths / max(float(lengths.sum()), 1.0)
        raw = weights * total_budget
        budgets = np.floor(raw).astype(int)
        for idx in np.argsort(raw - budgets)[::-1][: total_budget - int(budgets.sum())]:
            budgets[idx] += 1

        capped: list[int] = []
        for budget, run in zip(budgets, runs):
            max_bkps = max(0, len(run) // max(1, min_size) - 1)
            capped.append(min(int(budget), max_bkps))
        return capped

    def _expected_score_note_count(self) -> int:
        try:
            score_notes = self.recording.score_data.clipped_note_data(
                channel=self.recording.active_instrument
            )
        except (AttributeError, KeyError, TypeError):
            return 0
        return len(score_notes.read(i=0, j=len(score_notes.times), clean=True))

    @staticmethod
    def _default_features(model: str) -> tuple[str, ...]:
        if model == "normal":
            return ("pitch", "delta_pitch", "volume", "voiced_prob")
        return ("pitch",)

    def _feature_matrix(
        self,
        pitches: Sequence[Pitch],
        feature_names: Sequence[str],
        standardize: bool,
    ) -> np.ndarray:
        pitch_values = np.asarray(
            [float(self.detector._frame_pitch(p)) for p in pitches],
            dtype=float,
        )
        deltas = np.gradient(pitch_values) if len(pitch_values) > 1 else np.zeros_like(pitch_values)

        columns = []
        for name in feature_names:
            if name == "pitch":
                values = pitch_values
            elif name == "delta_pitch":
                values = deltas
            elif name == "volume":
                values = np.asarray([float(p.volume) for p in pitches], dtype=float)
            elif name == "voiced_prob":
                values = np.asarray([1.0 - float(p.unvoiced_prob) for p in pitches], dtype=float)
            elif name == "confidence":
                values = np.asarray([self._confidence(p) for p in pitches], dtype=float)
            elif name == "unvoiced_prob":
                values = np.asarray([float(p.unvoiced_prob) for p in pitches], dtype=float)
            else:
                raise ValueError(f"unknown ruptures feature: {name!r}")
            columns.append(values)

        signal = np.column_stack(columns).astype(float)
        if standardize:
            means = signal.mean(axis=0)
            stds = signal.std(axis=0)
            stds[stds < 1e-9] = 1.0
            signal = (signal - means) / stds
        return signal

    def _notes_from_run_breakpoints(
        self,
        runs: Sequence[Sequence[Pitch]],
        bkps_by_run: Sequence[Sequence[int]],
        merge_adjacent: bool,
    ) -> NoteData:
        note_data = NoteData()
        note_index = 0

        for pitches, bkps in zip(runs, bkps_by_run):
            prev = 0
            first_segment_in_run = True
            for bkp in bkps:
                end = min(int(bkp), len(pitches))
                if end <= prev:
                    continue

                segment = list(pitches[prev:end])
                midi_num = self.detector._pelt_segment_pitch(segment)
                start_time = self.detector._pelt_boundary_time(pitches, prev)
                end_time = self.detector._pelt_boundary_time(pitches, end)
                if end_time <= start_time:
                    prev = end
                    continue

                last = (
                    note_data.read_note(i=len(note_data.times) - 1)
                    if note_data.times and not first_segment_in_run
                    else None
                )
                if merge_adjacent and self.detector._same_note_pitch(last, midi_num):
                    last.end_time = end_time
                else:
                    note_data.write_note(
                        Note(
                            i=note_index,
                            start_time=start_time,
                            end_time=end_time,
                            midi_num=midi_num,
                        )
                    )
                    note_index += 1

                prev = end
                first_segment_in_run = False

        return self._reindex(note_data)

    # ------------------------------------------------------------ competitors
    def detect_crepe_notes(
        self,
        sensitivity: float = 0.001,
        min_duration: float = 0.03,
        min_velocity: int = 6,
        disable_splitting: bool = False,
        tuning_offset: float | bool = False,
        use_smoothing: bool = False,
        pitch_tracker: str = "crepe",
        detect_amplitude: bool = True,
        save_analysis_files: bool = False,
        **_unused: Any,
    ) -> NoteData:
        """External `crepe_notes` baseline.

        This intentionally calls the installed package instead of reimplementing
        the CREPE Notes paper over Attune PitchData. The package runs the chosen
        pitch tracker, applies its confidence-gradient postprocessor, writes MIDI,
        and we translate that MIDI back into Attune's NoteData.
        """
        if self.__class__._crepe_notes_unavailable_error is not None:
            raise RuntimeError(
                "CREPE Notes unavailable after a previous failed import: "
                f"{self.__class__._crepe_notes_unavailable_error}"
            )

        self._configure_external_model_environment()
        try:
            from crepe_notes.crepe_notes import process, run_pitch_tracker
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.__class__._crepe_notes_unavailable_error = detail
            raise RuntimeError(f"CREPE Notes unavailable: {detail}") from exc

        origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "attune-crepe-notes.wav"
            self._write_external_audio(audio_path)

            try:
                frequency, confidence = run_pitch_tracker(
                    audio_path,
                    tracker=pitch_tracker,
                )
                midi_path = process(
                    frequency,
                    confidence,
                    audio_path,
                    output_label=f"{pitch_tracker}.transcription",
                    sensitivity=float(sensitivity),
                    use_smoothing=bool(use_smoothing),
                    min_duration=float(min_duration),
                    min_velocity=int(min_velocity),
                    disable_splitting=bool(disable_splitting),
                    tuning_offset=tuning_offset,
                    use_cwd=False,
                    detect_amplitude=bool(detect_amplitude),
                    save_analysis_files=bool(save_analysis_files),
                    pitch_tracker=pitch_tracker,
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"CREPE Notes failed: {detail}") from exc

            return self._notedata_from_midi(Path(midi_path), origin=origin)

    def detect_onset_only(
        self,
        exclude_transitions: bool = False,
        merge_adjacent: bool = False,
        min_note_factor: float | None = None,
        onset_data=None,
        **_unused: Any,
    ) -> NoteData:
        """Segment from spectral onsets alone, then assign median pitch per span."""
        self._prepare_transition_flags(exclude_transitions)
        onset_data = self._onset_data(onset_data)
        if onset_data is None or len(onset_data) == 0:
            return NoteData()

        min_seconds = self.config.min_note_seconds(
            self.config.note_detection_min_note_factor
            if min_note_factor is None
            else min_note_factor
        )
        min_frames = self.config.min_note_pitch_frames(
            self.config.note_detection_min_note_factor
            if min_note_factor is None
            else min_note_factor
        )
        runs = self.detector._pelt_runs(
            self.recording.pitch_data,
            min_gap_frames=min_frames,
        )

        bkps_by_run = []
        for run in runs:
            run_start = self.detector._pelt_boundary_time(run, 0)
            run_end = self.detector._pelt_boundary_time(run, len(run))
            boundaries = []
            for onset_time in onset_data.read(run_start, run_end):
                if onset_time - run_start < min_seconds or run_end - onset_time < min_seconds:
                    continue
                idx = int(np.searchsorted([p.time for p in run], float(onset_time)))
                if min_frames <= idx <= len(run) - min_frames:
                    boundaries.append(idx)
            boundaries.append(len(run))
            bkps_by_run.append(sorted(set(boundaries)))

        return self._notes_from_run_breakpoints(
            runs,
            bkps_by_run,
            merge_adjacent=merge_adjacent,
        )

    def detect_basic_pitch(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float | None = None,
        **_unused: Any,
    ) -> NoteData:
        """External Basic Pitch baseline, loaded lazily."""
        if self.__class__._basic_pitch_unavailable_error is not None:
            raise RuntimeError(
                "Basic Pitch unavailable after a previous failed model load: "
                f"{self.__class__._basic_pitch_unavailable_error}"
            )

        audio_path = self._audio_path_for_external_model()
        self._configure_basic_pitch_environment()

        min_length = (
            float(minimum_note_length_ms)
            if minimum_note_length_ms is not None
            else 1000.0 * self.config.min_note_seconds(
                self.config.note_detection_min_note_factor
            )
        )
        model_path = None
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import AUDIO_SAMPLE_RATE, FFT_HOP, run_inference
            from basic_pitch.note_creation import model_output_to_notes

            model_path = self._basic_pitch_model_path(Path(ICASSP_2022_MODEL_PATH))
            model_output = run_inference(audio_path, model_path, debug_file=None)
            min_note_len = int(
                np.round(min_length / 1000.0 * (AUDIO_SAMPLE_RATE / FFT_HOP))
            )
            _, note_events = model_output_to_notes(
                model_output,
                onset_thresh=onset_threshold,
                frame_thresh=frame_threshold,
                min_note_len=min_note_len,
                min_freq=self.config.fmin,
                max_freq=self.config.fmax,
                include_pitch_bends=False,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if model_path is not None:
                detail = f"{detail} (model={model_path})"
            self.__class__._basic_pitch_unavailable_error = detail
            raise RuntimeError(f"Basic Pitch unavailable: {detail}") from exc

        note_data = NoteData()
        origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))
        for idx, event in enumerate(note_events):
            start, end, pitch, amplitude, *_ = event
            if end <= start:
                continue
            note_data.write_note(
                Note(
                    i=idx,
                    start_time=float(start) + origin,
                    end_time=float(end) + origin,
                    midi_num=[float(pitch)],
                    velocity=int(np.clip(round(float(amplitude) * 127), 1, 127)),
                )
            )
        return note_data

    def detect_tony_pyin(self, **_unused: Any) -> NoteData:
        """Tony-equivalent pYIN Vamp `notes` output via Sonic Annotator."""
        if self.__class__._tony_pyin_unavailable_error is not None:
            raise RuntimeError(
                "Tony pYIN unavailable after a previous failed run: "
                f"{self.__class__._tony_pyin_unavailable_error}"
            )

        audio_path = self._audio_path_for_external_model()
        try:
            from benchmarks.modules.pitch.TonyPyinRunner import TonyPyinRunner

            return TonyPyinRunner(self.config).detect_notes(audio_path)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.__class__._tony_pyin_unavailable_error = detail
            raise RuntimeError(f"Tony pYIN unavailable: {detail}") from exc

    @staticmethod
    def _configure_external_model_environment() -> None:
        mpl_dir = Path(tempfile.gettempdir()) / "attune-matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        BenchmarkNoteDetector._suppress_external_dependency_warnings()
        BenchmarkNoteDetector._patch_legacy_collections_abc()
        BenchmarkNoteDetector._patch_legacy_numpy_aliases()

    @staticmethod
    def _configure_basic_pitch_environment() -> None:
        BenchmarkNoteDetector._configure_external_model_environment()

    @staticmethod
    def _suppress_external_dependency_warnings() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        root_logger = logging.getLogger()
        if not any(
            isinstance(log_filter, _BasicPitchOptionalBackendFilter)
            for log_filter in root_logger.filters
        ):
            root_logger.addFilter(_BasicPitchOptionalBackendFilter())

    @staticmethod
    def _patch_legacy_collections_abc() -> None:
        """Expose moved ABCs for older optional deps under Python 3.12.

        PyPI madmom still imports MutableSequence from collections during
        CREPE Notes repeated-note splitting. Python 3.12 only exposes those ABCs
        from collections.abc, so patch the old names before upstream imports.
        """
        import collections
        import collections.abc

        for name in (
            "Callable",
            "Iterable",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "Sequence",
        ):
            if not hasattr(collections, name) and hasattr(collections.abc, name):
                setattr(collections, name, getattr(collections.abc, name))

    @staticmethod
    def _patch_legacy_numpy_aliases() -> None:
        """Expose NumPy aliases removed after 1.20 for older madmom releases."""
        aliases = {
            "bool": bool,
            "complex": complex,
            "float": float,
            "int": int,
            "object": object,
            "str": str,
        }
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)

    @classmethod
    def _basic_pitch_model_path(cls, default_model_path: Path) -> Path:
        for suffix, modules in (
            (".tflite", ("tensorflow", "tflite_runtime")),
            (".onnx", ("onnxruntime",)),
            (".mlpackage", ("coremltools",)),
        ):
            candidate = default_model_path.with_suffix(suffix)
            if candidate.exists() and any(
                cls._module_available(m) for m in modules
            ):
                return candidate
        return default_model_path

    @staticmethod
    def _module_available(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    # --------------------------------------------------------------- utilities
    @staticmethod
    def _normalize_method(method: str) -> str:
        aliases = {
            "pelt": "ruptures",
            "ruptures_pelt": "ruptures",
            "tony": "tony_pyin",
            "pyin-tony": "tony_pyin",
            "pyin_tony": "tony_pyin",
            "tony-pyin": "tony_pyin",
            "tony_pyin": "tony_pyin",
            "crepe": "crepe_notes",
            "crepe-notes": "crepe_notes",
            "basic-pitch": "basic_pitch",
            "onsets": "onset_only",
        }
        return aliases.get(method, method)

    def _prepare_transition_flags(self, exclude_transitions: bool) -> None:
        if exclude_transitions:
            self.detector.detect_transitions(self.recording.pitch_data)
            return
        for pitch in self.recording.pitch_data.data:
            if pitch is not None:
                pitch.is_transition = False

    def _onset_data(self, onset_data=None):
        if onset_data is not None:
            return onset_data
        existing = getattr(self.recording, "onset_data", None)
        if existing is not None:
            return existing
        return self.detector._detect_spectral_onsets()

    @staticmethod
    def _confidence(pitch: Pitch) -> float:
        if pitch is None or not pitch.candidates:
            return 0.0
        return float(pitch.candidates[0][1])

    def _trim_note_edges_by_volume(
        self,
        note_data: NoteData,
        floor_ratio: float,
    ) -> NoteData:
        frame_dt = self.config.h1 / self.config.sr
        min_seconds = self.config.min_note_seconds(
            self.config.note_detection_min_note_factor
        )
        out = NoteData()

        for idx, note in enumerate(note_data.read(i=0, j=len(note_data.times))):
            frames = self.recording.pitch_data.read(
                start_time=note.start_time,
                end_time=note.end_time,
                clean=False,
            )
            valid = [p for p in frames if p is not None]
            if not valid:
                out.write_note(copy_note(note, note_id=idx))
                continue

            volumes = np.asarray([float(p.volume) for p in valid], dtype=float)
            peak = float(volumes.max(initial=0.0))
            if peak <= 0:
                out.write_note(copy_note(note, note_id=idx))
                continue
            keep = np.flatnonzero(volumes >= peak * floor_ratio)
            if keep.size == 0:
                out.write_note(copy_note(note, note_id=idx))
                continue

            start = float(valid[int(keep[0])].time)
            end = float(valid[int(keep[-1])].time) + frame_dt
            if end - start < min_seconds:
                out.write_note(copy_note(note, note_id=idx))
                continue

            trimmed = copy_note(note, note_id=idx)
            trimmed.start_time = start
            trimmed.end_time = end
            out.write_note(trimmed)
        return out

    def _audio_path_for_external_model(self) -> str:
        candidate = getattr(self.recording, "audio_filepath", None)
        if candidate is not None and Path(candidate).exists():
            return str(candidate)

        import soundfile as sf

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        sf.write(
            handle.name,
            self.recording.audio_data.read_all(),
            int(self.recording.audio_data.sr),
        )
        return handle.name

    def _write_external_audio(self, audio_path: Path) -> None:
        import soundfile as sf

        sf.write(
            str(audio_path),
            self.recording.audio_data.read_all(),
            int(self.recording.audio_data.sr),
        )

    @staticmethod
    def _notedata_from_midi(midi_path: Path, origin: float = 0.0) -> NoteData:
        pretty_midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        raw_notes = sorted(
            (
                pretty_midi_note
                for instrument in pretty_midi_data.instruments
                for pretty_midi_note in instrument.notes
            ),
            key=lambda pretty_midi_note: (
                pretty_midi_note.start,
                pretty_midi_note.end,
                pretty_midi_note.pitch,
            ),
        )
        note_data = NoteData()
        for idx, pretty_midi_note in enumerate(raw_notes):
            if pretty_midi_note.end <= pretty_midi_note.start:
                continue
            start_time = float(pretty_midi_note.start) + origin
            while start_time in note_data.data:
                start_time = float(np.nextafter(start_time, float("inf")))
            note_data.write_note(
                Note(
                    i=idx,
                    start_time=start_time,
                    end_time=float(pretty_midi_note.end) + origin,
                    midi_num=[int(pretty_midi_note.pitch)],
                    velocity=int(pretty_midi_note.velocity),
                )
            )
        return note_data

    @staticmethod
    def _reindex(note_data: NoteData) -> NoteData:
        out = NoteData()
        for idx, note in enumerate(note_data.read(i=0, j=len(note_data.times))):
            copied = copy_note(note, note_id=idx)
            out.write_note(copied)
        return out
