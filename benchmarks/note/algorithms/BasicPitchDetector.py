from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import ClassVar

import numpy as np

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.Recording import Recording
from benchmarks.note.algorithms.external import (
    audio_path_for,
    configure_environment,
    min_note_seconds,
    suppress_output,
)


class BasicPitchDetector:
    """External Basic Pitch (Spotify) baseline; model loaded lazily."""

    #: remembered import/model failure so later tracks fail fast ("unavailable"
    #: in the message makes the parallel runner skip the whole method)
    _unavailable: ClassVar[str | None] = None

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.config = recording.config

    def detect(
        self, onset_threshold: float = 0.5, frame_threshold: float = 0.3,
    ) -> NoteData:
        if BasicPitchDetector._unavailable is not None:
            raise RuntimeError(f"Basic Pitch unavailable: {BasicPitchDetector._unavailable}")
        configure_environment()
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import AUDIO_SAMPLE_RATE, FFT_HOP, run_inference
            from basic_pitch.note_creation import model_output_to_notes

            model_path = self._model_path(Path(ICASSP_2022_MODEL_PATH))
            with suppress_output():  # keep TF/model-load logs off the progress display
                model_output = run_inference(
                    audio_path_for(self.recording), model_path, debug_file=None,
                )
        except Exception as exc:
            BasicPitchDetector._unavailable = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"Basic Pitch unavailable: {BasicPitchDetector._unavailable}"
            ) from exc

        min_note_len = int(np.round(
            min_note_seconds(self.recording) * AUDIO_SAMPLE_RATE / FFT_HOP
        ))
        _, note_events = model_output_to_notes(
            model_output,
            onset_thresh=onset_threshold,
            frame_thresh=frame_threshold,
            min_note_len=min_note_len,
            min_freq=self.config.fmin,
            max_freq=self.config.fmax,
            include_pitch_bends=False,
        )

        note_data = NoteData()
        origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))
        for idx, (start, end, pitch, amplitude, *_rest) in enumerate(note_events):
            if end <= start:
                continue
            note_data.write_note(Note(
                i=idx,
                start_time=float(start) + origin,
                end_time=float(end) + origin,
                midi_num=[float(pitch)],
                velocity=int(np.clip(round(float(amplitude) * 127), 1, 127)),
            ))
        return note_data

    @staticmethod
    def _model_path(default_model_path: Path) -> Path:
        """Pick the model flavour with an importable runtime, else the default."""
        for suffix, modules in (
            (".tflite", ("tensorflow", "tflite_runtime")),
            (".onnx", ("onnxruntime",)),
            (".mlpackage", ("coremltools",)),
        ):
            candidate = default_model_path.with_suffix(suffix)
            if candidate.exists() and any(
                importlib.util.find_spec(m) is not None for m in modules
            ):
                return candidate
        return default_model_path
