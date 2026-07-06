from __future__ import annotations

from typing import ClassVar

from app_logic.NoteData import NoteData
from app_logic.user.ds.Recording import Recording
from benchmarks.note.algorithms.external import audio_path_for


class TonyDetector:
    """Tony-equivalent pYIN Vamp `notes` output via Sonic Annotator."""

    _unavailable: ClassVar[str | None] = None

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.config = recording.config

    def detect(self) -> NoteData:
        if TonyDetector._unavailable is not None:
            raise RuntimeError(f"Tony pYIN unavailable: {TonyDetector._unavailable}")
        try:
            from benchmarks.modules.pitch.TonyPyinRunner import TonyPyinRunner

            return TonyPyinRunner(self.config).detect_notes(audio_path_for(self.recording))
        except Exception as exc:
            # sonic-annotator missing is the common failure -> latch method-wide
            TonyDetector._unavailable = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"Tony pYIN unavailable: {TonyDetector._unavailable}"
            ) from exc
