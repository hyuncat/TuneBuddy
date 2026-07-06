from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

from app_logic.NoteData import NoteData
from app_logic.user.ds.Recording import Recording
from benchmarks.note.algorithms.external import (
    configure_environment,
    notedata_from_midi,
    suppress_output,
    write_audio,
)


class CrepeNotesDetector:
    """External `crepe_notes` baseline.

    Calls the installed package rather than reimplementing the CREPE Notes
    paper over Attune PitchData: it runs its own pitch tracker, applies the
    confidence-gradient postprocessor, writes MIDI, and we translate that MIDI
    back into NoteData.
    """

    _unavailable: ClassVar[str | None] = None

    def __init__(self, recording: Recording) -> None:
        self.recording = recording

    def detect(
        self,
        sensitivity: float = 0.001,
        min_duration: float = 0.03,
        min_velocity: int = 6,
        pitch_tracker: str = "crepe",
    ) -> NoteData:
        if CrepeNotesDetector._unavailable is not None:
            raise RuntimeError(f"CREPE Notes unavailable: {CrepeNotesDetector._unavailable}")
        configure_environment()
        try:
            from crepe_notes.crepe_notes import process, run_pitch_tracker
        except Exception as exc:
            CrepeNotesDetector._unavailable = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"CREPE Notes unavailable: {CrepeNotesDetector._unavailable}"
            ) from exc

        origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "attune-crepe-notes.wav"
            write_audio(self.recording, audio_path)
            try:
                # the package prints the audio path / onset status and runs tqdm
                # bars; muzzle them so the parallel progress display stays intact
                with suppress_output():
                    frequency, confidence = run_pitch_tracker(audio_path, tracker=pitch_tracker)
                    midi_path = process(
                        frequency, confidence, audio_path,
                        output_label=f"{pitch_tracker}.transcription",
                        sensitivity=float(sensitivity),
                        use_smoothing=False,
                        min_duration=float(min_duration),
                        min_velocity=int(min_velocity),
                        disable_splitting=False,
                        tuning_offset=False,
                        use_cwd=False,
                        detect_amplitude=True,
                        save_analysis_files=False,
                        pitch_tracker=pitch_tracker,
                    )
            except Exception as exc:
                raise RuntimeError(f"CREPE Notes failed: {type(exc).__name__}: {exc}") from exc
            return notedata_from_midi(Path(midi_path), origin=origin)
