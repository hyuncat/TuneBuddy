from __future__ import annotations

"""CocoChorales note-detection benchmark.

Runs the note-detector comparison (see ``NoteBenchmarker``) on materialized
CocoChorales stems. Unlike the etude benchmark it does NOT synthesize its own
audio: the reference is the stem's ``stems_midi`` file and the pitch track is
loaded from the shared ``pyin_smooth`` cache the pitch benchmark already
produced — pitch detection is never re-run and never timed.

Multiply-inherits the note methods from ``NoteBenchmarker`` and the
CocoChorales data plumbing (manifest / materialization / f0 / cache layout)
from ``CocoChoralesBenchmarker``; both share ``PitchBenchmarker`` so the MRO is
linear (CocoNote -> Note -> Coco -> Pitch).
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from app_logic.user.ds.Recording import Recording
from benchmarks.modules.CocoChoralesBenchmarker import CocoChoralesBenchmarker, CocoStemRecord
from benchmarks.modules.pitch.PitchBenchmarker import PathLike
from benchmarks.note.NoteBenchmarker import NoteBenchmarker, OneInstrumentScoreData


class CocoNoteBenchmarker(NoteBenchmarker, CocoChoralesBenchmarker):
    """Note-detection benchmark over materialized CocoChorales stems."""

    def __init__(
        self,
        root: PathLike | None = None,
        f0_fps: float | None = None,
        onset_tolerance: float = 0.05,
    ) -> None:
        # NoteBenchmarker.__init__ only takes onset_tolerance; drive the Coco half
        # explicitly so the dataset root / f0 fps are wired up too.
        NoteBenchmarker.__init__(self, onset_tolerance=onset_tolerance)
        from benchmarks.modules.CocoChoralesBenchmarker import F0_FPS_DEFAULT

        self.COCO_ROOT = (
            Path(root) if root is not None else self.DATASETS / self.DEFAULT_ROOT_NAME
        )
        self.F0_FPS = float(F0_FPS_DEFAULT if f0_fps is None else f0_fps)

    # ------------------------------------------------------------------- paths
    def local_midi_path(self, record: CocoStemRecord) -> Path | None:
        candidates = [
            self.materialized_midi_path(record),
            self.COCO_ROOT / "main_dataset" / record.split / record.track / "stems_midi" / f"{record.stem}.mid",
            self.COCO_ROOT / "main_dataset" / record.split / record.track / "stems_MIDI" / f"{record.stem}.mid",
        ]
        return next((p for p in candidates if p.exists()), None)

    def records_to_note_tracks(
        self, records: Sequence[CocoStemRecord],
    ) -> list[tuple[str, Path, Path]]:
        """(track_id, wav, midi) for stems whose audio AND stems_midi are present."""
        tracks: list[tuple[str, Path, Path]] = []
        for record in records:
            wav = self.local_wav_path(record)
            midi = self.local_midi_path(record)
            if wav is None or midi is None:
                continue
            tracks.append((record.track_id, wav, midi))
        return tracks

    def iter_note_tracks(self, dataset: str = "test") -> Iterator[tuple[str, Path, Path]]:
        yield from self.records_to_note_tracks(self.load_or_build_manifest(dataset))

    def has_note_input_cache(self, wav_path: PathLike) -> bool:
        """Whether the pyin_smooth pitch cache (the note detector's input) exists."""
        return self.has_pitch_cache(self.cache_path_for_wav(wav_path), smooth=True)

    def note_cache_dir(self, primary_path: PathLike, reference_path: PathLike) -> Path:
        """<root>/note_data/, mirroring the flat <root>/pitch_data/ cache."""
        return self.cache_path_for_wav(primary_path).parent.parent

    def note_cache_track_id(self, primary_path: PathLike) -> str:
        """The flat unique id the pitch cache uses (split__track__stem) — the
        wav stem alone ("1_violin") repeats across every coco track."""
        return self.track_id_for_wav(primary_path)

    # ------------------------------------------------------- recording assembly
    def _prepare_note_recording(
        self,
        primary_path: PathLike,
        reference_path: PathLike,
        align: str = "identity",
        needs_audio: bool = True,
    ) -> tuple[Recording, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Assemble a Recording from the cached pyin_smooth pitches + stems_midi.

        The waveform is only loaded when the method actually needs it (external
        transcribers, spectral onsets) or when the pitch cache is missing and
        the track has to be pitch-detected from scratch."""
        wav_path = Path(primary_path)
        midi_path = Path(reference_path)

        score_data = OneInstrumentScoreData(midi_path)
        config = self.config_for(*self.range_from_midi(score_data.midi_numbers))
        recording = self.recording_for(config, score_data=score_data)
        recording.audio_filepath = wav_path

        cache_path = self.cache_path_for_wav(wav_path)
        if needs_audio or not self.has_pitch_cache(cache_path, smooth=True):
            recording.audio_data = self.load_resampled_audio(wav_path, config.sr)

        self.load_or_detect_pitches(
            recording,
            cache_path=cache_path,
            smooth=True,
            write_cache=True,
            verbose=self.algorithm_verbose,
        )
        self.prepare_for_note_detection(recording, resize_score_to_pitch=(align == "resize"))
        ref_iv, ref_pi = self._reference_intervals(recording)
        return recording, ref_iv, ref_pi, self._audio_seconds(wav_path)

    # ------------------------------------------------------------- row metadata
    def note_row_meta(self, wav_path: PathLike) -> dict[str, Any]:
        """Split / ensemble / instrument / voice for the raw CSVs. f0_voice is a
        pitch-benchmark key (0-based f0-pickle column), meaningless against the
        stems_midi reference, so it is dropped here."""
        return {k: v for k, v in self.meta_for_wav(wav_path).items() if k != "f0_voice"}
