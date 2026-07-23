from __future__ import annotations

"""CocoChorales note-detection benchmark.

Runs the note-detector comparison (see ``NoteBenchmarker``) on materialized
CocoChorales stems. Unlike the etude benchmark it does NOT synthesize its own
audio: the reference is the stem's ``stems_midi`` file and the pitch track is
loaded from the shared ``pyin_smooth`` pitch cache the pitch benchmark already
produced. Pitch detection is therefore never re-run and never timed — the rows
carry only the note detector's own compute time and Audio(s)/Compute(s).

The class multiply-inherits the note-method matrix from ``NoteBenchmarker`` and
the CocoChorales data plumbing (manifest / materialization / f0 / cache layout)
from ``CocoChoralesBenchmarker``; both share ``PitchBenchmarker`` so the MRO is
linear (CocoNote -> Note -> Coco -> Pitch).
"""

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

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

from app_logic.user.ds.Recording import Recording  # noqa: E402
from benchmarks.modules.CocoChoralesBenchmarker import CocoChoralesBenchmarker, CocoStemRecord  # noqa: E402
from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker, OneInstrumentScoreData  # noqa: E402
from benchmarks.modules.pitch.PitchBenchmarker import PathLike  # noqa: E402


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
        transcribers, spectral onsets) or when the pitch cache is missing and the
        track has to be pitch-detected from scratch. Change-point methods run on
        the cached pitch track alone."""
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
        self.prepare_for_note_detection(
            recording,
            resize_score_to_pitch=(align == "resize"),
            detect_transitions=False,
        )
        ref_iv, ref_pi = self.notedata_to_intervals(
            recording.score_data.clipped_note_data(channel=recording.active_instrument),
            recording.config,
        )
        return recording, ref_iv, ref_pi, self._audio_seconds(wav_path)

    # ------------------------------------------------------------- row metadata
    def note_row_meta(self, wav_path: PathLike) -> dict[str, Any]:
        """Split / ensemble / instrument / voice for grouping the raw CSVs."""
        return self.meta_for_wav(wav_path)
