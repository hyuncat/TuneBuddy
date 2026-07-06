"""Shared plumbing for the external note-transcription baselines."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pretty_midi

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.Recording import Recording


@contextlib.contextmanager
def suppress_output():
    """Silence stdout+stderr at the fd level for the duration of the block.

    The external packages chatter to the terminal — crepe_notes ``print()``s the
    audio path / onset status, CREPE runs tqdm bars, and TF/model loads log at
    the C level. In the parallel runner the workers share the terminal with the
    main process's live progress display (cursor-up + clear-line redraws), so any
    stray line corrupts it. fd-level redirection catches all three; the caught
    exception still carries the detail the runner prints outside this scope."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def audio_path_for(recording: Recording) -> str:
    """Existing source WAV if any, else the recording's buffer written to a temp file."""
    candidate = getattr(recording, "audio_filepath", None)
    if candidate is not None and Path(candidate).exists():
        return str(candidate)
    import soundfile as sf

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, recording.audio_data.read_all(), int(recording.audio_data.sr))
    return handle.name


def write_audio(recording: Recording, audio_path: Path) -> None:
    import soundfile as sf

    sf.write(str(audio_path), recording.audio_data.read_all(), int(recording.audio_data.sr))


def min_note_seconds(recording: Recording) -> float:
    """Production min-note floor (assumes update_min_note_length already ran)."""
    return max(
        0.0,
        float(recording.config.get_min_note_length())
        * recording.note_detector.MIN_NOTE_FACTOR,
    )


def notedata_from_midi(midi_path: Path, origin: float = 0.0) -> NoteData:
    raw_notes = sorted(
        (note for inst in pretty_midi.PrettyMIDI(str(midi_path)).instruments
         for note in inst.notes),
        key=lambda note: (note.start, note.end, note.pitch),
    )
    note_data = NoteData()
    for idx, pm_note in enumerate(raw_notes):
        if pm_note.end <= pm_note.start:
            continue
        start_time = float(pm_note.start) + origin
        # NoteData holds one note per onset; nudge exact collisions apart
        while start_time in note_data.data:
            start_time = float(np.nextafter(start_time, float("inf")))
        note_data.write_note(Note(
            i=idx,
            start_time=start_time,
            end_time=float(pm_note.end) + origin,
            midi_num=[int(pm_note.pitch)],
            velocity=int(pm_note.velocity),
        ))
    return note_data


class _OptionalBackendFilter(logging.Filter):
    _PREFIXES = (
        "Coremltools is not installed.",
        "tflite-runtime is not installed.",
        "onnxruntime is not installed.",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.getMessage().startswith(p) for p in self._PREFIXES)


def configure_environment() -> None:
    """Quiet optional-backend noise and shim legacy APIs the old deps expect."""
    mpl_dir = Path(tempfile.gettempdir()) / "attune-matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    warnings.filterwarnings(
        "ignore", message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    root_logger = logging.getLogger()
    if not any(isinstance(f, _OptionalBackendFilter) for f in root_logger.filters):
        root_logger.addFilter(_OptionalBackendFilter())
    _patch_legacy_collections_abc()
    _patch_legacy_numpy_aliases()


def _patch_legacy_collections_abc() -> None:
    """madmom (pulled in by CREPE Notes) still imports the moved ABCs from
    `collections`, which Python 3.12 only exposes from collections.abc."""
    import collections
    import collections.abc

    for name in ("Callable", "Iterable", "Mapping", "MutableMapping",
                 "MutableSequence", "Sequence"):
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))


def _patch_legacy_numpy_aliases() -> None:
    """NumPy aliases removed after 1.20 that older madmom releases still read."""
    for name, value in {
        "bool": bool, "complex": complex, "float": float,
        "int": int, "object": object, "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)
