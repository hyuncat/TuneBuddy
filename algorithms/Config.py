import numpy as np
from dataclasses import dataclass
from typing import ClassVar

# --- pYIN voicing / volume-gate defaults (shared with the benchmark harness) ---
PRAAT_DEFAULT_VOICING_THRESHOLD = 0.45
PYIN_PRAAT_MIRROR_UNV_THRESH = 1.0 - PRAAT_DEFAULT_VOICING_THRESHOLD
PYIN_DEFAULT_UNV_THRESH = 0.9
PYIN_DEFAULT_MIN_VOLUME = 0.05
PYIN_DEFAULT_MAX_VOLUME = 0.95


@dataclass
class Config:
    DEFAULT_MIN_NOTE_LENGTH: ClassVar[float] = 0.03

    # note-name spellings indexed by pitch class (midi % 12). get_note_name()
    # picks one; the transpose autocomplete offers both.
    SHARP_NOTE_NAMES: ClassVar[list] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    FLAT_NOTE_NAMES: ClassVar[list] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    verbose: bool = False

    # --- PITCH DETECTION PARAMETERS ---
    sr: int = 44100     # sample rate
    w1: int = 1024 * 4  # frame size
    h1: int = 128       # hop size
    fmin: float = 196.0 # Hz
    fmax: float = 3000.0 # Hz
    tuning: float = 440.0  # Hz
    unv_thresh: float = 0.9  # if unvoiced_prob > unv_thresh, consider the frame unvoiced

    # volume gating
    min_volume: float = 0.05  # remove any frame < min_volume * max_volume reference
    max_volume: float = 0.95  # %th percentile (as a fraction) of frame RMS used as the loud reference

    # --- NOTE DETECTION PARAMETERS ---
    pitch_thresh: float = 0.25  # in semitones, min diff b/w 2 notes to consider them distinct
    min_note_length: float = 0.03  # in sec
    h2: int = 1  # PELT jump parameter
    min_gap_length: float = 0.1 # in sec

    # --- STRING EDIT PARAMETERS ---
    ins_cost: float = 5
    del_cost: float = 5
    pitch_tolerance: float = 0.5   # semitones
    timing_tolerance: float = 0.25  # sec

    # --- note-name helper ---
    @staticmethod
    def get_note_name(midi_num: float | None, prefer_flats: bool = False) -> str:
        """Convert a MIDI number to a letter name like C4, F#3 (or Bb3 with
        prefer_flats). Note naming is tuning-independent, so this is a static
        method that both Pitch and Note route through. Rests/unvoiced (None or a
        negative midi_num) render as an em dash."""
        if midi_num is None or midi_num < 0:
            return "—"
        n = int(round(midi_num))
        names = Config.FLAT_NOTE_NAMES if prefer_flats else Config.SHARP_NOTE_NAMES
        return f"{names[n % 12]}{n // 12 - 1}"
    
    def get_min_note_length(self, type: str="sec"):
        """Return the minimum note length in seconds or pitch frames (h1/sr grid)."""
        if type == "sec":
            return self.min_note_length
        elif type == "frames":
            fr = self.sr / self.h1
            return max(1, int(np.ceil(self.min_note_length * fr)))
        else:
            raise ValueError(f"Invalid type {type} for get_min_note_length()")

    def set_min_note_length(self, sec: float):
        """Set the central shortest-note estimate in seconds."""
        if sec is None or sec <= 0:
            sec = self.DEFAULT_MIN_NOTE_LENGTH
        self.min_note_length = float(sec)

    def set_min_note_length_from_notedata(self, note_data) -> float:
        """Compatibility helper for older benchmark/notebook callers."""
        if note_data is None:
            self.set_min_note_length(self.DEFAULT_MIN_NOTE_LENGTH)
            return self.min_note_length
        try:
            sec = note_data.get_min_note_length(
                default=self.DEFAULT_MIN_NOTE_LENGTH,
                clean=True,
            )
        except AttributeError:
            sec = self.DEFAULT_MIN_NOTE_LENGTH
        self.set_min_note_length(sec)
        return self.min_note_length

    def min_note_seconds(self, factor: float = 1.0) -> float:
        return max(0.0, float(self.min_note_length) * float(factor))

    def min_note_pitch_frames(self, factor: float = 1.0) -> int:
        frame_rate = self.sr / self.h1
        return max(1, int(np.ceil(self.min_note_seconds(factor) * frame_rate)))

    # --- pitch conversion methods ---
    def freq_to_midi(self, freq: float) -> float:
        """
        Convert a frequency to a MIDI note number.
        """
        if freq <= 0:
            # print("bad freq")
            return(-1)
        return 69 + 12 * np.log2(freq / self.tuning)

    def midi_to_freq(self, midi_num: float) -> float:
        """
        Convert a MIDI note number to frequency.
        """
        return self.tuning * (2 ** ((midi_num - 69) / 12))


    def __repr__(self):
        return (f"Config\n---\n   sr={self.sr}, w1={self.w1}, h1={self.h1}, fmin={self.fmin}, fmax={self.fmax}, tuning={self.tuning}, unv_thresh={self.unv_thresh}, min_volume={self.min_volume}, max_volume={self.max_volume},\n"
                f"   pitch_thresh={self.pitch_thresh}, min_note_length={self.min_note_length:.3f}, h2={self.h2},\n"
                f"   ins_cost={self.ins_cost}, del_cost={self.del_cost}, pitch_tolerance={self.pitch_tolerance}, timing_tolerance={self.timing_tolerance}")
