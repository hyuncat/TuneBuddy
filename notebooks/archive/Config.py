import numpy as np
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Config:
    DEFAULT_MIN_NOTE_LENGTH: ClassVar[float] = 0.03

    # --- PITCH DETECTION PARAMETERS ---
    sr: int = 44100    # sample rate
    w1: int = 1024 * 4  # frame size
    h1: int = 128       # hop size
    fmin: float = 196.0
    fmax: float = 3000.0
    tuning: float = 440.0
    unv_thresh: float = 0.9 # if unvoiced_prob > unv_thresh, consider the frame unvoiced

    # --- NOTE DETECTION PARAMETERS ---
    w2: int = 29 # frame size (NOTE: should always be odd)
    h2: int = 19 # hop size
    pitch_thresh: float = 0.5 # rk: used to be 0.75
    slope_thresh: float = 0.5 / 29
    unv_ratio: float = 0.8 # proportion of unvoiced pitches in a window to consider the window unvoiced
    min_note_length: float = DEFAULT_MIN_NOTE_LENGTH # shortest expected note, in seconds
    note_detection_min_note_factor: float = 0.6
    mistake_checker_min_note_factor: float = 0.3

    # --- STRING EDIT PARAMETERS ---
    ins_cost: float = 5
    del_cost: float = 5
    sub_cost: float = 1
    pitch_tolerance: float = 0.5

    # tiger-mom parameter
    tiger_level: int = 1

    # --- TIMING FEEDBACK PARAMETERS ---
    # Threshold for post-alignment timing mistakes. A matched note is flagged
    # early/late when its onset is off by more than this many seconds, and
    # short/long when its duration differs by more than this many seconds.
    timing_tolerance: float = 0.25

    # --- loader ---
    def load_config(self, config: dict):
        """load in a config dictionary"""
        self.w1 = config.get("w1", self.w1)
        self.h1 = config.get("h1", self.h1)
        self.fmin = config.get("fmin", self.fmin)
        self.fmax = config.get("fmax", self.fmax)
        self.tuning = config.get("tuning", self.tuning)
        self.unv_thresh = config.get("unv_thresh", self.unv_thresh)

        self.w2 = config.get("w2", self.w2)
        self.h2 = self.w2 - 2
        self.pitch_thresh = config.get("pitch_thresh", self.pitch_thresh)
        self.slope_thresh = self.pitch_thresh / self.w2 
        # self.slope_thresh = config.get("slope_thresh", self.slope_thresh)
        self.min_note_length = config.get("min_note_length", self.min_note_length)
        self.note_detection_min_note_factor = config.get(
            "note_detection_min_note_factor",
            self.note_detection_min_note_factor,
        )
        self.mistake_checker_min_note_factor = config.get(
            "mistake_checker_min_note_factor",
            self.mistake_checker_min_note_factor,
        )

        self.ins_cost = config.get("ins_cost", self.ins_cost)
        self.del_cost = config.get("del_cost", self.del_cost)
        self.sub_cost = config.get("sub_cost", self.sub_cost)
        self.pitch_tolerance = config.get(
            "pitch_tolerance",
            config.get("tolerance", self.pitch_tolerance),
        )

        self.tiger_level = config.get("tiger_level", self.tiger_level)

        self.timing_tolerance = config.get(
            "timing_tolerance",
            config.get(
                "timing_onset_tol",
                config.get("timing_dur_tol", self.timing_tolerance),
            ),
        )

    def set_min_note_length(self, seconds: float | None) -> float:
        """Set the central shortest-note estimate in seconds."""
        if seconds is None or seconds <= 0:
            seconds = self.DEFAULT_MIN_NOTE_LENGTH
        self.min_note_length = float(seconds)
        return self.min_note_length

    def set_min_note_length_from_notedata(self, note_data) -> float:
        """Update min_note_length from a score/annotation NoteData."""
        if note_data is None:
            return self.set_min_note_length(self.DEFAULT_MIN_NOTE_LENGTH)
        try:
            seconds = note_data.get_min_note_length(
                default=self.DEFAULT_MIN_NOTE_LENGTH,
                clean=True,
            )
        except AttributeError:
            seconds = self.DEFAULT_MIN_NOTE_LENGTH
        return self.set_min_note_length(seconds)

    def min_note_seconds(self, factor: float = 1.0) -> float:
        """Shortest-note estimate after applying an algorithm-specific factor."""
        return max(0.0, float(self.min_note_length) * float(factor))

    def min_note_pitch_frames(self, factor: float = 1.0) -> int:
        """Shortest-note estimate converted to pitch frames (h1/sr grid)."""
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
        return (f"Config\n---\n   sr={self.sr}, w1={self.w1}, h1={self.h1}, fmin={self.fmin}, fmax={self.fmax}, tuning={self.tuning}, unv_thresh={self.unv_thresh},\n"
                f"   w2={self.w2}, h2={self.h2}, pitch_thresh={self.pitch_thresh}, slope_thresh={self.slope_thresh:.3f}, unv_ratio={self.unv_ratio}, min_note_length={self.min_note_length:.3f},\n"
                f"   ins_cost={self.ins_cost}, del_cost={self.del_cost}, sub_cost={self.sub_cost}, pitch_tolerance={self.pitch_tolerance}, timing_tolerance={self.timing_tolerance}, tiger_level={self.tiger_level}")
