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
    pitch_thresh: float = 0.5 # minimum difference between two notes to consider them distinct (in semitones)
    min_note_length: float = DEFAULT_MIN_NOTE_LENGTH # shortest expected note, in seconds
    note_detection_min_note_factor: float = 0.6
    note_detection_pelt_jump: int = 1
    note_detection_refine_with_onsets: bool = False
    note_detection_onset_min_note_factor: float = 1.0
    note_detection_onset_min_stable_ratio: float = 0.8
    mistake_checker_min_note_factor: float = 0.3

    # --- STRING EDIT PARAMETERS (pitch-only aligner + MistakeChecker) ---
    ins_cost: float = 5
    del_cost: float = 5
    pitch_tolerance: float = 0.5

    # --- ONSET-AWARE ALIGNMENT COST MODEL ---
    # The onset-aware aligner sets its DP costs to the negative log-priors of a
    # simple generative mistake model (plus Gaussian onset/pitch penalties), so the
    # substitution-vs-(insertion+deletion) tradeoff is determined by these stats
    # rather than hand-tuned magic numbers. `mistake_rate` and the screwup-type
    # shares default to the PolyTune injector (16 uniform codes: 1 deletion,
    # 1 substitution, 2 insertion, 12 timing-only) — change mistake_rate and the
    # costs follow. See alignment_priors() / MistakeDetector._align_cost_model().
    mistake_rate: float = 0.25
    screwup_deletion_share: float = 1 / 16
    screwup_substitution_share: float = 1 / 16
    screwup_insertion_share: float = 2 / 16
    align_onset_sigma: float = 0.20   # sec; stdev of a matched/substituted onset
    align_pitch_sigma: float = 2.0    # semitones; stdev of a substitution's pitch

    # --- TIMING FEEDBACK PARAMETERS ---
    # Threshold for post-alignment timing mistakes. A matched note is flagged
    # early/late when its onset is off by more than this many seconds, and
    # short/long when its duration differs by more than this many seconds.
    timing_tolerance: float = 0.25

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

    def alignment_priors(self) -> tuple[float, float, float, float]:
        """Per-score-note operation priors (p_correct, p_substitution, p_insertion,
        p_deletion) for the onset-aware aligner, derived from `mistake_rate` and the
        screwup-type shares. Automated: the aligner's DP costs follow whenever these
        change (a note is 'correct' unless it was deleted or substituted; insertions
        are extra notes scored against the same per-note budget)."""
        lam = max(0.0, min(1.0, float(self.mistake_rate)))
        p_del = lam * self.screwup_deletion_share
        p_sub = lam * self.screwup_substitution_share
        p_ins = lam * self.screwup_insertion_share
        p_correct = max(1e-6, 1.0 - p_del - p_sub)
        return p_correct, p_sub, p_ins, p_del

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
                f"   pitch_thresh={self.pitch_thresh}, min_note_length={self.min_note_length:.3f}, note_detection_pelt_jump={self.note_detection_pelt_jump}, note_detection_refine_with_onsets={self.note_detection_refine_with_onsets}, note_detection_onset_min_note_factor={self.note_detection_onset_min_note_factor}, note_detection_onset_min_stable_ratio={self.note_detection_onset_min_stable_ratio},\n"
                f"   ins_cost={self.ins_cost}, del_cost={self.del_cost}, pitch_tolerance={self.pitch_tolerance}, timing_tolerance={self.timing_tolerance}")
