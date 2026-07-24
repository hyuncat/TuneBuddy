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
    # These settings change note boundaries while reusing the cached pitch
    # track.  They are code-owned (there is currently no per-take UI for them),
    # so a changed default must invalidate cached notes/alignment instead of
    # being silently replaced by an older sidecar value.
    NOTE_SEGMENTATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "unv_thresh",
        "pitch_thresh",
        "min_note_length_factor",
        "min_silence_duration_ms",
    )
    # These are likewise production-owned rather than per-take preferences.
    # Cached sidecars must not silently restore weights from an older cost model.
    ALIGNMENT_FIELDS: ClassVar[tuple[str, ...]] = (
        "ins_cost",
        "del_cost",
        "alignment_alpha_onset",
        "alignment_alpha_duration",
        "alignment_gamma_pitch",
        "alignment_gamma_time",
    )

    # note-name spellings indexed by pitch class (midi % 12)
    # get_note_name() picks one; the transpose autocomplete offers both
    SHARP_NOTE_NAMES: ClassVar[list] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    FLAT_NOTE_NAMES: ClassVar[list] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    verbose: bool = False

    # --- PITCH DETECTION PARAMETERS ---
    sr: int = 44100     # sample rate
    w1: int = 1024 * 2  # frame size
    h1: int = 128       # hop size
    fmin: float = 196.0 # Hz
    fmax: float = 3000.0 # Hz
    tuning: float = 440.0  # Hz
    unv_thresh: float = 0.9  # if unvoiced_prob > unv_thresh, consider the frame unvoiced

    # volume gating
    min_volume: float = 0.05  # remove any frame < min_volume * max_volume reference
    max_volume: float = 0.95  # %th percentile (as a fraction) of frame RMS used as the loud reference

    # --- NOTE DETECTION PARAMETERS ---
    # Smallest idealized pitch step worth a KernelCPD boundary. It determines
    # beta = 0.5 * min_segment_frames * pitch_thresh**2.
    pitch_thresh: float = 0.75
    # Score-derived shortest-note duration. Recording.update_min_note_length()
    # refreshes this after every score-to-take fit and before correction.
    min_note_length: float = 0.03  # seconds
    # minimum detected segment length as a fraction of the shortest score note
    min_note_length_factor: float = 0.60
    # width of the local decoded-silence majority window. At sr=44100 and
    # h1=128, 10 ms maps to three frames and requires two unvoiced frames.
    min_silence_duration_ms: float = 10.0

    # --- STRING EDIT PARAMETERS ---
    ins_cost: float = 5
    del_cost: float = 5
    # Time-aware string-edit weights:
    #   C_time = alpha_onset*|onset error| + alpha_duration*|duration error|
    #   C_pair = gamma_pitch*|pitch error| + gamma_time*C_time
    #   C_ins  = ins_cost + gamma_time*user duration
    #   C_del  = del_cost + gamma_time*score duration
    # alpha_onset + alpha_duration must equal 1. The gamma weights convert the
    # raw semitone and second-valued terms into a common edit-cost scale. Gap
    # operations use the full unmatched duration because no paired onset exists
    # with which to blend it.
    # Production defaults selected by the runner-v2 seed-0 CocoChorales sweep
    # after full-duration gap costs were introduced. Robust score-time fitting
    # handles the global onset placement before string editing; duration then
    # supplies the local temporal evidence without letting expressive onset
    # shifts manufacture insertion/deletion cascades.
    alignment_alpha_onset: float = 0.0
    alignment_alpha_duration: float = 1.0
    alignment_gamma_pitch: float = 2.0
    alignment_gamma_time: float = 1.0
    pitch_tolerance: float = 0.5   # semitones
    timing_tolerance: float = 0.25  # sec

    # --- VIBRATO (instantaneous windowed LS-Prony — see algorithms/VibratoDetector) ---
    vib_win_sec: float = 0.4      # sliding analysis window (McLeod ch. 9)
    # Frame-dense is an invariant, not a persisted per-take option: old caches
    # may contain the former value 4 and must not silently restore decimation.
    vib_stride: ClassVar[int] = 1
    vib_order: int = 2            # linear-prediction order (2 = one real sinusoid)
    vib_min_cycles: float = 1.5   # cycles the fit AND the observed alternations must cover
    vib_min_quality: float = 0.3  # below this, report continuous 0 Hz / 0 cents
    # Unvoiced dropouts up to this long are bridged so a few lost frames don't
    # cut the vibrato curve; longer gaps and any transition frame end the
    # analysis segment. Keep below the effective note-splitting gap.
    vib_max_gap_sec: float = 0.06

    # --- TIMBRE (semitone-spaced pseudo-CQT heatmap) ---
    cqt_midi_min: int = 36        # C2
    cqt_midi_max: int = 108       # C8 (inclusive)
    cqt_stride: int = 4           # pitch frames per spectrum column; <=0 disables

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

    def note_segmentation_config(self) -> dict[str, float | int]:
        """The code-owned settings that determine detected note boundaries."""
        return {
            name: getattr(self, name)
            for name in self.NOTE_SEGMENTATION_FIELDS
        }

    def note_segmentation_signature(self) -> tuple:
        """Stable comparison key used to invalidate stale note analysis."""
        return tuple(
            (name, getattr(self, name))
            for name in self.NOTE_SEGMENTATION_FIELDS
        )

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
                f"   pitch_thresh={self.pitch_thresh}, min_note_length={self.min_note_length:.3f}, "
                f"min_note_length_factor={self.min_note_length_factor}, "
                f"min_silence_duration_ms={self.min_silence_duration_ms},\n"
                f"   ins_cost={self.ins_cost}, del_cost={self.del_cost}, "
                f"alignment_alpha_onset={self.alignment_alpha_onset}, alignment_alpha_duration={self.alignment_alpha_duration}, "
                f"alignment_gamma_pitch={self.alignment_gamma_pitch}, alignment_gamma_time={self.alignment_gamma_time}, "
                f"pitch_tolerance={self.pitch_tolerance}, timing_tolerance={self.timing_tolerance}")
