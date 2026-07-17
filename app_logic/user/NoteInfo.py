from dataclasses import dataclass

import numpy as np

from algorithms.Config import Config


@dataclass
class NoteInfo:
    """Per-note descriptors (pitch/cents, timing, vibrato, relative volume).
    Generated on demand from a Recording's pitch frames + alignment via
    analyze(), and cached on the inspected Note (Note.info)."""

    # Vibrato is only reported when a dominant pitch oscillation sits in a
    # plausible band and is big/long enough to be deliberate: violin vibrato
    # runs ~5-8 Hz, the band is wider to catch slow student vibrato.
    VIB_MIN_HZ = 2.5
    VIB_MAX_HZ = 10.0
    VIB_MIN_CENTS = 5.0     # smaller oscillations are pitch jitter, not vibrato
    VIB_MIN_CYCLES = 2.0    # need >= 2 full cycles inside the note to trust the peak
    VIB_MIN_VOICED_FRAC = 0.5
    VIB_FFT_PAD = 4096      # zero-pad so short notes still get a fine-grained peak

    # Volume slider floor: a note this far below the take's loudest reads as
    # empty. The absolute readout uses Config.volume_to_db (raw-signal RMS on
    # the full-scale = 94 dB convention; SPL-like, not calibrated SPL).
    VOLUME_FLOOR_DB = -30.0

    note_name: str
    cents: float                            # offset from the nearest semitone
    onset: float                            # app-time (sec)
    duration: float                         # sec
    onset_mistake: str | None = None        # "early" / "late" / None
    duration_mistake: str | None = None     # "long" / "short" / None
    vibrato_rate_hz: float | None = None
    vibrato_extent_cents: float | None = None  # sinusoid amplitude (± around center)
    volume_db: float | None = None          # signed dB vs the take's median note (tooltip)
    volume_abs_db: float | None = None      # absolute dB of the note's mean volume (Config.volume_to_db)
    volume_frac: float = 0.0                # 0..1 slider position (vs loudest note)

    @classmethod
    def analyze(cls, recording, note) -> "NoteInfo":
        midi = float(note.midi_num[0])
        onset_mistake, duration_mistake = cls._timing_mistakes(recording, note)
        rate, extent = cls._vibrato(recording, note)
        volume_db, volume_abs_db, volume_frac = cls._volume(recording, note)
        return cls(
            note_name=note.get_note_name(),
            cents=(midi - round(midi)) * 100.0,
            onset=note.start_time,
            duration=note.end_time - note.start_time,
            onset_mistake=onset_mistake,
            duration_mistake=duration_mistake,
            vibrato_rate_hz=rate,
            vibrato_extent_cents=extent,
            volume_db=volume_db,
            volume_abs_db=volume_abs_db,
            volume_frac=volume_frac,
        )

    @staticmethod
    def _timing_mistakes(recording, note) -> tuple[str | None, str | None]:
        """This note's onset ("early"/"late") and duration ("long"/"short")
        mistakes from the alignment, if analyze() has produced them."""
        onset_mistake = duration_mistake = None
        alignment = getattr(recording, "alignment", None)
        for m in (alignment.timing_mistakes if alignment else []):
            if m.user_note is not note:
                # alignment normally holds the same Note objects as note_data;
                # fall back to onset identity in case a rebuild copied them
                if m.user_note is None or abs(m.user_note.start_time - note.start_time) > 1e-9:
                    continue
            if m.type in ("early", "late"):
                onset_mistake = m.type
            elif m.type in ("long", "short"):
                duration_mistake = m.type
        return onset_mistake, duration_mistake

    @classmethod
    def _note_frames(cls, recording, note) -> list:
        return recording.pitch_data.read(
            start_time=note.start_time, end_time=note.end_time,
        )

    @classmethod
    def _vibrato(cls, recording, note) -> tuple[float | None, float | None]:
        """Dominant pitch oscillation inside the note: (rate Hz, extent cents),
        or (None, None) when there's no credible vibrato. Transition frames are
        excluded and gaps interpolated so slides in/out don't fake a peak; a
        linear detrend removes scoop/drift while leaving the oscillation."""
        pd = recording.pitch_data
        frames = cls._note_frames(recording, note)
        frame_rate = recording.config.sr / recording.config.h1

        vals = np.full(len(frames), np.nan)
        for i, p in enumerate(frames):
            if pd.is_voiced_pitch(p, include_transitions=False):
                vals[i] = p.value
        voiced = ~np.isnan(vals)
        n = len(vals)
        if n < 8 or voiced.mean() < cls.VIB_MIN_VOICED_FRAC:
            return None, None

        idx = np.arange(n, dtype=np.float64)
        vals = np.interp(idx, idx[voiced], vals[voiced])
        vals = vals - np.polyval(np.polyfit(idx, vals, 1), idx)

        window = np.hanning(n)
        n_fft = max(n, cls.VIB_FFT_PAD)
        spec = np.fft.rfft(vals * window, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / frame_rate)
        # |X_k| of a windowed sinusoid of amplitude a is ~ a * sum(window)/2
        amps = np.abs(spec) * 2.0 / window.sum()

        band = (freqs >= cls.VIB_MIN_HZ) & (freqs <= cls.VIB_MAX_HZ)
        if not band.any():
            return None, None
        k = int(np.argmax(amps[band]))
        rate = float(freqs[band][k])
        extent_cents = float(amps[band][k]) * 100.0

        if extent_cents < cls.VIB_MIN_CENTS or (n / frame_rate) * rate < cls.VIB_MIN_CYCLES:
            return None, None
        return rate, extent_cents

    @classmethod
    def _volume(cls, recording, note) -> tuple[float | None, float | None, float]:
        """(signed dB vs the take's MEDIAN note for the tooltip, absolute dB,
        0..1 slider fraction vs the LOUDEST note). Two references: the slider
        bar stays pinned to the loudest note, while the relative readout is
        centered on the median so a note reads +louder / -quieter than typical."""
        pd = recording.pitch_data
        vol = pd.mean_volume(note.start_time, note.end_time)
        notes = recording.note_data.read(i=0, j=len(recording.note_data.times), clean=True)
        vols = [v for v in (pd.mean_volume(n.start_time, n.end_time) for n in notes) if v > 0]
        if vol <= 0 or not vols:
            return None, None, 0.0
        abs_db = Config.volume_to_db(vol)
        # slider position: relative to the loudest note, floored at VOLUME_FLOOR_DB
        frac_db = 20.0 * np.log10(vol / max(vols))
        frac = (frac_db - cls.VOLUME_FLOOR_DB) / (0.0 - cls.VOLUME_FLOOR_DB)
        # tooltip readout: signed dB around the median note
        db = 20.0 * np.log10(vol / float(np.median(vols)))
        return float(db), float(abs_db), float(min(max(frac, 0.0), 1.0))
