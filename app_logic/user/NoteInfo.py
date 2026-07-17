from dataclasses import dataclass

import numpy as np


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

    # Volume is shown RELATIVE to this take only — never as an absolute number.
    # A computer mic's level is uncalibrated (mic gain/distance/AGC all unknown),
    # and an absolute dBFS reading disagrees with the color ramp (a note can be
    # loud FOR THIS TAKE yet low in absolute dBFS). Normalizing to the take's own
    # [quietest, loudest] is gain-invariant and matches the ramp, so volume_frac
    # is the only volume descriptor — the readout is the color bar, no number.

    note_name: str
    cents: float                            # offset from the nearest semitone
    onset: float                            # app-time (sec)
    duration: float                         # sec
    onset_mistake: str | None = None        # "early" / "late" / None
    duration_mistake: str | None = None     # "long" / "short" / None
    vibrato_rate_hz: float | None = None
    vibrato_extent_cents: float | None = None  # sinusoid amplitude (± around center)
    volume_frac: float | None = None        # 0..1 bar position in the take's dBFS range; None = no data

    @classmethod
    def analyze(cls, recording, note) -> "NoteInfo":
        midi = float(note.midi_num[0])
        onset_mistake, duration_mistake = cls._timing_mistakes(recording, note)
        rate, extent = cls._vibrato(recording, note)
        return cls(
            note_name=note.get_note_name(),
            cents=(midi - round(midi)) * 100.0,
            onset=note.start_time,
            duration=note.end_time - note.start_time,
            onset_mistake=onset_mistake,
            duration_mistake=duration_mistake,
            vibrato_rate_hz=rate,
            vibrato_extent_cents=extent,
            volume_frac=cls._volume(recording, note),
        )

    @staticmethod
    def _timing_mistakes(recording, note) -> tuple[str | None, str | None]:
        """This note's onset ("early"/"late") and duration ("long"/"short")
        mistakes from the alignment, if analyze() has produced them."""
        onset_mistake = duration_mistake = None
        alignment = getattr(recording, "alignment", None)
        for m in (alignment.mistakes_for(note, "timing") if alignment else []):
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
    def _volume(cls, recording, note) -> float | None:
        """0..1 bar position within the take's dBFS range (None when the note
        carries no measurable volume): the note's mean level placed across the
        take's [quietest, loudest] frame dBFS — the same normalization the plot
        dots and ScoreViewer dots color across (mirrored inline here since
        app_logic can't import ui.Colors)."""
        pd = recording.pitch_data
        vol = pd.mean_volume(note.start_time, note.end_time)
        vmin, vmax = pd.volume_range_db()
        if vol <= 0 or vmin is None or vmax is None or vmax <= vmin:
            return None
        frac = (20.0 * np.log10(vol) - vmin) / (vmax - vmin)
        return float(min(max(frac, 0.0), 1.0))
