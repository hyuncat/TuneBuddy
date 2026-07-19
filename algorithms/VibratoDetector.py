import numpy as np
from math import ceil

from algorithms.Config import Config
from app_logic.user.ds.VibratoData import VibratoData


class VibratoDetector:
    """Instantaneous vibrato estimation: windowed LS-Prony over the pitch
    track.

    Each grid point gets a centered `vib_win_sec` window of pitch frames, cut
    down to the contiguous voiced SEGMENT around its center: unvoiced dropouts
    up to `vib_max_gap_sec` are bridged by interpolation, while transition
    frames and longer gaps end the segment — a fit must never mix two notes,
    which is what faked large slow "vibrato" at every note boundary. The
    segment is DC/linear detrended (plus a Heaviside per bridged dropout, so
    an unmarked note change reads as a step, not oscillation), then an
    order-`vib_order` least-squares
    linear prediction (covariance-method Prony) whose least-damped oscillatory
    root gives the modulation RATE; the EXTENT is the LS sinusoid amplitude
    (± around center, in cents) at that rate, fit on real (non-bridged)
    samples only. There are no fixed Hz or cents thresholds: every
    pitch-frame center gets a stored value. Segments without `vib_min_cycles`
    (of both the fitted rate and the OBSERVED sign alternations — a ramp,
    step, or bump fits a sinusoid but cannot alternate), enough voiced pitch,
    or a coherent fit are explicitly stored as 0 Hz / 0 cents rather than
    disappearing as gaps. Transition marks come from note detection, so the
    pre-analysis pass cuts only on gaps; Recording.detect_notes re-runs
    detect() once the marks exist.

    Why Prony instead of an FFT peak: at 0.4 s the DFT's resolution (2.5 Hz
    Rayleigh, ~10 Hz Hann mainlobe) is on the order of the 4-8 Hz being
    measured, and zero-padding refines only the grid — parametric fits are
    the short-window standard (McLeod 2008, "Fast, Accurate Pitch Detection
    Tools for Music Analysis", ch. 9: Prony over ~0.4 s windows; Yang, Rajab
    & Chew 2017, J. Math & Music: the Filter Diagonalisation Method). Prony's
    classic noise sensitivity is tamed by the smoothed input track and the LS
    formulation; low-quality results become continuous zero samples rather
    than missing timepoints."""

    VIB_MIN_VOICED_FRAC = 0.5
    # The pitch frame rate (~344 fps at h1=128) puts a 4-8 Hz vibrato pole at
    # w ~ 0.1 rad, where noise collapses the conjugate LP pair onto the real
    # axis (the classic low-normalized-frequency Prony failure). Mean-pool the
    # window down to ~this rate first: the pole moves to a well-conditioned
    # angle and the pooling averages the noise down. The EXTENT is then fit at
    # full rate, so pooling's slight sinc rolloff never touches it.
    PRONY_RATE_HZ = 50.0
    # A same-sign excursion of the detrended track counts toward the observed
    # alternation total when its peak reaches this fraction of the fitted
    # amplitude: big enough that jitter blips neither count nor split a real
    # half-cycle, small enough to keep the fading half-cycles of
    # amplitude-modulated vibrato.
    ALT_PROM_FRAC = 0.3
    # Bridged dropouts at least this long get a Heaviside column in the
    # detrend, so an unmarked note change across a gap is explained as a level
    # step instead of being forced into the sinusoid (a linear-detrended step
    # is a +-+- sawtooth that fools even the alternation count). Real vibrato
    # is untouched: over full cycles each side's mean is ~0, so the fitted
    # step is ~0. Shorter dropouts can't hide a real note change (those carry
    # >=10 ms of transition/unvoiced frames) and don't spend the extra dof.
    STEP_MIN_GAP_SEC = 0.01

    def __init__(self, recording=None, config: Config = None):
        if not recording and not config:
            raise ValueError("Must provide either a recording or a config to initialize the VibratoDetector.")
        self.recording = recording
        self.config = config if config else recording.config

    def update_config(self, config: Config):
        self.config = config

    # --- whole-track / incremental passes ---
    def detect(self, pitch_data) -> VibratoData:
        """Full pass over every pitch-frame center, including clipped edge
        windows when they still contain `vib_min_cycles` estimated cycles."""
        vd = VibratoData(config=self.config)
        vd.t_origin = pitch_data.t_origin
        self.extend(vd, pitch_data, finalize=True)
        return vd

    def extend(self, vibrato_data: VibratoData, pitch_data,
               finalize: bool = False) -> None:
        """Compute every not-yet-computed grid point whose full centered
        window of pitch frames exists. Incremental: cheap to call per written
        frame from the live pitch thread (values land half a window behind
        the playhead), and it is the whole offline pass when called once."""
        cfg = self.config
        frame_rate = cfg.sr / cfg.h1
        half = max(4, int(round(cfg.vib_win_sec * frame_rate / 2)))
        if vibrato_data.computed_until == 0:
            vibrato_data.t_origin = pitch_data.t_origin
        available = pitch_data.frames_available()
        if available <= 0:
            return
        stride = vibrato_data.stride
        i = vibrato_data.computed_until
        first_written = vibrato_data.source_first_index
        if first_written is None:
            first_written = next(
                (j for j, pitch in enumerate(pitch_data.data[:available])
                 if pitch is not None),
                available,
            )
            if first_written < available:
                vibrato_data.source_first_index = first_written
        if first_written >= available:
            return
        last_written = next(
            (j for j in range(available - 1, first_written - 1, -1)
             if pitch_data.data[j] is not None),
            first_written,
        )
        first_grid = ceil(first_written / stride)
        if i < first_grid:
            if first_grid:
                vibrato_data.write(first_grid - 1, np.nan, np.nan, np.nan)
            i = first_grid

        # Live values remain centered and therefore land half a window behind
        # the playhead. A completed/offline track also tries the asymmetric
        # edge windows, so every pitch timepoint is represented when two cycles
        # fit in the audio that actually exists there.
        last_grid = last_written // stride
        ready_grid = ((available - half - 1) // stride
                      if not finalize else last_grid)
        while i <= min(last_grid, ready_grid):
            center = i * stride
            lo = max(first_written, center - half)
            hi = min(last_written + 1, center + half + 1)
            vals = self._segment_values(pitch_data, lo, hi, center)
            sample = (self.vibrato_sample(vals, frame_rate)
                      if vals is not None else None)
            if sample is None:
                # Computed-but-no-oscillation is real information, distinct
                # from an unwritten/not-yet-computed NaN grid slot. Zero keeps
                # the UI curve continuous through unvoiced and non-vibrato
                # moments while note_summary can still ignore it.
                vibrato_data.write(i, 0.0, 0.0, 0.0)
            else:
                vibrato_data.write(i, *sample)
            i += 1

    def _segment_values(self, pitch_data, lo: int, hi: int,
                        center: int) -> np.ndarray | None:
        """Midi values for the contiguous analysis segment around frame
        `center` within [lo, hi): voiced frames with interior dropout gaps
        (<= vib_max_gap_sec, NaN for the fit to bridge). Transition frames and
        longer gaps end the segment. None when no voiced frame is reachable
        from the center — that instant genuinely has no oscillation to
        measure."""
        cfg = self.config
        frames = pitch_data.data[lo:hi]
        n = len(frames)
        vals = np.full(n, np.nan)
        hard = np.zeros(n, dtype=bool)
        for k, p in enumerate(frames):
            if pitch_data.is_voiced_pitch(p, include_transitions=False):
                vals[k] = p.value
            elif p is not None and getattr(p, "is_transition", False):
                hard[k] = True
        voiced = np.isfinite(vals)
        max_gap = max(0, int(round(cfg.vib_max_gap_sec * cfg.sr / cfg.h1)))
        c = min(max(center - lo, 0), n - 1)

        def reach(step: int) -> int | None:
            """Nearest voiced index walking from the center `step`-ward,
            within the gap tolerance and never across a transition."""
            j, gap = c, 0
            while 0 <= j < n and gap <= max_gap:
                if hard[j]:
                    return None
                if voiced[j]:
                    return j
                gap += 1
                j += step
            return None

        anchors = [a for a in (reach(-1), reach(1)) if a is not None]
        if not anchors:
            return None

        def expand(a: int, step: int) -> int:
            """Last voiced index reachable from anchor `a` walking `step`-ward."""
            last, j, gap = a, a + step, 0
            while 0 <= j < n and gap <= max_gap and not hard[j]:
                if voiced[j]:
                    last, gap = j, 0
                else:
                    gap += 1
                j += step
            return last

        start = expand(min(anchors), -1)
        end = expand(max(anchors), 1)
        return vals[start:end + 1]

    # --- the windowed estimator ---
    def vibrato_sample(self, vals: np.ndarray, frame_rate: float):
        """One analysis segment -> (rate_hz, extent_cents, quality), or None
        when no credible vibrato. `vals` = midi pitches whose NaN runs are
        short dropouts (segment endpoints are always voiced)."""
        n = len(vals)
        voiced = np.isfinite(vals)
        if n < 8 or voiced.mean() < self.VIB_MIN_VOICED_FRAC:
            return None
        idx = np.arange(n, dtype=np.float64)
        # nuisance design: DC + slope + one Heaviside per bridged dropout
        # (see STEP_MIN_GAP_SEC), fit on real samples only
        vi = np.flatnonzero(voiced)
        min_gap = max(1, int(round(self.STEP_MIN_GAP_SEC * frame_rate)))
        gap_starts = vi[np.flatnonzero(np.diff(vi) > min_gap) + 1]
        if len(gap_starts) > 4:  # that torn, the segment is not a note
            return None
        T = np.column_stack(
            [np.ones(n), idx] + [(idx >= g).astype(float) for g in gap_starts])
        beta, *_ = np.linalg.lstsq(T[voiced], vals[voiced], rcond=None)
        # interpolate the RESIDUALS so bridged gaps carry no step energy
        x = np.interp(idx, idx[voiced], vals[voiced] - T[voiced] @ beta)
        xv = x[voiced]
        var = float(np.dot(xv, xv))
        if var <= np.finfo(float).eps * n:
            return None
        rate = self._prony_rate(x, frame_rate)
        if rate is None:
            return None
        min_cycles = max(0.0, float(self.config.vib_min_cycles))
        window_sec = max(0.0, (n - 1) / frame_rate)
        if rate * window_sec < min_cycles:
            return None
        # amplitude/phase by LS at the Prony rate (cos/sin, nuisance columns
        # refined jointly), on real samples only: bridged points must not
        # shape amplitude/quality
        w = 2.0 * np.pi * rate / frame_rate
        A = np.column_stack([np.cos(w * idx), np.sin(w * idx), T])[voiced]
        coef, *_ = np.linalg.lstsq(A, xv, rcond=None)
        amp = float(np.hypot(coef[0], coef[1]))
        resid = xv - A @ coef
        quality = 1.0 - float(np.dot(resid, resid)) / var
        if quality < float(self.config.vib_min_quality):
            return None
        if self._observed_cycles(xv, amp) < min_cycles:
            return None
        return float(rate), amp * 100.0, float(quality)

    @classmethod
    def _observed_cycles(cls, xv: np.ndarray, amp: float) -> float:
        """Cycles of actual alternation in the detrended real samples, at two
        counted half-cycles per cycle. A half-cycle counts when its peak
        reaches ALT_PROM_FRAC of the fitted amplitude AND its sign alternates
        with the last counted one — this is what a ramp, step, or bump cannot
        fake no matter how well a single sinusoid fits it."""
        if amp <= 0.0:
            return 0.0
        floor = cls.ALT_PROM_FRAC * amp
        signs = np.sign(xv)
        halves = 0
        counted_sign = 0.0
        i, n = 0, len(xv)
        while i < n:
            s = signs[i]
            if s == 0.0:
                i += 1
                continue
            peak = 0.0
            while i < n and signs[i] != -s:  # zeros continue the stretch
                peak = max(peak, abs(xv[i]))
                i += 1
            if peak >= floor and s != counted_sign:
                halves += 1
                counted_sign = s
        return halves / 2.0

    def _prony_rate(self, x: np.ndarray, frame_rate: float) -> float | None:
        """Rate from the least-damped oscillatory root of an order-p LS linear prediction
        (covariance method), on the mean-pooled window (see PRONY_RATE_HZ).
        p=2 is one real sinusoid; higher p absorbs residual drift. The root's
        angle naturally bounds the result below the pooled Nyquist frequency;
        no musical Hz band is imposed."""
        p = max(2, int(self.config.vib_order))
        decim = max(1, int(round(frame_rate / self.PRONY_RATE_HZ)))
        m = len(x) // decim
        if m <= p + 4:
            return None
        xd = x[:m * decim].reshape(m, decim).mean(axis=1)
        fs = frame_rate / decim
        cols = [xd[p - 1 - k: len(xd) - 1 - k] for k in range(p)]
        X = np.column_stack(cols)
        a, *_ = np.linalg.lstsq(X, xd[p:], rcond=None)
        roots = np.roots(np.concatenate(([1.0], -a)))
        rates = np.angle(roots) * fs / (2.0 * np.pi)
        oscillatory = [
            (abs(abs(z) - 1.0), float(f))
            for z, f in zip(roots, rates)
            if np.isfinite(f) and f > 0.0
        ]
        if not oscillatory:
            return None
        return min(oscillatory)[1]
