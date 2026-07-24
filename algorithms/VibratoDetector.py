import numpy as np
import threading
from bisect import bisect_right
from math import ceil, floor

from algorithms.Config import Config
from app_logic.user.ds.VibratoData import VibratoData


class VibratoDetector:
    """Instantaneous vibrato estimation: windowed LS-Prony over the pitch
    track.

    Each grid point gets a centered `vib_win_sec` window of pitch frames. In
    the offline, note-aware pass that window is hard-clipped to the detected
    note containing its center; within that note, unvoiced dropouts up to
    `vib_max_gap_sec` are bridged by interpolation, while longer gaps end the
    segment. A fit must never mix two notes: two otherwise-flat note changes
    can look exactly like a few cycles of slow vibrato in a centered window.
    The segment is centered on the full note's voiced median and linearly
    detrended (plus a Heaviside per bridged dropout, so
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
    disappearing as gaps. Before notes exist (including live recording), the
    detector falls back to gap/transition segmentation; Recording.detect_notes
    replaces that provisional track with a note-aware pass.

    Why Prony instead of an FFT peak: at 0.4 s the DFT's resolution (2.5 Hz
    Rayleigh, ~10 Hz Hann mainlobe) is on the order of the 4-8 Hz being
    measured, and zero-padding refines only the grid — parametric fits are
    the short-window standard (McLeod 2008, "Fast, Accurate Pitch Detection
    Tools for Music Analysis", ch. 9: Prony over ~0.4 s windows; Yang, Rajab
    & Chew 2017, J. Math & Music: the Filter Diagonalisation Method). Prony's
    classic noise sensitivity is tamed by the smoothed input track and the LS
    formulation; low-quality results become continuous zero samples rather
    than missing timepoints.

    A successful note-aware fit describes the whole pitch span used to make
    it, not only the center frame where the sliding-window calculation was
    requested. After the raw fits are complete, each one is therefore mapped
    back across that source span and overlapping fits are quality-weighted.
    This makes the stored curve a many-frames-to-one-characteristic mapping:
    note edges inherit the estimate that depended on them, while hard gaps and
    note boundaries still stop propagation."""

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

        # live worker (see run): recompute vibrato off the pitch thread.
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()

    def update_config(self, config: Config):
        self.config = config

    # --- live worker: keep vibrato off the pitch/GUI critical path ---
    def run(self):
        """Start the live worker. It recomputes vibrato whenever notify() flags
        new pitch frames, so the pitch->plot loop never blocks on an LS-Prony
        fit and vibrato is free to lag behind under load."""
        self.stop()
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Drain notify() wakeups into incremental extend() passes, then flush
        once more so the drained tail is covered after recording stops."""
        rec = self.recording
        while not self._stop.is_set():
            self._wake.wait(0.1)  # the timeout just re-checks the stop flag
            self._wake.clear()
            self.extend(rec.vibrato_data, rec.pitch_data)
        self.extend(rec.vibrato_data, rec.pitch_data)

    def stop(self):
        """Stop the live worker (a final extend() flush runs as it exits)."""
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._wake.set()  # unblock the wait so it sees the stop flag
            self._thread.join()
        self._thread = None

    def notify(self):
        """Producer hook (pitch thread): new pitch frames are ready to fit."""
        self._wake.set()

    # --- whole-track / incremental passes ---
    def detect(self, pitch_data, note_data=None) -> VibratoData:
        """Full pass over every pitch-frame center, including clipped edge
        windows when they still contain `vib_min_cycles` estimated cycles.

        When ``note_data`` is available, every analysis window and its robust
        pitch center come exclusively from the detected note containing that
        grid point. This is deliberately independent of transition flags.
        """
        vd = VibratoData(config=self.config)
        vd.t_origin = pitch_data.t_origin
        self.extend(vd, pitch_data, finalize=True, note_data=note_data)
        return vd

    def extend(self, vibrato_data: VibratoData, pitch_data,
               finalize: bool = False, note_data=None) -> None:
        """Compute every not-yet-computed grid point whose full centered
        window of pitch frames exists. Incremental: the live worker calls this
        as pitch frames arrive (values land half a window behind the playhead),
        and it is the whole offline pass when called once."""
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
        note_aware = note_data is not None and bool(
            getattr(note_data, "times", None)
        )
        note_starts, note_regions = self._note_regions(
            pitch_data,
            note_data,
            available,
        )
        # Successful finalized note-aware fits are mapped back over the local
        # note segment they describe. Keeping this off the live path preserves
        # its causal half-window delay.
        coverage = [] if note_aware and finalize else None
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
            baseline = None
            if note_aware:
                center_time = (
                    pitch_data.t_origin
                    + (center * cfg.h1 + 0.5 * cfg.w1) / cfg.sr
                )
                region_i = bisect_right(note_starts, center_time) - 1
                region = (
                    note_regions[region_i]
                    if 0 <= region_i < len(note_regions)
                    else None
                )
                if region is None or center_time > region[1]:
                    lo = hi  # the center is in a rest, not an analyzed note
                else:
                    _, _, note_lo, note_hi, baseline = region
                    lo = max(lo, note_lo)
                    hi = min(hi, note_hi)
            segment = (
                self._segment_values_with_bounds(pitch_data, lo, hi, center)
                if lo < hi
                else None
            )
            vals = segment[0] if segment is not None else None
            sample = (self.vibrato_sample(vals, frame_rate, baseline=baseline)
                      if vals is not None else None)
            if sample is None:
                # Computed-but-no-oscillation is real information, distinct
                # from an unwritten/not-yet-computed NaN grid slot. Zero keeps
                # the UI curve continuous through unvoiced and non-vibrato
                # moments while note_summary can still ignore it.
                vibrato_data.write(i, 0.0, 0.0, 0.0)
            else:
                vibrato_data.write(i, *sample)
                if coverage is not None:
                    _, source_lo, source_hi = segment
                    coverage.append((source_lo, source_hi, *sample))
            i += 1

        if coverage:
            self._map_fits_to_source_spans(
                vibrato_data,
                coverage,
                first_grid=first_grid,
                last_grid=last_grid,
            )

    def _note_regions(self, pitch_data, note_data, available: int):
        """Sorted note windows as frame-center bounds plus a robust baseline.

        Recompute the median from the pitch frames inside the *current* note
        boundaries instead of trusting an older note summary. MistakeChecker
        can split/merge notes after the first detection pass, and this keeps
        vibrato centered on those final replacement spans.
        """
        if note_data is None or not getattr(note_data, "times", None):
            return [], []

        cfg = self.config
        starts = []
        regions = []
        notes = note_data.read(i=0, j=len(note_data.times), clean=True)
        for note in notes:
            pos0 = (
                ((note.start_time - pitch_data.t_origin) * cfg.sr - 0.5 * cfg.w1)
                / cfg.h1
            )
            pos1 = (
                ((note.end_time - pitch_data.t_origin) * cfg.sr - 0.5 * cfg.w1)
                / cfg.h1
            )
            note_lo = max(0, int(ceil(pos0)))
            note_hi = min(available, int(floor(pos1)) + 1)
            if note_lo >= note_hi:
                continue
            voiced = [
                p.value
                for p in pitch_data.data[note_lo:note_hi]
                if pitch_data.is_voiced_pitch(p, include_transitions=False)
            ]
            if not voiced:
                continue
            starts.append(float(note.start_time))
            regions.append((
                float(note.start_time),
                float(note.end_time),
                note_lo,
                note_hi,
                float(np.median(voiced)),
            ))
        return starts, regions

    def _segment_values(self, pitch_data, lo: int, hi: int,
                        center: int) -> np.ndarray | None:
        """Compatibility wrapper returning only the source values."""
        segment = self._segment_values_with_bounds(pitch_data, lo, hi, center)
        return segment[0] if segment is not None else None

    def _segment_values_with_bounds(
            self, pitch_data, lo: int, hi: int, center: int,
    ) -> tuple[np.ndarray, int, int] | None:
        """Midi values for the contiguous analysis segment around frame
        `center` within [lo, hi): voiced frames with interior dropout gaps
        (<= vib_max_gap_sec, NaN for the fit to bridge). Transition frames and
        longer gaps end the segment. Returns the values plus the absolute
        half-open bounds of the local note span they describe, including a
        tolerated unvoiced edge gap. None means no voiced frame is reachable
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

        def include_edge_gap(edge: int, step: int) -> int:
            """Include the short unvoiced tail associated with this segment.

            Those frames do not enter the numerical fit, but in the note-aware
            result they belong to the same wave characteristic. A hard
            transition or a gap beyond the configured bridge length still
            stops the mapping.
            """
            last, j, gap = edge, edge + step, 0
            while (0 <= j < n and gap < max_gap and not hard[j]
                   and not voiced[j]):
                last = j
                gap += 1
                j += step
            return last

        mapped_start = include_edge_gap(start, -1)
        mapped_end = include_edge_gap(end, 1)
        return vals[start:end + 1], lo + mapped_start, lo + mapped_end + 1

    @staticmethod
    def _map_fits_to_source_spans(
            vibrato_data: VibratoData,
            coverage: list[tuple[int, int, float, float, float]],
            first_grid: int,
            last_grid: int,
    ) -> None:
        """Map successful window fits back to their note-bounded grid spans.

        Difference arrays make the many-window overlap linear in the number of
        fits plus output points instead of repeatedly visiting every frame in
        every window. Fit quality is the evidence weight. Points that no
        credible fit depended on retain their raw zero (no oscillation).
        """
        if last_grid < first_grid:
            return
        stride = vibrato_data.stride
        n = last_grid - first_grid + 1
        rate_diff = np.zeros(n + 1, dtype=np.float64)
        extent_diff = np.zeros(n + 1, dtype=np.float64)
        quality_diff = np.zeros(n + 1, dtype=np.float64)
        weight_diff = np.zeros(n + 1, dtype=np.float64)

        for source_lo, source_hi, rate, extent, quality in coverage:
            # Grid center i represents pitch frame i*stride. Include the grid
            # points belonging to this fit's note-bounded local span.
            i0 = max(first_grid, int(ceil(source_lo / stride)))
            i1 = min(last_grid + 1, int(ceil(source_hi / stride)))
            if i0 >= i1:
                continue
            j0, j1 = i0 - first_grid, i1 - first_grid
            weight = max(float(quality), np.finfo(np.float64).eps)
            for diff, value in (
                    (rate_diff, float(rate) * weight),
                    (extent_diff, float(extent) * weight),
                    (quality_diff, float(quality) * weight),
                    (weight_diff, weight),
            ):
                diff[j0] += value
                diff[j1] -= value

        weights = np.cumsum(weight_diff[:-1])
        covered = weights > 0.0
        if not covered.any():
            return
        rates = np.cumsum(rate_diff[:-1])
        extents = np.cumsum(extent_diff[:-1])
        qualities = np.cumsum(quality_diff[:-1])
        for offset in np.flatnonzero(covered):
            weight = weights[offset]
            vibrato_data.write(
                first_grid + int(offset),
                rates[offset] / weight,
                extents[offset] / weight,
                qualities[offset] / weight,
            )

    # --- the windowed estimator ---
    def vibrato_sample(self, vals: np.ndarray, frame_rate: float,
                       baseline: float | None = None):
        """One analysis segment -> (rate_hz, extent_cents, quality), or None
        when no credible vibrato. `vals` = midi pitches whose NaN runs are
        short dropouts (segment endpoints are always voiced). ``baseline`` is
        the full detected note's voiced median in the note-aware pass."""
        n = len(vals)
        voiced = np.isfinite(vals)
        if n < 8 or voiced.mean() < self.VIB_MIN_VOICED_FRAC:
            return None
        idx = np.arange(n, dtype=np.float64)
        if baseline is None or not np.isfinite(baseline):
            baseline = float(np.median(vals[voiced]))
        centered = vals - baseline
        # The note median is the DC center. Nuisance columns therefore contain
        # only a centered slope plus zero-mean Heavisides for bridged dropouts;
        # this keeps a boundary window referenced to the underlying note rather
        # than letting each clipped fragment invent a new local pitch center.
        vi = np.flatnonzero(voiced)
        min_gap = max(1, int(round(self.STEP_MIN_GAP_SEC * frame_rate)))
        gap_starts = vi[np.flatnonzero(np.diff(vi) > min_gap) + 1]
        if len(gap_starts) > 4:  # that torn, the segment is not a note
            return None
        slope = idx - float(np.median(idx[voiced]))
        nuisance = [slope]
        for gap_start in gap_starts:
            step = (idx >= gap_start).astype(float)
            step -= float(np.mean(step[voiced]))
            nuisance.append(step)
        T = np.column_stack(nuisance)
        beta, *_ = np.linalg.lstsq(T[voiced], centered[voiced], rcond=None)
        # interpolate the RESIDUALS so bridged gaps carry no step energy
        x = np.interp(
            idx,
            idx[voiced],
            centered[voiced] - T[voiced] @ beta,
        )
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
