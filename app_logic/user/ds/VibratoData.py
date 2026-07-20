import threading
from math import ceil, floor

import numpy as np

from algorithms.Config import Config


class VibratoData:
    """Time-indexed vibrato-characteristic track: rate (Hz), extent (± cents)
    and fit quality per point of a uniform pitch-frame grid
    (index i <-> pitch frame i*stride <-> that frame's center time). A computed
    0 Hz / 0 cents sample means no measurable oscillation (including unvoiced
    pitch); NaN is reserved for unwritten/not-yet-computed time. Filled by
    VibratoDetector; never persisted — it derives purely from the pitch
    track, so a cache load just recomputes it.

    Offline note-aware values are source-mapped: every credible centered fit
    contributes its characteristics to the local note span it describes, and
    overlapping estimates are combined. Thus edge frames can carry the
    vibrato characteristics that were inferred using their neighboring
    periods instead of falsely appearing as isolated zero-vibrato points.

    The note association is arithmetic on the uniform grid
    (note_index_range), not a stored map, so it cannot go stale when
    analysis rebuilds the note objects."""

    GROW = 1024

    def __init__(self, config: Config):
        self.config = config
        self.stride = max(1, int(config.vib_stride))
        self.t_origin = 0.0
        self.lock = threading.Lock()
        self.rates = np.full(self.GROW, np.nan, dtype=np.float32)
        self.extents = np.full(self.GROW, np.nan, dtype=np.float32)
        self.qualities = np.full(self.GROW, np.nan, dtype=np.float32)
        self.computed_until = 0  # grid high-water mark (exclusive)
        # Cached by VibratoDetector after the first real pitch frame appears;
        # avoids rescanning a long leading clip gap on every live callback.
        self.source_first_index: int | None = None

    # --- the uniform grid (pitch-frame centers, mirroring Pitch.time) ---
    def grid_dt(self) -> float:
        return self.stride * self.config.h1 / self.config.sr

    def index_time(self, i):
        """App-time of grid index i (vectorizes over numpy arrays)."""
        cfg = self.config
        return self.t_origin + (i * self.stride * cfg.h1 + 0.5 * cfg.w1) / cfg.sr

    def grid_pos(self, t: float) -> float:
        """Fractional grid position of app-time t (inverse of index_time)."""
        cfg = self.config
        return ((t - self.t_origin) * cfg.sr - 0.5 * cfg.w1) / cfg.h1 / self.stride

    def index_range(self, t0: float, t1: float) -> tuple[int, int]:
        """Half-open computed grid range whose center times lie in [t0, t1]."""
        i0 = max(0, ceil(self.grid_pos(t0)))
        i1 = min(self.computed_until, floor(self.grid_pos(t1)) + 1)
        return i0, max(i0, i1)

    def note_index_range(self, note) -> tuple[int, int]:
        """The note -> vibrato-samples association: half-open grid indices
        covering the note's [start, end]."""
        return self.index_range(note.start_time, note.end_time)

    # --- writing (VibratoDetector) ---
    def write(self, i: int, rate: float, extent: float, quality: float):
        with self.lock:
            if i >= len(self.rates):
                grow = max(self.GROW, i + 1 - len(self.rates))
                pad = np.full(grow, np.nan, dtype=np.float32)
                self.rates = np.concatenate([self.rates, pad])
                self.extents = np.concatenate([self.extents, pad.copy()])
                self.qualities = np.concatenate([self.qualities, pad.copy()])
            self.rates[i] = rate
            self.extents[i] = extent
            self.qualities[i] = quality
            self.computed_until = max(self.computed_until, i + 1)

    # --- queries ---
    @staticmethod
    def _median3(a: np.ndarray) -> np.ndarray:
        """3-point median where both neighbors exist: one bad analysis window
        can't flick the curve (isolated nonzero islands drop, single-sample
        dropouts heal); any two agreeing neighbors pass through unchanged."""
        if len(a) < 3:
            return a
        prev, cur, nxt = a[:-2], a[1:-1], a[2:]
        ok = np.isfinite(prev) & np.isfinite(cur) & np.isfinite(nxt)
        out = a.copy()
        out[1:-1] = np.where(ok, np.median(np.vstack([prev, cur, nxt]), axis=0), cur)
        return out

    def curve(self, t0: float, t1: float):
        """(times, rates, extents) over [t0, t1]; NaN where not credible.
        Read-side 3-point median (one extra sample pulled past each end so
        edge values smooth identically) — the stored grid stays raw."""
        i0, i1 = self.index_range(t0, t1)
        j0, j1 = max(0, i0 - 1), min(self.computed_until, i1 + 1)
        with self.lock:
            rates = self.rates[j0:j1].astype(float, copy=True)
            extents = self.extents[j0:j1].astype(float, copy=True)
        rates = self._median3(rates)[i0 - j0:i1 - j0]
        extents = self._median3(extents)[i0 - j0:i1 - j0]
        times = self.index_time(np.arange(i0, i1, dtype=float))
        return times, rates, extents

    def global_characteristic_range(
            self, metric: str,
    ) -> tuple[float, float] | None:
        """Recording-wide min/max for a displayed vibrato characteristic.

        Uses the same median-smoothed values as :meth:`curve` and only samples
        with a positive detected rate. The stored 0 Hz / 0 cents sentinel means
        "no measurable vibrato", so including it would make every recording's
        slow/narrow endpoint zero rather than the least/most subtle vibrato the
        performer actually produced.
        """
        if metric not in {"rate", "extent"}:
            raise ValueError(f"Unknown vibrato metric: {metric}")
        with self.lock:
            rates = self.rates[:self.computed_until].astype(float, copy=True)
            extents = self.extents[:self.computed_until].astype(float, copy=True)
        rates = self._median3(rates)
        extents = self._median3(extents)
        values = rates if metric == "rate" else extents
        detected = (
            np.isfinite(rates)
            & np.isfinite(extents)
            & (rates > 0.0)
        )
        if not detected.any():
            return None
        return float(np.min(values[detected])), float(np.max(values[detected]))

    def at(self, t: float) -> tuple[float, float] | tuple[None, None]:
        """(rate, extent) at the grid point nearest app-time t (may be NaN)."""
        i = int(round(self.grid_pos(t)))
        if not (0 <= i < self.computed_until):
            return None, None
        with self.lock:
            return float(self.rates[i]), float(self.extents[i])

    def note_summary(self, note) -> tuple[float, float] | tuple[None, None]:
        """Per-note median (rate_hz, extent_cents), or (None, None). A note
        must itself be long enough to contain config.vib_min_cycles at its
        median estimated rate; there are no fixed Hz/cents gates."""
        i0, i1 = self.note_index_range(note)
        with self.lock:
            rates = self.rates[i0:i1].astype(float, copy=True)
            extents = self.extents[i0:i1].astype(float, copy=True)
        mask = np.isfinite(rates) & np.isfinite(extents) & (rates > 0.0)
        if not mask.any():
            return None, None
        rate = float(np.median(rates[mask]))
        extent = float(np.median(extents[mask]))
        note_duration = max(0.0, note.end_time - note.start_time)
        if note_duration * rate < max(0.0, float(self.config.vib_min_cycles)):
            return None, None
        return rate, extent

    def trim_to(self, t: float):
        """Drop samples past app-time t (mirrors the take's trim_end)."""
        self.computed_until = min(self.computed_until,
                                  max(0, floor(self.grid_pos(t)) + 1))
