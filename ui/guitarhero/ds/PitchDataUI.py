import pyqtgraph as pg

from app_logic.user.ds.PitchData import PitchData, Pitch
from ui.Colors import Colors


class PitchDataUI(pg.ScatterPlotItem):
    """The detected pitch dots on the GuitarHero plot. Owns the PitchData it
    reads from, the pitch/align/volume brush pools (built by Colors), and the
    live/review volume normalization.

    Coloring is per update_view `mode`:
        - "pitch": along the plasma ramp (yellow = on-pitch -> indigo = way
          off) by pitch distance. Post-analysis frames carry an
          `aligned_distance` and use the adaptive ramp (yellow within the
          recording's pitch tolerance — see set_tolerance); otherwise the fixed
          live ramp on `live_distance`. Transitions are grey.
        - "volume": along the truncated viridis ramp (purple = quiet ->
          sea-green = loud). Live recording normalizes against a FIXED absolute
          dBFS window (a dot's color locks the moment it's drawn); in review
          it's remapped against the take's own quietest/loudest frame.
    """

    def __init__(self):
        colors = Colors.plot_colors()
        super().__init__(x=[], y=[], pen=pg.mkPen(None),
                         brush=colors['user_pitch'], size=8)
        self.setZValue(3)  # above user notes
        self.pitch_data: PitchData | None = None

        self.live = False               # recording right now? (see set_live)
        self._vol_range: tuple | None = None  # cached per-take (min, max) dBFS

        self.rest_brush = colors['rest']  # transition frames in pitch mode
        self.distance_brushes = Colors.pitch_brushes(
            Colors.LIVE_CORRECT_THRESH, Colors.LIVE_MAX_DIST)
        self.volume_brushes = Colors.volume_brushes()
        # knocked-back twins of the pooled brushes, made on demand for the
        # hovered note (see hover_brush)
        self._hover_brushes: dict[int, object] = {}
        # adaptive post-analysis ramp; rebuilt per-recording (set_tolerance).
        # Mutated in place so palette refs (GuitarHero.colors) never go stale.
        self.align_distance_brushes: list = []
        self.align_max_dist = 0.0
        self.set_tolerance(0.3)  # default; rebuilt on load

    # --- data ---
    def load_pitchdata(self, pitch_data: PitchData | None, tolerance: float = None):
        """Point at a (new) PitchData; drops the cached per-take volume range.
        `tolerance` (the recording's pitch-mistake tolerance) rebuilds the
        adaptive align ramp when given."""
        self.pitch_data = pitch_data
        self._vol_range = None
        if tolerance is not None:
            self.set_tolerance(tolerance)

    def sync(self, pitch_data: PitchData | None):
        """Per-redraw ref re-assert: analysis swaps Recording.pitch_data behind
        the views, so re-point at it — but only on identity change, keeping the
        volume-range cache warm while scrolling."""
        if pitch_data is not self.pitch_data:
            self.load_pitchdata(pitch_data)

    def set_tolerance(self, tolerance: float):
        """Rebuild the adaptive ramp: yellow within `tolerance` semitones of the
        aligned note, ramping to indigo (see Colors.align_pitch_brushes)."""
        brushes, self.align_max_dist = Colors.align_pitch_brushes(tolerance)
        self.align_distance_brushes[:] = brushes

    def set_live(self, live: bool):
        """Toggle live-recording mode for volume coloring. Live => normalize
        each frame against a fixed absolute dBFS window (its color locks when
        drawn, no re-shade as later notes arrive); review => normalize against
        the take's own quietest/loudest frame."""
        self.live = live
        self._vol_range = None  # recompute the take's range on the next redraw

    def read(self, start_time: float, end_time: float) -> list[Pitch]:
        """The voiced pitch frames in view."""
        if self.pitch_data is None:
            return []
        return self.pitch_data.read(start_time, end_time, clean=True)

    # --- drawing ---
    def update_view(self, start_time: float, end_time: float, mode: str = "pitch",
                    hover: tuple[float, float] | None = None):
        """Redraw the dots for the given time window, colored per `mode`. `hover`
        is the hovered note's time span, whose frames are knocked back so the
        note being pointed at reads apart from the rest of the track."""
        volume_mode = mode == "volume"
        vmin_db, vmax_db = (None, None)
        if volume_mode and not self.live:
            vmin_db, vmax_db = self._review_vol_range()

        xs, ys, brushes = [], [], []
        for p in self.read(start_time, end_time):
            if not p.candidate_pitches:
                continue
            xs.append(p.time)
            ys.append(p.value)  # primary pitch value
            brush = self._brush_for(p, volume_mode, vmin_db, vmax_db)
            if hover is not None and hover[0] <= p.time <= hover[1]:
                brush = self.hover_brush(brush)
            brushes.append(brush)

        self.setData(x=xs, y=ys, brush=brushes)

    def _brush_for(self, p: Pitch, volume_mode: bool, vmin_db, vmax_db):
        """The brush a frame draws with, before any hover knock-back."""
        if volume_mode:
            frac = Colors.volume_frac(getattr(p, "volume", 0.0), vmin_db, vmax_db)
            return self.get_volume_brush(frac)
        if getattr(p, "is_transition", False):
            return self.rest_brush
        ad = getattr(p, "aligned_distance", None)
        if ad:
            return self.get_align_distance_brush(ad)
        return self.get_distance_brush(getattr(p, "live_distance", None))

    def setData(self, *args, **kwargs):
        # ScatterPlotItem.setData flushes old points via self.clear(); flag the
        # re-entry so only DIRECT clear() calls drop the data refs.
        self._in_set_data = True
        try:
            super().setData(*args, **kwargs)
        finally:
            self._in_set_data = False

    def clear(self):
        """Empty the scatter and drop the PitchData (a mid-setData flush only
        empties the points)."""
        if not getattr(self, "_in_set_data", False):
            self.pitch_data = None
            self._vol_range = None
        super().clear()

    # --- brush lookups ---
    def get_distance_brush(self, d: float | None):
        """Live-ramp brush for a distance-to-target (None => transition grey)."""
        if d is None:
            return self.rest_brush
        return Colors.ramp_brush(self.distance_brushes, d, Colors.LIVE_MAX_DIST)

    def get_align_distance_brush(self, d: float):
        """Adaptive-ramp brush for an alignment-based distance. inf (insertions)
        clamps to the max bucket => solid indigo."""
        return Colors.ramp_brush(self.align_distance_brushes, d, self.align_max_dist)

    def get_volume_brush(self, frac: float):
        """Pooled brush for a 0..1 volume fraction."""
        idx = int(max(0.0, min(1.0, frac)) * (len(self.volume_brushes) - 1))
        return self.volume_brushes[idx]

    def hover_brush(self, brush):
        """The knocked-back twin of a pooled brush (see Colors.hover_brush),
        pooled in turn. Keyed by COLOR, not by brush identity, so a ramp rebuilt
        in place (set_tolerance) can never hand back a stale twin."""
        key = brush.color().rgba()
        twin = self._hover_brushes.get(key)
        if twin is None:
            twin = Colors.hover_brush(brush)
            self._hover_brushes[key] = twin
        return twin

    def _review_vol_range(self) -> tuple[float, float] | tuple[None, None]:
        """The take's (min_dBFS, max_dBFS) from PitchData.volume_range_db,
        cached per take. Post-recording the quietest frame maps to the purple
        end of the ramp and the loudest to sea-green; (None, None) falls back
        to the absolute window."""
        if self._vol_range is None:
            self._vol_range = (self.pitch_data.volume_range_db()
                               if self.pitch_data else (None, None))
        return self._vol_range
