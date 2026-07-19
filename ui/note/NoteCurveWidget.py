import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from app_logic.user.ds.Recording import Recording
from ui.Colors import Colors


class NoteCurveWidget(QWidget):
    """Base for the note-detail panel's graphs (Volume / Vibrato / Timbre):
    one pyqtgraph plot windowed to the note under the cursor.

    Owns everything the graphs share — the plot (near-white axes, app-time
    seconds on x), the timeline, the flat-color pitch contour, and the window
    state machine:
      - review: the analyzed user note containing the slider time
        (NoteData.note_containing); blank — curves cleared, axes kept —
        during rests / unanalyzed takes.
      - live (recording): a trailing LIVE_WINDOW_SEC window with the timeline
        pinned at LIVE_TIMELINE_FRAC, x still in ground-truth app-time.
    Subclasses draw their own curve in _render()/_render_blank() and may
    override _contour_transform (Timbre's y-axis IS midi, so it uses
    identity). The first render accepts a subclass's padded default y-range;
    after that, native y pan/zoom is persistent while note/live movement only
    replaces the x window, matching GuitarHero's stored view-state behavior."""

    CONTOUR_ROLE = "volume"   # key into Colors.NOTE_CONTOUR_RGB
    HELP = ""                 # the panel's (?) blurb for this graph
    LIVE_WINDOW_SEC = 3.0
    LIVE_TIMELINE_FRAC = 0.75
    # The shared slider is quantized to milliseconds, so seeking to an exact
    # onset can round down just before the note. As with ScoreViewer's playback
    # cursor, use a small display-only lookahead to prefer the newly selected
    # note without moving the transport or the timeline itself.
    NOTE_ONSET_LOOKAHEAD_SEC = 0.010
    # contour normalization: midi mapped into this fraction band of the
    # y-range, over the window's own midi span floored at CONTOUR_MIN_SPAN
    # semitones (so a dead-flat note doesn't amplify jitter)
    CONTOUR_BAND = (0.2, 0.8)
    CONTOUR_MIN_SPAN = 1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self.recording: Recording | None = None
        self.live = False
        self.t = 0.0
        self._note = None                              # review-mode current note
        self._window: tuple[float, float] | None = None  # (t0, t1) rendered
        self._yr = (0.0, 1.0)
        self._stored_y_range: tuple[float, float] | None = None
        self._moving_view = False

        # The app supplies the initial note/live window, but the PlotWidget
        # keeps pyqtgraph's native interaction: left-drag pans, the wheel zooms,
        # right-drag zooms one axis, and the A button restores auto-range.
        axis_items = self._axis_items()
        self.plot = pg.PlotWidget(axisItems=axis_items) if axis_items else pg.PlotWidget()
        self.plot.setBackground(Colors.PLOT_BG_RGB)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.enableAutoRange("xy", False)
        self.plot.showButtons()
        self.plot.setMenuEnabled(True)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().setClipToView(True)
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.plot.setLabel("bottom", "")
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(Colors.note_axis_pen())
            axis.setTextPen(Colors.note_axis_pen())
        self._layout.addWidget(self.plot)

        # A traditional rotated y-axis title consumes too much of this narrow
        # side panel. Keep numeric ticks compact and put the unit inside the
        # plot at its upper-left, repositioned whenever the user pans/zooms.
        self.y_unit = pg.TextItem(
            text="", color=Colors.NOTE_AXIS_RGB, anchor=(0, 0),
            fill=pg.mkBrush(*Colors.PLOT_BG_RGB, 210),
        )
        self.y_unit.setZValue(10)
        self.plot.addItem(self.y_unit, ignoreBounds=True)
        self.y_unit.hide()
        self.x_unit = pg.TextItem(
            text="Time (s)", color=Colors.NOTE_AXIS_RGB, anchor=(1, 1),
            fill=pg.mkBrush(*Colors.PLOT_BG_RGB, 210),
        )
        self.x_unit.setZValue(10)
        self.plot.addItem(self.x_unit, ignoreBounds=True)
        self.plot.getViewBox().sigRangeChanged.connect(
            self._on_view_range_changed)

        self.contour = pg.PlotDataItem(
            pen=Colors.note_contour_pen(self.CONTOUR_ROLE), connect="finite")
        self.contour.setZValue(1)
        self.plot.addItem(self.contour)
        self.timeline = pg.InfiniteLine(
            pos=0, angle=90, pen=Colors.note_timeline_pen())
        self.timeline.setZValue(4)
        self.plot.addItem(self.timeline)
        self.timeline.hide()

    # --- public API (driven by NotePanel) ---
    def set_recording(self, rec: Recording | None):
        self.recording = rec
        self.refresh()

    def set_live(self, live: bool):
        if live == self.live:
            return
        self.live = live
        self.refresh()

    def update_time(self, t: float):
        """Every tick. Cheap when the window is unchanged (timeline only)."""
        self.t = t
        window = self._current_window()
        if window != self._window:
            self._window = window
            self._redraw()
        self._move_timeline()

    def refresh(self):
        """The data behind the views changed — force a full redraw."""
        self._window = self._current_window()
        self._redraw()
        self._move_timeline()

    def header_widgets(self) -> list[QWidget]:
        """Extra controls the panel lays into its header for this graph."""
        return []

    def _axis_items(self) -> dict | None:
        """Optional pyqtgraph axis replacements supplied by a subclass."""
        return None

    # --- subclass hooks ---
    def _render(self, t0: float, t1: float):
        """Rebuild the graph's own curve items for the window (set the
        y-range here — the contour is mapped into it afterwards)."""

    def _render_blank(self):
        """Clear the graph's own curve items (axes stay)."""

    def set_default_y_range(self, y0: float, y1: float,
                            padding: float = 0.15):
        """Set the initial y view once; later renders preserve user pan/zoom.

        `padding` is a fraction of the supplied span on EACH side. A flat
        range gets a scale-aware fallback span so it still has visible room.
        """
        if self._stored_y_range is not None:
            self._yr = self._stored_y_range
            return
        y0, y1 = sorted((float(y0), float(y1)))
        span = y1 - y0
        if span <= 0:
            span = max(abs(y0), abs(y1), 1.0)
        pad = max(0.0, float(padding)) * span
        self.set_y_range(y0 - pad, y1 + pad)

    def set_y_range(self, y0: float, y1: float):
        """Force and remember an exact y view (used by explicit UI restores)."""
        target = (float(y0), float(y1))
        self._stored_y_range = self._yr = target
        self._moving_view = True
        try:
            self.plot.setYRange(*target, padding=0)
        finally:
            self._moving_view = False
        self._position_units()

    def reset_y_range(self):
        """Forget the current y view so the next render chooses a new default."""
        self._stored_y_range = None

    def current_y_range(self) -> tuple[float, float] | None:
        return self._stored_y_range

    def set_y_label(self, text: str):
        """Show the y unit inside the plot instead of widening the left axis."""
        self.plot.setLabel("left", "")
        self.y_unit.setText(text)
        self.y_unit.setVisible(bool(text))
        self._position_units()

    def _position_units(self, *_args):
        if not hasattr(self, "y_unit"):
            return
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        dx, dy = x1 - x0, y1 - y0
        if self.y_unit.isVisible():
            self.y_unit.setPos(x0 + 0.02 * dx, y1 - 0.03 * dy)
        self.x_unit.setPos(x1 - 0.02 * dx, y0 + 0.03 * dy)

    def _on_view_range_changed(self, _viewbox, view_range):
        """Persist native pyqtgraph y pan/zoom, like GuitarHero.update_zoom."""
        self._position_units()
        if self._moving_view or self._stored_y_range is None:
            return
        self._stored_y_range = self._yr = tuple(map(float, view_range[1]))

    # --- window state machine ---
    def _current_window(self) -> tuple[float, float] | None:
        if self.recording is None:
            return None
        if self.live:
            w = self.LIVE_WINDOW_SEC
            return (self.t - self.LIVE_TIMELINE_FRAC * w,
                    self.t + (1.0 - self.LIVE_TIMELINE_FRAC) * w)
        self._note = self.recording.note_data.note_containing(
            self.t + self.NOTE_ONSET_LOOKAHEAD_SEC
        )
        if self._note is None:
            return None
        return (self._note.start_time, self._note.end_time)

    def _redraw(self):
        if self._window is None:
            self.contour.setData([], [])
            self._render_blank()
            return
        t0, t1 = self._window
        # Note/live movement dictates x only. Preserve the user's y translation
        # and zoom exactly; suppress range-signal bookkeeping while applying
        # this programmatic move to avoid accumulating floating-point drift.
        self._moving_view = True
        try:
            self.plot.enableAutoRange("xy", False)
            self.plot.setXRange(t0, t1, padding=0)
        finally:
            self._moving_view = False
        self._position_units()
        self._render(t0, t1)
        self._draw_contour(t0, t1)

    def _move_timeline(self):
        if self._window is None:
            self.timeline.hide()
            return
        t0, t1 = self._window
        self.timeline.setPos(min(max(self.t, t0), t1))
        self.timeline.show()

    # --- the shared pitch contour ---
    def _draw_contour(self, t0: float, t1: float):
        times, midis = self.recording.pitch_data.pitch_curve(t0, t1)
        x, y = self._contour_transform(times, midis)
        self.contour.setData(x, y, connect="finite")

    def _contour_transform(self, times: np.ndarray, midis: np.ndarray):
        """Map the midi contour into a band of the current y-range (this
        graph's y units aren't midi). Timbre overrides with identity."""
        if not np.isfinite(midis).any():
            return times, midis
        lo, hi = np.nanmin(midis), np.nanmax(midis)
        center = 0.5 * (lo + hi)
        span = max(hi - lo, self.CONTOUR_MIN_SPAN)
        y0, y1 = self._yr
        b0 = y0 + self.CONTOUR_BAND[0] * (y1 - y0)
        b1 = y0 + self.CONTOUR_BAND[1] * (y1 - y0)
        y = b0 + ((midis - (center - span / 2.0)) / span) * (b1 - b0)
        return times, y
