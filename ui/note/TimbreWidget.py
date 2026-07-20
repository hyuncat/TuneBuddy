import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, QRectF

from ui.Colors import Colors
from ui.guitarhero.MidiBackground import MidiAxis
from ui.info.Gradient import TimbreGradient
from ui.info.Legend import Legend
from ui.note.NoteCurveWidget import NoteCurveWidget


class TimbreWidget(NoteCurveWidget):
    """Semitone spectrum heatmap for the note under the cursor."""

    CONTOUR_ROLE = "timbre"
    HELP = ("Timbre shows the note's spectrum in semitone-spaced bins. Hotter "
            "colors mean more energy at that frequency; stacked bright bands "
            "are the fundamental and harmonics. The grey line is the detected "
            "pitch contour.")
    LIVE_LEVELS = (-80.0, 0.0)

    def _axis_items(self):
        return {"left": MidiAxis(orientation="left")}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(Colors.magma_lut())
        self.image.setZValue(0)
        self.plot.addItem(self.image)
        self.contour.setZValue(2)
        self.set_y_label("Pitch")
        self._legend_levels = None
        self.legend = None

        self._backfill_timer = QTimer(self)
        self._backfill_timer.setInterval(150)
        self._backfill_timer.timeout.connect(self._poll_backfill)
        self._set_levels(self.LIVE_LEVELS)

    def set_recording(self, rec):
        super().set_recording(rec)
        self._ensure_timbre()

    def refresh(self):
        super().refresh()
        self._ensure_timbre()

    def _ensure_timbre(self):
        rec = self.recording
        if rec is None or not rec.timbre_data.is_empty():
            return
        if rec.ensure_timbre():
            self._backfill_timer.start()

    def _poll_backfill(self):
        rec = self.recording
        if rec is None:
            self._backfill_timer.stop()
            return
        self._window = self._current_window()
        self._redraw()
        thread = getattr(rec, "_timbre_thread", None)
        if thread is None or not thread.is_alive():
            self._backfill_timer.stop()

    def _render(self, t0: float, t1: float):
        data = self.recording.timbre_data
        y0, y1 = data.midi_min - 0.5, data.midi_max + 0.5
        self.set_default_y_range(y0, y1, padding=0.0)
        levels = self.LIVE_LEVELS if self.live else data.range_db()
        self._set_levels(levels)
        _times, matrix = data.matrix(t0, t1)
        if matrix.shape[1] == 0:
            self._render_blank()
            return
        self.image.setImage(matrix, autoLevels=False, levels=levels)
        self.image.setRect(QRectF(t0, y0, max(t1 - t0, 1e-9), y1 - y0))

    def _render_blank(self):
        self.image.setImage(np.empty((0, 0)), autoLevels=False)

    def _contour_transform(self, times, midis):
        return times, midis

    def _set_levels(self, levels):
        levels = (float(levels[0]), float(levels[1]))
        if levels[1] <= levels[0]:
            levels = (levels[0], levels[0] + 1.0)
        if self._legend_levels == levels:
            return
        self._legend_levels = levels
        if self.legend is not None:
            self._layout.removeWidget(self.legend)
            self.legend.hide()
            self.legend.deleteLater()
        self.legend = Legend.gradient_strip(
            TimbreGradient(*levels), width=110, help=False)
        self._layout.addWidget(self.legend)
