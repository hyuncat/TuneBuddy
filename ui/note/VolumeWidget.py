import pyqtgraph as pg

from ui.Colors import Colors
from ui.note.NoteCurveWidget import NoteCurveWidget


class VolumeWidget(NoteCurveWidget):
    """The note's loudness curve in dBFS (the app's volume unit), with the
    pitch contour underlaid for context. The review y-range spans the take's
    own [quietest, loudest] (PitchData.volume_range_db), initially padded 15%
    on each side; live starts from [VOL_LIVE_FLOOR_DB, 0]. Native y pan/zoom
    is then preserved by NoteCurveWidget across redraws and note movement."""

    CONTOUR_ROLE = "volume"
    HELP = ("How loud the note under the cursor was over its duration, in "
            "dBFS (decibels below the microphone's digital full scale — 0 is "
            "as loud as the mic can record). The grey line is the pitch "
            "contour, for reading loudness against what was being played.")
    Y_PADDING = 0.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._range_cache = None  # volume_range_db walks every frame: cache per take
        self.curve = pg.PlotDataItem(
            pen=pg.mkPen(*Colors.NOTE_VOLUME_RGB, 255, width=4),
            connect="finite")
        self.curve.setZValue(2)  # above the contour
        self.plot.addItem(self.curve)
        self.set_y_label("dBFS")

    def refresh(self):
        self._range_cache = None
        super().refresh()

    def _render(self, t0: float, t1: float):
        self.set_default_y_range(
            *self._volume_data_range(), padding=self.Y_PADDING)
        times, dbs = self.recording.pitch_data.volume_curve(
            t0, t1, floor_db=self._yr[0])
        self.curve.setData(times, dbs, connect="finite")

    def _render_blank(self):
        self.curve.setData([], [])

    def _volume_data_range(self) -> tuple[float, float]:
        if not self.live:
            if self._range_cache is None:
                self._range_cache = self.recording.pitch_data.volume_range_db()
            vmin, vmax = self._range_cache
            if vmin is not None and vmax is not None:
                return (vmin, vmax)
        return (Colors.VOL_LIVE_FLOOR_DB, 0.0)
