from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from ui.Colors import Colors


class MidiBackground(pg.ImageItem):
    """The GuitarHero plot's static scenery, all owned here:
        - the fixed 0..127 MIDI color-stripe texture (this ImageItem)
        - the pooled beat/measure gridlines (InfiniteLines)
        - the clip bounds: two dim bands darkening everything outside the clip
          (LinearRegionItems)
    Adds itself and its items to `plot` with ignoreBounds so none of it ever
    affects autorange."""

    N_MIDI = 128
    GRIDLINE_Z = 0  # above the stripes (-1), below the notes (1)

    def __init__(self, plot: pg.PlotWidget):
        super().__init__(axisOrder='row-major')
        self.setZValue(-1)  # behind everything
        self.plot = plot
        self.colors = Colors.plot_colors()
        self._init_bg()
        plot.addItem(self, ignoreBounds=True)

        # --- beat / measure gridlines ---
        # lines for measure starts are thicker than beat lines; generated from
        # score_data.beats, the pool grows lazily to the max # beats visible
        self.gridlines: list[pg.InfiniteLine] = []

        # --- clip bounds ---
        # when the score is clipped, the regions OUTSIDE [b0, b1] are darkened:
        # two translucent black bands (left of b0, right of b1) drawn above the
        # stripes/gridlines but below the notes (so score notes keep their own
        # dimmed-grey brush — see NoteDataUI). hidden when not clipped.
        self._last_clip = None  # cache so the bands only re-position on change
        dim = self.colors['clip_dim']
        self.clip_bounds = tuple(
            pg.LinearRegionItem(
                values=(0, 0), orientation='vertical', brush=dim,
                pen=pg.mkPen(None), hoverBrush=dim, hoverPen=pg.mkPen(None),
                movable=False,
            )
            for _ in range(2)
        )
        for region in self.clip_bounds:
            region.setZValue(0.5)  # over stripes/gridlines, under notes (z>=1)
            region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            region.setAcceptHoverEvents(False)
            for line in getattr(region, "lines", []):
                line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                line.setAcceptHoverEvents(False)
            region.hide()
            plot.addItem(region, ignoreBounds=True)

    # --- the MIDI stripe texture ---
    def _init_bg(self):
        """Build a fixed 0..127 MIDI RGBA texture (height=128 rows, one per MIDI).
        This never changes, so colors are locked to absolute MIDI."""
        # width can be tiny; GPU stretches it. Use width=2 for stability.
        arr = np.zeros((self.N_MIDI, 2, 4), dtype=np.ubyte)
        for m in range(self.N_MIDI):
            arr[m, :, :] = Colors.midi_rgba(m)
        self.setImage(arr, autoLevels=False)

        # pin the image's Y rect to the MIDI domain forever;
        # set X span to default dummy values
        self.update_x(-1, 4)

    def update_x(self, xmin: float, xmax: float):
        """Update the image's X span; uses setRect() to change only X, keeping
        Y locked to the MIDI domain 0..128 (1 unit = 1 MIDI)."""
        self.setRect(pg.QtCore.QRectF(xmin, 0.325, xmax - xmin, 128.325))

    # --- gridlines ---
    def _get_gridline(self, idx: int) -> pg.InfiniteLine:
        """Return the idx-th pooled gridline, lazily creating (and adding) it."""
        while idx >= len(self.gridlines):
            line = pg.InfiniteLine(angle=90, pen=self.colors['beat'])
            line.setZValue(self.GRIDLINE_Z)
            self.plot.addItem(line, ignoreBounds=True)
            self.gridlines.append(line)
        return self.gridlines[idx]

    def update_gridlines(self, x_range: tuple[float, float], beats):
        """Update the beat/measure gridlines to fit the given x_range.

        `beats` is score_data.beats (the metronome beatmap): (time_sec,
        is_downbeat) tuples. Downbeats are measure starts and get the thicker
        'measure' pen; the rest get the thinner 'beat' pen."""
        if not beats:
            for line in self.gridlines:
                line.hide()
            return

        xmin, xmax = x_range
        idx = 0
        for beat_time, is_downbeat in beats:
            if beat_time < xmin or beat_time > xmax:
                continue
            line = self._get_gridline(idx)
            line.setPos(beat_time)
            line.setPen(self.colors['measure'] if is_downbeat else self.colors['beat'])
            line.show()
            idx += 1

        # hide any pooled lines left over from a wider/denser view
        for j in range(idx, len(self.gridlines)):
            self.gridlines[j].hide()

    # --- clip bounds ---
    def update_clip_bounds(self, clip: tuple[float, float] | None):
        """Darken the area OUTSIDE the clip's [b0, b1] window (hidden when
        unclipped). Only re-positions the bands when the window actually
        changes, otherwise pyqtgraph just transforms the existing static bands
        as the view scrolls, which avoids the per-tick setRegion flicker."""
        if clip == self._last_clip:
            return
        self._last_clip = clip
        left, right = self.clip_bounds
        if clip is None:
            left.hide()
            right.hide()
            return
        b0, b1 = clip
        BIG = 1e6  # well past any view; small enough to avoid transform precision jitter
        left.setRegion((-BIG, b0))
        right.setRegion((b1, BIG))
        left.show()
        right.show()


class MidiAxis(pg.AxisItem):
    """
    Overloaded pyqtgraph AxisItem to display y-axis as note names
    rather than as raw MIDI numbers. Eg, 60 -> C4.
    """
    NOTE_NAMES = [
        'C', 'C#', 'D', 'D#', 'E', 'F',
        'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyle(autoExpandTextSpace=True)

    def tickValues(self, minVal, maxVal, size):
        """
        Return tick levels as [(majorStep, majorValues), (minorStep, minorValues)].
        We choose a semitone-based major step based on zoom span.
        """
        span = float(maxVal - minVal)
        if span <= 0:
            return []

        # target ~8–12 major labels depending on pixel height
        target_labels = max(6, min(12, int(size / 35)))

        # candidate steps in semitones
        candidates = np.array([1, 2, 3, 4, 6, 12, 24, 36, 48], dtype=int)
        # pick the smallest step that yields <= target_labels
        labels_per_span = span / candidates
        try:
            major_step = int(candidates[np.argmax(labels_per_span <= target_labels)])
            if labels_per_span.max() > target_labels and major_step == 0:
                major_step = int(candidates[-1])
        except Exception:
            major_step = 12  # sane default
        if major_step <= 0:
            major_step = 12

        # align majors to the step boundary
        start_major = int(np.floor(minVal / major_step) * major_step)
        end_major   = int(np.ceil (maxVal / major_step) * major_step)
        majors = np.arange(start_major, end_major + 1, major_step, dtype=int)

        # minors at 1 semitone (only when not too dense)
        if major_step >= 6:
            start_minor = int(np.floor(minVal))
            end_minor   = int(np.ceil (maxVal))
            minors = np.arange(start_minor, end_minor + 1, 1, dtype=int)
            # drop those that coincide with majors
            minors = minors[~np.isin(minors, majors)]
            return [(major_step, majors), (1, minors)]
        else:
            return [(major_step, majors)]

    def tickStrings(self, values, scale, spacing):
        """
        Label only the first tick level (majors). Pyqtgraph passes majors first.
        Values for minors will be ignored by this method for that level.
        """
        # values can be floats; they are exactly integers from our tickValues
        return [self.midi_to_name(int(round(v))) for v in values]

    @staticmethod
    def midi_to_name(m: int) -> str:
        """Convert MIDI number to name, e.g. 60 -> C4."""
        pitch = m % 12
        octave = (m // 12) - 1
        return f"{MidiAxis.NOTE_NAMES[pitch]}{octave}"
