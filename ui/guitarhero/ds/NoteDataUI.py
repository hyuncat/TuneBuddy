import numpy as np
import pyqtgraph as pg

from app_logic.NoteData import NoteData, Note
from ui.Colors import Colors


class NoteDataUI(pg.BarGraphItem):
    """A NoteData drawn as piano-roll bars.

    `type` picks the styling and read semantics:
        - "score": white bars, one per chord member (NoteData stores one Note
          per onset with EVERY simultaneous pitch in Note.midi_num — see
          MidiData.make_notedatas), dimmed grey outside an active clip.
        - "user": teal bars, primary pitch only, rests filtered out.
    """

    NOTE_HEIGHT = 0.5  # height of note rectangles (semitones)

    def __init__(self, type: str = "user"):
        colors = Colors.plot_colors()
        brush = colors['user_note'] if type == "user" else colors['midi']
        super().__init__(
            x=[], height=self.NOTE_HEIGHT, width=[], y0=0,
            brush=brush, pen=None,
        )
        self.setZValue(2 if type == "user" else 1)  # user notes above score notes
        self.type = type
        self.brush = brush
        self.dim_brush = colors['midi_dim']  # score notes OUTSIDE the clip
        self.note_data: NoteData | None = None

    @staticmethod
    def bar_opts(starts, ends, midis, height: float = NOTE_HEIGHT) -> dict:
        """setOpts kwargs for bars centered on their (time span, midi) boxes."""
        starts = np.asarray(starts, dtype=np.float64)
        ends = np.asarray(ends, dtype=np.float64)
        midis = np.asarray(midis, dtype=np.float64)
        return dict(
            x=0.5 * (starts + ends), width=ends - starts,
            y0=midis - 0.5 * height, height=np.full_like(midis, height),
        )

    def load_notedata(self, note_data: NoteData | None):
        self.note_data = note_data

    def read(self, start_time: float, end_time: float) -> list[Note]:
        """The notes in view; user notes skip rests (clean)."""
        if self.note_data is None:
            return []
        return self.note_data.read(
            start_time=start_time, end_time=end_time, clean=(self.type == "user"),
        )

    def update_view(self, start_time: float, end_time: float,
                    clip: tuple[float, float] | None = None):
        """Redraw the bars for the given time window. For score notes, `clip`
        (the [b0, b1] window) draws notes outside it dimmer grey — a note is
        "in the clip" iff its START is in [b0, b1), matching the alignment."""
        if self.type == "user":
            clip = None  # only score notes dim outside the clip
        notes = self.read(start_time, end_time)

        starts, ends, midis, brushes = [], [], [], []
        for n in notes:
            in_clip = clip is None or (clip[0] - 1e-6 <= n.start_time < clip[1] - 1e-6)
            brush = self.brush if in_clip else self.dim_brush
            for m in n.midi_num:
                if m == -1:  # rest / unvoiced placeholder, nothing to draw
                    continue
                starts.append(n.start_time)
                ends.append(n.end_time)
                midis.append(m)
                brushes.append(brush)
                if self.type == "user":
                    break  # user bars show the primary pitch only

        # per-bar brushes only needed when clipped (else every bar is default)
        self.setOpts(brush=self.brush,
                     brushes=(brushes if clip is not None else None),
                     **self.bar_opts(starts, ends, midis))

    def clear(self):
        """Empty the bars and drop the NoteData."""
        self.note_data = None
        self.setOpts(x=[], width=[], y0=[], height=[])
