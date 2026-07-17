import numpy as np
import pyqtgraph as pg

from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import Note
from ui.Colors import Colors
from ui.guitarhero.ds.NoteDataUI import NoteDataUI


class AlignmentUI:
    """The alignment overlay on the GuitarHero plot, owning its plot items:
        - match_lines: dashed user<->score lines for matched (good/sub) pairs
        - insertions:  user notes with no score match (green bars)
        - deletions:   score notes the user never played (red bars)
        - highlights:  the box around a selected mistake's / clicked note(s)
          (red while the mistake stands, green once overridden)
    """

    def __init__(self, plot: pg.PlotWidget):
        self.alignment: Alignment | None = None
        colors = Colors.plot_colors()

        self.match_lines = pg.PlotDataItem(x=[], y=[], pen=colors['match_line'])
        self.match_lines.setZValue(2.2)

        self.insertions = pg.BarGraphItem(
            x=[], height=NoteDataUI.NOTE_HEIGHT, y0=0, width=[],
            brush=colors['insertion'], pen=None,
        )
        self.insertions.setZValue(2.1)  # just above the user notes

        self.deletions = pg.BarGraphItem(
            x=[], height=NoteDataUI.NOTE_HEIGHT, y0=0, width=[],
            brush=colors['deletion'], pen=None,
        )
        self.deletions.setZValue(1.1)  # just above the score notes

        self.highlights = pg.BarGraphItem(x=[], height=[], y0=[], width=[])
        self.highlights.setZValue(5)  # above everything
        self.override_mistake(False)

        for item in (self.match_lines, self.insertions, self.deletions,
                     self.highlights):
            plot.addItem(item)

    def load_alignment(self, alignment: Alignment | None):
        self.alignment = alignment

    def read(self, start_time: float, end_time: float):
        """The alignment components in view: (goods, subs, ins, dels)."""
        if self.alignment is None:
            return [], [], [], []
        return self.alignment.get_alignment(start_time, end_time)

    def update_view(self, start_time: float, end_time: float):
        """Redraw the match lines + insertion/deletion bars for the window."""
        goods, subs, ins, dels = self.read(start_time, end_time)

        # ---> MATCH LINES: user-note midpoint <-> score-note midpoint --->
        xs, ys = [], []
        for n, m in goods + subs:
            if n is None or m is None:
                continue
            # rk: np.nan separates line segments from e/o
            xs.extend([0.5 * (n.start_time + n.end_time),
                       0.5 * (m.start_time + m.end_time), np.nan])
            ys.extend([float(n.midi_num[0]), float(m.midi_num[0]), np.nan])
        self.match_lines.setData(x=np.asarray(xs, dtype=np.float32),
                                 y=np.asarray(ys, dtype=np.float32))

        # ---> INSERTION / DELETION BARS (cleared when none in view, so bars
        # from a previous recording/window never linger) --->
        for item, notes in ((self.insertions, ins), (self.deletions, dels)):
            item.setOpts(**NoteDataUI.bar_opts(
                [n.start_time for n in notes],
                [n.end_time for n in notes],
                [n.midi_num[0] for n in notes],
            ))

    # --- MISTAKE / NOTE HIGHLIGHTING ---
    def highlight_mistake(self, mistake: Mistake) -> float | None:
        """Box the note(s) involved in a mistake, colored by its override
        state. Returns the notes' center time so the host can pan to it."""
        if not mistake.user_note and not mistake.midi_note:
            return None

        if mistake.type == "substitution":
            notes = [mistake.user_note, mistake.midi_note]
        elif mistake.type == "insertion":
            notes = [mistake.user_note]
        elif mistake.type == "deletion":
            notes = [mistake.midi_note]
        else:
            # timing mistakes (early / late / short / long): both notes exist, so
            # box them both to make the onset/duration discrepancy visible.
            notes = [n for n in (mistake.user_note, mistake.midi_note)
                     if n is not None]
        return self.highlight_note(notes, overridden=mistake.is_overridden())

    def highlight_note(self, notes: list[Note], overridden: bool = False) -> float | None:
        """Box the given note(s) — how any clicked/selected note is highlighted.
        Returns their mean center time (None if there's nothing to box)."""
        notes = [n for n in notes if n is not None]
        if not notes:
            return None
        self.highlights.setOpts(**NoteDataUI.bar_opts(
            [n.start_time for n in notes],
            [n.end_time for n in notes],
            [n.midi_num[0] for n in notes],
            height=NoteDataUI.NOTE_HEIGHT * 2,
        ))
        self.override_mistake(overridden)
        return float(np.mean([0.5 * (n.start_time + n.end_time) for n in notes]))

    def override_mistake(self, overridden: bool):
        """Swap the highlight color: green if overridden, red if not."""
        brush, pen = Colors.highlight_style(overridden)
        self.highlights.setOpts(brush=brush, pen=pen)

    def clear_highlight(self):
        self.highlights.setOpts(x=[], width=[], y0=[], height=[])
        self.override_mistake(False)  # reset color for next use

    def clear(self):
        """Empty every overlay item and drop the alignment."""
        self.alignment = None
        self.match_lines.setData(x=[], y=[])
        self.insertions.setOpts(x=[], width=[], y0=[], height=[])
        self.deletions.setOpts(x=[], width=[], y0=[], height=[])
        self.clear_highlight()
