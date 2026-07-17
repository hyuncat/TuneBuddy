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
        - highlights:  the pointer box around a selected mistake's / clicked
          note(s), colored by what it points at (see Colors.highlight_style)
    """

    def __init__(self, plot: pg.PlotWidget):
        self.alignment: Alignment | None = None
        # which kind of mistake the pointer flags (mirrors the MistakeWidget's
        # "Mistakes:" dropdown), and the user note it points at — kept so a mode
        # switch can re-color a live selection in place (see set_mode).
        self.mode = "pitch"
        self._pointed_note: Note | None = None
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
        self._set_state("clean")

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
    def set_mode(self, mode: str):
        """Which kind of mistake reddens the pointer ("pitch" | "timing"). A
        note it's already pointing at is re-colored in place: one played in tune
        but late goes blue -> red the moment Timing is picked."""
        mode = "timing" if mode == "timing" else "pitch"
        if mode == self.mode:
            return
        self.mode = mode
        if self._pointed_note is not None:
            self.highlight_user_note(self._pointed_note)

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
        # the list handed us this exact mistake, so its own state colors the
        # pointer whatever mode is showing; there's no note to re-judge later.
        self._pointed_note = None
        return self._highlight_notes(
            notes, "overridden" if mistake.is_overridden() else "mistake")

    def highlight_user_note(self, note: Note) -> float | None:
        """Point at a detected user note AND the score note it was aligned to,
        reddened only if the current mode flags it. Returns their mean center
        time (None if there's nothing to box)."""
        self._pointed_note = note
        match = self.alignment.get_match(user_note=note) if self.alignment else None
        return self._highlight_notes([note, match], self._note_state(note))

    def _note_state(self, note: Note) -> str:
        """A pointed-at note's color state: red while the mode's mistakes flag
        it, green once every one of them is dismissed, blue when it's clean."""
        mistakes = self.alignment.mistakes_for(note, self.mode) if self.alignment else []
        if not mistakes:
            return "clean"
        return "mistake" if any(not m.is_overridden() for m in mistakes) else "overridden"

    def _highlight_notes(self, notes: list[Note], state: str) -> float | None:
        """Box the given note(s) in `state`'s color — how any mistake/selected
        note is highlighted. Returns their mean center time (None if there's
        nothing to box)."""
        notes = [n for n in notes if n is not None]
        if not notes:
            return None
        self.highlights.setOpts(**NoteDataUI.bar_opts(
            [n.start_time for n in notes],
            [n.end_time for n in notes],
            [n.midi_num[0] for n in notes],
            height=NoteDataUI.NOTE_HEIGHT * 2,
        ))
        self._set_state(state)
        return float(np.mean([0.5 * (n.start_time + n.end_time) for n in notes]))

    def override_mistake(self, overridden: bool):
        """Recolor the pointer when the mistake it points at is dismissed
        (green) or restored (red)."""
        self._set_state("overridden" if overridden else "mistake")

    def _set_state(self, state: str):
        brush, pen = Colors.highlight_style(state)
        self.highlights.setOpts(brush=brush, pen=pen)

    def clear_highlight(self):
        self._pointed_note = None
        self.highlights.setOpts(x=[], width=[], y0=[], height=[])
        self._set_state("clean")  # reset color for next use

    def clear(self):
        """Empty every overlay item and drop the alignment."""
        self.alignment = None
        self.match_lines.setData(x=[], y=[])
        self.insertions.setOpts(x=[], width=[], y0=[], height=[])
        self.deletions.setOpts(x=[], width=[], y0=[], height=[])
        self.clear_highlight()
