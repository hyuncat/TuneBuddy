from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
import qdarktheme


from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.Recording import Recording
from app_logic.user.NoteInfo import NoteInfo
from app_logic.Alignment import Alignment
from ui.Colors import Colors
from ui.guitarhero.MidiBackground import MidiBackground, MidiAxis
from ui.guitarhero.ds.PitchDataUI import PitchDataUI
from ui.guitarhero.ds.NoteDataUI import NoteDataUI
from ui.guitarhero.ds.AlignmentUI import AlignmentUI
from ui.guitarhero.ds.NotePopupGH import NotePopupGH
from ui.info.Gradient import PitchGradient, VolumeGradient
from ui.info.Legend import Legend


class GuitarHero(QWidget):
    """The scrolling piano-roll: score notes, detected user notes/pitches, and
    the alignment overlay, coordinated over a shared plot. The heavy lifting
    lives in the per-layer items:
        - MidiBackground  — MIDI stripes + gridlines + clip dim bands
        - NoteDataUI      — score / user note bars
        - PitchDataUI     — pitch dots + their color ramps
        - AlignmentUI     — match lines, ins/del bars, mistake highlights
    This widget owns the viewbox/timeline, the data refs, the legend row, and
    the note-info popup."""

    plot_moved = pyqtSignal(float) # emits plot time in seconds

    # Pitch-dot coloring is switchable via the "Colors:" dropdown (see the legend
    # row): "pitch" colors each dot along the plasma ramp (yellow = on-pitch ->
    # indigo = way off) by pitch distance; "volume" colors it along a truncated
    # viridis ramp (purple = quiet -> sea-green = loud), which reads more clearly
    # than opacity. Both run FULL strength here — only the score dims them
    # (Colors.SCORE_DIM). See PitchDataUI.

    # combobox style mirrors the "Instrument" select (SettingsWidget._COMBO_STYLE)
    _COMBO_STYLE = """
        QComboBox {
            padding-left: 6px;
            padding-right: 22px;
        }
        QComboBox QAbstractItemView::item {
            padding: 2px 8px 2px 6px;
            min-height: 24px;
        }
    """

    def __init__(self, recording: Recording=None):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # important objects
        self.recording = recording
        self.score_data = recording.score_data if recording else None
        self.alignment: Alignment = recording.alignment if recording else None

        # pitch-dot coloring: "pitch" (distance) or "volume" (see PitchDataUI)
        self.color_mode = "pitch"

        # --- TIMEKEEPING VARIABLES ---
        # windowing variables
        self.w, self.h = 5, 50 # seconds, midi numbers
        self.x_range, self.y_range = (-1, 4), (40, 90) # initial view ranges
        self.x_margin = 0.5 # 50% margin on the x-axis

        # timeline variables
        self.t = 0 # current time in seconds
        self.timeline_offset = 0.2 # x fraction of screen from left
        self.is_moving = False

        # ---- THE PLOT. ----
        self.y_axis = MidiAxis(orientation='left') # our custom y-axis
        self.plot = pg.PlotWidget(axisItems={'left': self.y_axis})
        self._layout.addWidget(self.plot)

        self.init_background()
        self.init_colors()
        self.init_objects()
        self.init_view()
        self.init_legend()

        # left-clicking a detected user note pops up its per-note characteristics
        self.note_popup: NotePopupGH | None = None
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        # hovering a detected user note shows a clickable (pointing-hand) cursor
        self._hover_cursor_on = False
        self.plot.scene().sigMouseMoved.connect(self._on_plot_hover)

    # --- INIT STUFF ---
    def init_background(self):
        self.plot.setBackground((20,20,25))
        # our colorful custom MIDI background — also owns the gridlines + the
        # clip dim bands (it adds itself + its items to the plot)
        self.bg = MidiBackground(self.plot)

    def init_colors(self):
        """The shared palette (see ui.Colors); init_objects mirrors the pitch/
        volume dot gradients into it too."""
        self.colors = Colors.plot_colors()

    def init_objects(self):
        """Initialize all foreground plot items: score/user note bars, pitch
        dots, the alignment overlay, and the timeline."""
        self.midi_notes = NoteDataUI(type="score")
        self.user_notes = NoteDataUI(type="user")
        self.user_pitches = PitchDataUI()
        self.timeline = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(self.colors['timeline'])
        )
        self.timeline.setZValue(4) # above everything (but the highlights)

        self.plot.addItem(self.midi_notes)
        self.plot.addItem(self.user_notes)
        self.plot.addItem(self.user_pitches)
        self.plot.addItem(self.timeline)
        # match lines / ins / dels / mistake highlights (adds its own items)
        self.alignment_ui = AlignmentUI(self.plot)

        # the dot gradients are owned by PitchDataUI (the adaptive one is
        # rebuilt in place per recording); mirror them into the palette so
        # every plot color is reachable from `colors`.
        self.colors['distance_brushes'] = self.user_pitches.distance_brushes
        self.colors['align_distance_brushes'] = self.user_pitches.align_distance_brushes
        self.colors['volume_brushes'] = self.user_pitches.volume_brushes

    def init_view(self):
        """initialize the viewbox settings"""
        vb = self.plot.getViewBox()
        self.plot.enableAutoRange('xy', False)
        vb.setLimits(yMin=0, yMax=128)

        # set initial ranges
        vb.setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        vb.sigRangeChanged.connect(self.update_zoom)

    # ---------- PAN/ZOOM HANDLING ----------
    def update_zoom(self, viewbox, view_range):
        """updates the zoom of the plot when the viewbox range changes
        also updates the background accordingly
        """
        # ignore update_zoom calls while moving to avoid error accumulation
        if self.is_moving:
            return

        xmin, xmax = self.plot.viewRange()[0]
        self.bg.update_x(xmin, xmax)

        # store the new ranges
        self.x_range = view_range[0]
        self.y_range = view_range[1]
        self.w = self.x_range[1] - self.x_range[0]
        self.h = self.y_range[1] - self.y_range[0]

        self.update_view_items()

    def move_plot(self, t: float):
        """Move the plot to time t (sec).
        Update the window boundaries, the background, and our viewbox range.
        Note that we keep a "is_moving" flag on to avoid accumulating errors
        in the auto-zoom logic.

        Args:
            t (float): time in seconds to move the plot to
        """
        self.is_moving = True # avoid accumulating errors in zoom

        # update the window boundaries
        self.t = t # update current time
        x_lower = self.t - (self.w*self.timeline_offset)
        x_upper = self.t + (self.w * (1-self.timeline_offset))
        self.x_range = (x_lower, x_upper)

        # update the background and our viewbox range
        self.bg.update_x(x_lower, x_upper)
        self.plot.getViewBox().setRange(xRange=self.x_range, yRange=self.y_range, padding=0)
        self.timeline.setPos(t) # also update the timeline pos

        self.is_moving = False # now we good
        self.update_view_items()
        self.plot_moved.emit(t)


    # ---------- DATA LOADING ----------
    def load_score(self, score_data: ScoreData):
        """Load a ScoreData object and display its notes."""
        print("Loading MIDI data into GuitarHero...")
        self.score_data = score_data
        self.update_view_items()

    def load_user(self, recording: Recording):
        """Load a Recording object and display its notes and pitches."""
        print(f"Loading Recording: {recording} into GuitarHero...")
        self.recording = recording
        self.score_data = recording.score_data
        self.alignment = recording.alignment
        # fresh volume range for the fade + the green band / red ramp track
        # this recording's pitch-mistake tolerance
        self.user_pitches.load_pitchdata(
            recording.pitch_data, tolerance=recording.config.pitch_tolerance)
        self.clear_highlight()
        self.update_view_items()

    def load_alignment(self, alignment: Alignment):
        """Plot the alignment results (user notes + mistakes)."""
        print("Plotting alignment...")
        self.alignment = alignment
        self.update_view_items()

    # --- THE ESSENTIAL PLOTTING STUFF (called every time we refresh the view) ---
    def update_view_items(self):
        """Force all view items to update/redraw. Called whenever:
        1. view range changes
        2. view items change
            2.1 midi/user data loaded
            2.2 alignment loaded
            2.3 pitch detected
        """
        PAD = 1
        xmin, xmax = self.plot.viewRange()[0]
        x_range = (xmin-PAD, xmax+PAD)

        # analysis / tempo changes rebuild the underlying data objects behind
        # the views, so re-assert each item's data ref before redrawing
        # (identity-checked where it matters, so this is ~free per tick).
        self.midi_notes.load_notedata(self._active_score_note_data())
        self.user_notes.load_notedata(self.recording.note_data if self.recording else None)
        self.user_pitches.sync(self.recording.pitch_data if self.recording else None)
        self.alignment_ui.load_alignment(self.alignment)

        self.bg.update_gridlines(x_range, self.score_data.beats if self.score_data else None)
        self.user_pitches.update_view(*x_range, mode=self.color_mode)
        self.user_notes.update_view(*x_range)
        self.midi_notes.update_view(*x_range, clip=self._active_clip_window())
        self._sync_alignment()
        self.alignment_ui.update_view(*x_range)
        self.update_clip_overlay()

    def _active_score_note_data(self):
        """The active instrument's current NoteData, or None."""
        if self.score_data is None:
            return None
        return self.score_data.note_datas.get(self.score_data.active_instrument)

    def _active_clip_window(self) -> tuple[float, float] | None:
        """The clip's [b0, b1] window (derived from note indices), or None.
        Single source of truth: ScoreData.get_bounds(respect_clip=True)."""
        if self.score_data is None or not self.score_data.is_clipped():
            return None
        return self.score_data.get_bounds(respect_clip=True)

    def _sync_alignment(self):
        """Relink the alignment's score-side note refs after score timing is
        rebuilt (see Alignment.sync_score_notes)."""
        if self.alignment is None:
            return
        self.alignment.sync_score_notes(self._active_score_note_data())

    def update_clip_overlay(self):
        """(Re)position the dim bands outside the clip (see MidiBackground)."""
        self.bg.update_clip_bounds(self._active_clip_window())

    # ---------- NOTE INFO POPUP ----------
    def _on_plot_clicked(self, ev):
        """Left-click on a detected user note -> per-note characteristics popup."""
        if ev.button() != Qt.MouseButton.LeftButton or ev.double():
            return
        if self.recording is None or not self.recording.note_data.times:
            return

        view_pt = self.plot.getViewBox().mapSceneToView(ev.scenePos())
        note = self._user_note_at(view_pt.x(), view_pt.y())
        if note is None:
            return
        ev.accept()

        if self.note_popup is not None:
            self.note_popup.close()
            self.note_popup.deleteLater()
        note.info = NoteInfo.analyze(self.recording, note)
        self.note_popup = NotePopupGH(note.info, parent=self)
        self.note_popup.popup_at(QCursor.pos())

    def _user_note_at(self, t: float, midi: float):
        """The user note under a click at (time, midi), or None. Bars are drawn
        at midi ± NOTE_HEIGHT/2; the slack is a little wider for clickability."""
        CLICK_SLACK = 0.5  # semitones
        notes = self.recording.note_data.read(start_time=t, end_time=t, clean=True)
        best, best_dist = None, CLICK_SLACK
        for n in notes:
            dist = abs(n.midi_num[0] - midi)
            if dist <= best_dist:
                best, best_dist = n, dist
        return best

    # ---------- MISTAKE HIGHLIGHTING (delegates to AlignmentUI) ----------
    def highlight_mistake(self, mistake):
        """Pan to and highlight the note(s) involved in a mistake."""
        self._sync_alignment()
        t = self.alignment_ui.highlight_mistake(mistake)
        if t is not None:
            self.move_plot(t)

    def update_highlight_override(self, overridden: bool):
        self.alignment_ui.override_mistake(overridden)

    def clear_highlight(self):
        self.alignment_ui.clear_highlight()

    # ---------- PITCH-DOT VOLUME COLORING ----------
    def set_live(self, live: bool):
        """Toggle live-recording mode for the volume coloring (see
        PitchDataUI.set_live). Called by the tab on record start/stop."""
        self.user_pitches.set_live(live)

    # ---------- LEGEND + COLOR-MODE DROPDOWN ----------
    def init_legend(self):
        """Bottom row under the plot: a color legend (left) + a right-aligned
        "Colors:" dropdown switching pitch/volume dot coloring."""
        row = QHBoxLayout()
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(14)

        self._legend_items = QHBoxLayout()
        self._legend_items.setContentsMargins(0, 0, 0, 0)
        self._legend_items.setSpacing(14)
        row.addLayout(self._legend_items)
        row.addStretch(1)

        # label + dropdown share a tighter gap than the legend's item spacing
        picker = QHBoxLayout()
        picker.setSpacing(6)
        picker.addWidget(QLabel("Colors:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Pitch", "Volume"])
        self.color_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.color_combo.setStyleSheet(self._COMBO_STYLE)
        self.color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        picker.addWidget(self.color_combo)
        row.addLayout(picker)

        self._layout.addLayout(row)
        self._rebuild_legend()

    def _rebuild_legend(self):
        """Repopulate the legend for the current color mode: pitch gets the
        correct->way-off plasma strip plus a transition-grey swatch; volume gets
        the quiet->loud strip (transitions are only grey in pitch mode — volume
        mode colors every frame by its volume)."""
        while self._legend_items.count():
            w = self._legend_items.takeAt(0).widget()
            if w is not None:
                w.hide()  # deleteLater is deferred; hide now so no stale flash
                w.deleteLater()

        if self.color_mode == "volume":
            self._legend_items.addWidget(Legend.gradient_strip(VolumeGradient()))
            return
        self._legend_items.addWidget(Legend.gradient_strip(PitchGradient()))
        self._legend_items.addWidget(
            Legend.swatch(Colors.TRANSITION_RGB, "transition"))

    def _on_color_mode_changed(self, text: str):
        self.color_mode = "volume" if text.lower() == "volume" else "pitch"
        self._rebuild_legend()
        self.update_view_items()

    # ---------- HOVER CURSOR ----------
    def _on_plot_hover(self, scene_pos):
        """Show a pointing-hand cursor while hovering a detected user note (they're
        left-clickable for the characteristics popup)."""
        over = False
        if self.recording is not None and self.recording.note_data.times:
            view_pt = self.plot.getViewBox().mapSceneToView(scene_pos)
            over = self._user_note_at(view_pt.x(), view_pt.y()) is not None
        if over != self._hover_cursor_on:
            self.plot.viewport().setCursor(
                Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.ArrowCursor)
            self._hover_cursor_on = over



class RunGuitarHero:
    def __init__(self, recording: Recording=None, app=None):
        import sys
        from PyQt6.QtWidgets import QApplication, QMainWindow

        if app is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = app

        self.main_window = QMainWindow()
        self.app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        self.main_window.setWindowTitle("Attune [Guitar Hero]")
        self.main_window.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.main_window.setCentralWidget(self.central_widget)

        # initialize the visualizer widget adding it to the layout
        self.vis = GuitarHero()
        self.central_layout.addWidget(self.vis)
        self.init_toolbar()

        if recording is not None:
            self.vis.load_user(recording)

        self.main_window.show()
        self.app.exec()

    def init_toolbar(self):
        from PyQt6.QtWidgets import QToolBar
        from PyQt6.QtCore import Qt

        self.toolbar = QToolBar()
        self.toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.toolbar.addAction("Exit", self.close)
        self.main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

    def close(self):
        self.app.quit()
