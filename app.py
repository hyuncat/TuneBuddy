import os
# Force QtWebEngine's Chromium GPU process to use software rasterization. On
# macOS, the embedded WebEngine, pyqtgraph, and Qt's native compositor all
# share the same Metal memory pool, and a busy splitter (e.g. lots of rows
# with QTreeWidget.setItemWidget(QPushButton)) can starve Chromium's GPU
# process — it reports "Insufficient Memory" / "GL_OUT_OF_MEMORY", loses its
# context, and the next paint segfaults. Software rasterization sidesteps
# that contention entirely. Must be set BEFORE QApplication is constructed.
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-gpu-compositing --disable-software-rasterizer=false",
)

from pathlib import Path
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
import qdarktheme

from ui.time.Slider import Slider
from ui.time.WallClock import WallClock
from ui.time.Clipper import ClipperDialog
from ui.time.CountdownTimer import CountdownTimer

from ui.info.Toolbar import Toolbar
from ui.info.StatusBar import StatusBar
from ui.info.RecordingTree import RecordingTree
from ui.info.InstrumentPanel import InstrumentPanel
from ui.info.MistakeWidget import MistakeWidget
from ui.info.TolerancePanel import TolerancePanel
from ui.info.Settings import SettingsDialog

# app logic imports
from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth

# the two center "mode" tabs, each in its own file
from perform import PerformPanel
from practice import PracticePanel


class Attune(QMainWindow):
    """Each Attune instance is associated with a single score (midi/musicxml) and
    allows you to create multiple recordings associated to that score, each with
    its own analysis and settings.

    The window is glue: it owns the score, the recordings model, the side panels
    (recording tree / instrument / tolerance / mistakes), and the bottom transport
    (play / record / slider / status bar). The center is a tabbed area with two
    self-contained "mode" panels — **Perform** (`perform.py`) and **Practice**
    (`practice.py`) — that SHARE this transport. The host routes the transport's
    button clicks and the shared clock/slider ticks to whichever tab is active,
    and pushes side-panel changes into both panels via their public setters.
    """
    def __init__(self):
        super().__init__()
        self.score_data = ScoreData()
        self.recordings: dict[str, Recording] = {}  # name -> Recording
        self.active_recording: Recording | None = None

        # shared transport engines. The synth + wall clock are shared by both
        # tabs; each tab builds its OWN MidiPlayer on them (in attach_transport)
        # so it plays its own independent score.
        self.wall_clock = WallClock(hz=10)
        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        # the metronome count-in is shared by both tabs
        self.is_counting_in = False

        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """
        Initialize all UI components for the window.
            - main window (title + geom)
            - splitter
                - recordings file tree + instrument panel (left)
                - center tabs: Perform | Practice
                - mistake widget + tolerance panel (right)
            - transport row (play / record / time / slider / analyze)
            - status bar (+ countdown timer)
            - toolbar
            - dialogs (settings, clipper)
        """
        self.setWindowTitle("Attune")
        self.setGeometry(100, 100, 1300, 800)

        # --- CENTRAL LAYOUT WIDGET ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = QVBoxLayout(self.central_widget)

        # --- (splitter) RECORDINGS TREE | [PERFORM / PRACTICE] | MISTAKES ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.recordings_tree = RecordingTree(self.recordings)
        self.instrument_panel = InstrumentPanel()

        # left column: recordings tree (top) and the instrument/range panel below
        # it, in a vertical splitter so each section's height is user-adjustable.
        self.left_column = QSplitter(Qt.Orientation.Vertical)
        self.left_column.addWidget(self.recordings_tree)
        self.left_column.addWidget(self.instrument_panel)
        self.left_column.setStretchFactor(0, 1)  # tree takes the slack
        self.left_column.setStretchFactor(1, 0)  # instrument panel fixed-ish
        self.left_column.setMinimumWidth(180)
        self.left_column.setMaximumWidth(320)

        # right column widgets
        self.mistake_widget = MistakeWidget()
        self.tolerance_panel = TolerancePanel()

        # --- CENTER: tabbed "mode" area ---
        # Perform = record-and-analyze view; Practice = real-time note feedback.
        # Each panel lives in its own file with its own views + Recording, but
        # they SHARE this window's transport (attached below, once it exists).
        self.perform_panel = PerformPanel(self.score_data)
        self.practice_panel = PracticePanel()  # owns its own independent score
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self.perform_panel, "Perform")
        self.center_tabs.addTab(self.practice_panel, "Practice")

        # right column: the mistake list (top) and a tolerance tuner (bottom),
        # in a vertical splitter so the tuner's height is user-adjustable.
        self.right_column = QSplitter(Qt.Orientation.Vertical)
        self.right_column.addWidget(self.mistake_widget)
        self.right_column.addWidget(self.tolerance_panel)
        self.right_column.setStretchFactor(0, 1)  # mistake list takes the slack
        self.right_column.setStretchFactor(1, 0)  # tolerance tuner stays compact
        _tp_h = self.tolerance_panel.sizeHint().height()
        self.right_column.setSizes([700, _tp_h])   # initial heights (resizable)
        # keep the column as narrow as the (fixed-width) MistakeWidget
        self.right_column.setMinimumWidth(self.mistake_widget.minimumWidth())
        self.right_column.setMaximumWidth(self.mistake_widget.maximumWidth())

        # add the widgets
        self.splitter.addWidget(self.left_column)
        self.splitter.addWidget(self.center_tabs)
        self.splitter.addWidget(self.right_column)
        # set behavior controls
        self.splitter.setStretchFactor(0, 0)  # recordings tree is fixed-ish
        self.splitter.setStretchFactor(1, 1)  # center column grows
        self.splitter.setStretchFactor(2, 0)  # mistake widget is fixed-ish
        # default split: side panels start compact (still user-resizable).
        self.splitter.setSizes([240, 764, 296])

        self._layout.addWidget(self.splitter)

        # --- INIT TRANSPORT ROW ---
        self.init_slider_layout()

        # --- UTILITIES ---
        self.status_bar = StatusBar(name="untitled_recording")
        self.setStatusBar(self.status_bar)
        self.countdown_timer = CountdownTimer(self.status_bar, midi_synth=self.midi_synth)
        self.toolbar = Toolbar(score_data=self.score_data)
        self.addToolBar(self.toolbar)

        # hand both tabs the shared transport now that it all exists. The Perform
        # tab also drives the (Perform-only) MistakeWidget.
        self.perform_panel.attach_transport(
            wall_clock=self.wall_clock,
            slider=self.slider,
            status_bar=self.status_bar,
            midi_synth=self.midi_synth,
            mistake_widget=self.mistake_widget,
        )
        self.practice_panel.attach_transport(
            wall_clock=self.wall_clock,
            slider=self.slider,
            status_bar=self.status_bar,
            midi_synth=self.midi_synth,
        )

        # --- DIALOGS ---
        self.settings_dialog = SettingsDialog()
        self.clipper_dialog = ClipperDialog()

        self.show()  # run the show :)

    def init_slider_layout(self):
        """Initialize the shared transport row: play/pause, record, time label,
        slider, and the (Perform-only) Analyze button."""
        self.transport_widget = QWidget()
        self.slider_layout = QHBoxLayout(self.transport_widget)
        self.slider_layout.setContentsMargins(0, 0, 0, 0)

        # get the play/pause button icons
        app_directory = os.path.dirname(__file__)
        play_filepath = os.path.join(app_directory, 'resources', 'icons', 'play.png')
        pause_filepath = os.path.join(app_directory, 'resources', 'icons', 'pause.png')
        record_filepath = os.path.join(app_directory, 'resources', 'icons', 'record.png')

        self.play_icon = QIcon(play_filepath)
        self.pause_icon = QIcon(pause_filepath)
        self.record_icon = QIcon(record_filepath)

        # play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(QSize(26, 26))
        self.play_button.clicked.connect(self.toggle_playback)
        self.slider_layout.addWidget(self.play_button)

        # record button
        self.record_button = QPushButton()
        self.record_button.setIcon(self.record_icon)
        self.record_button.setFixedSize(QSize(26, 26))
        self.record_button.clicked.connect(self.toggle_recording)
        self.slider_layout.addWidget(self.record_button)

        # time label (current/total)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setMinimumWidth(100)
        self.slider_layout.addWidget(self.time_label)

        # the slider
        self.slider = Slider(self.wall_clock)
        self.slider_layout.addWidget(self.slider)

        # analyze button (Perform-only; hidden on the Practice tab)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_panel.analyze)
        self.slider_layout.addWidget(self.analyze_button)

        self._layout.addWidget(self.transport_widget)

    def init_signals(self):
        """Connect all signals and slots for UI / app logic."""
        # toolbar signals
        self.toolbar.score_uploaded.connect(self.load_score)
        self.toolbar.audio_uploaded.connect(self.load_audio)
        self.toolbar.show_settings.connect(self.settings_dialog.show)
        self.toolbar.show_clipper.connect(self.clipper_dialog.show)
        self.toolbar.user_audio_toggled.connect(self.on_user_audio_toggled)
        self.toolbar.practice_toggled.connect(self.on_practice_toggled)
        self.toolbar.tempo_changed.connect(self.on_tempo_changed)
        self.center_tabs.currentChanged.connect(self.on_center_tab_changed)

        # shared timekeeping signals
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._on_countdown_finished)

        # side panels
        self.recordings_tree.selected.connect(self.on_recording_selected)
        self.recordings_tree.score_renamed.connect(self.on_score_renamed)
        self.instrument_panel.instrument_applied.connect(self.on_instrument_applied)
        self.instrument_panel.range_applied.connect(self.on_range_applied)
        self.instrument_panel.tuning_applied.connect(self.on_tuning_applied)
        self.instrument_panel.full_score_toggled.connect(self.on_full_score_toggled)
        self.tolerance_panel.tolerance_applied.connect(self.on_tolerance_applied)

        # the Perform tab's score viewer is what triggers the initial demo load
        self.perform_panel.viewer_ready.connect(self.on_score_viewer_loaded)
        # after Analyze resizes the Perform score, reflect its new BPM/length
        self.perform_panel.analyzed.connect(self._on_analyzed)

    # --- ACTIVE-TAB HELPERS ---
    def _practice_active(self) -> bool:
        """True when the Practice tab is the one currently shown."""
        return self.center_tabs.currentWidget() is self.practice_panel

    def _active_panel(self):
        """The mode panel that the shared transport should drive right now."""
        return self.practice_panel if self._practice_active() else self.perform_panel

    def _has_recording(self, warn=False) -> bool:
        """True if there's an active recording, else optionally warn. Used by the
        side-panel handlers (instrument / range / tuning / tolerance)."""
        if self.active_recording is None:
            if warn:
                QMessageBox.warning(self, "No recording selected", "Please select a recording first.")
            return False
        return True

    # --- LOAD SCORE / AUDIO ---
    def load_score(self, filepath: str):
        """Load the score into the app and push it into every view/engine."""
        filepath = Path(filepath)
        self.score_data.load(filepath)

        # default to the first real (non-metronome) instrument channel
        first_ch = next(
            (ch for ch in self.score_data.instruments
             if ch != self.score_data.metronome_channel),
            0,
        )
        self.score_data.active_instrument = first_ch

        # load into side UI
        self.toolbar.populate_instrument_menu()
        self.toolbar.set_tempo(self.score_data.bpm)  # reflect the score's tempo
        self.instrument_panel.load_score(self.score_data)
        self.slider.update_range(score_data=self.score_data)
        self.recordings_tree.init_score(filepath=filepath, score_data=self.score_data)
        # adding (and selecting) a recording fires on_recording_selected, which
        # sets active_recording and hands it to the Perform tab.
        self.recordings_tree._add_recording(name="untitled_recording")

        # fresh score => wipe any analysis/artifacts left over from the previous one
        self.perform_panel.cleanup()
        self.perform_panel.load_score(self.score_data)
        # the Practice tab parses its OWN independent copy from the same file, so
        # Perform's later resize/tempo edits never mutate the score shown there.
        self.practice_panel.load_score(filepath)

    def load_audio(self, filepath: str):
        if not self._has_recording():
            return
        rec = self.active_recording
        rec.load_audio(filepath)  # loads the raw waveform only
        # default the recording's name to the uploaded audio file's name
        self.recordings_tree.set_recording_name(Path(filepath).stem)
        self._sync_slider_range()
        # make it playable + (re-)detect pitches in the background (the Perform
        # tab owns the audio player + detection pipeline)
        self.perform_panel.refresh_audio()

    # --- SHARED TRANSPORT DISPATCH ---
    def toggle_playback(self):
        """Play/pause the active tab via the shared play button."""
        playing = self._active_panel().toggle_playback()
        self.play_button.setIcon(self.pause_icon if playing else self.play_icon)

    def toggle_recording(self):
        panel = self._active_panel()
        # Perform needs an active recording to capture into; Practice always has
        # its own, so only guard on the Perform tab.
        if panel is self.perform_panel and not self._has_recording(warn=True):
            return
        if self.is_counting_in:
            # clicking again during the count-in cancels it (un-arm recording)
            self.countdown_timer.cancel()
            self.is_counting_in = False
            self.record_button.setIcon(self.record_icon)
            return
        if not panel.is_recording:
            # play a one-measure metronome count-in; _on_countdown_finished starts
            # the active tab's recording on finish. Use the active tab's score so
            # the count-in beats land at that tab's (possibly different) tempo.
            sd = self._active_score_data()
            self.is_counting_in = True
            self.record_button.setIcon(self.pause_icon)
            self.countdown_timer.start(
                beats=sd.count_in_beats(),
                channel=sd.metronome_channel,
            )
        else:
            panel.stop_recording()
            self.record_button.setIcon(self.record_icon)
            self.status_bar.update_status("")

    def _on_countdown_finished(self):
        """Shared count-in finished: start recording in whichever tab is active."""
        self.is_counting_in = False
        self.record_button.setIcon(self.pause_icon)
        self._active_panel().start_recording()

    def update_time_label(self, t: float):
        """Update the shared time label (current/total) from time `t`."""
        def format_time(seconds: float) -> str:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins:02}:{secs:04.1f}"

        current_time_str = format_time(t)
        total_time_str = format_time(self.slider.get_total_time())
        self.time_label.setText(f"{current_time_str} / {total_time_str}")

    def time_changed(self, t: float):
        """Shared wall-clock tick: refresh the time label, then drive the active
        tab's plots (each panel guards on its own playing state)."""
        self.update_time_label(t)
        self._active_panel().on_clock_tick(t)

    def slider_changed(self, t: float):
        """Shared slider moved: refresh the time label, then drive the active
        tab's plots (each panel guards against moving while it's playing)."""
        self.update_time_label(t)
        self._active_panel().on_slider_changed(t)

    def slider_end(self, t: float):
        """Shared slider hit its end. On the Practice tab, stop the take and reset
        the shared transport buttons (Perform has no auto-stop-at-end behavior)."""
        if self._practice_active():
            self.practice_panel.on_slider_end()
            self.play_button.setIcon(self.play_icon)
            self.record_button.setIcon(self.record_icon)
            self.status_bar.update_status("")

    # --- SIDE-PANEL HANDLERS ---
    def on_recording_selected(self, recording_name: str):
        """A recording was selected in the tree: make it active and reflect it in
        the side panels + the Perform tab."""
        if recording_name not in self.recordings.keys():
            print(f"No recording named '{recording_name}' was found.")
            return
        self.active_recording = self.recordings[recording_name]
        rec = self.active_recording
        # reflect this recording's Config in the tunable inputs
        self.instrument_panel.set_active_instrument(rec.active_instrument)
        self.instrument_panel.set_tuning(rec.config.tuning)
        self.tolerance_panel.set_tolerance(rec.config.tolerance)
        self.status_bar.update_name(recording_name)
        # hand the take to the Perform tab (loads its views/audio/mistakes)
        self.perform_panel.set_active_recording(rec)
        self._sync_slider_range()

    def on_instrument_applied(self, channel: int):
        """Make `channel` the active instrument for the active recording, and keep
        both tabs (which share this score_data) in sync."""
        if not self._has_recording():
            return
        self.perform_panel.set_active_instrument(channel)
        # the range defaults follow the newly selected instrument's note range
        self.instrument_panel.populate_range_from_score(self.score_data, channel)
        self.practice_panel.set_active_instrument(channel)

    def on_full_score_toggled(self, show_full: bool):
        """Toggle the Perform score viewer between full score and active part."""
        self.perform_panel.on_full_score_toggled(show_full)

    def on_range_applied(self, low_midi: int, high_midi: int):
        """Set the active recording's Config frequency range, then re-detect."""
        if not self._has_recording():
            return
        rec = self.active_recording
        config = rec.config
        config.fmin = config.midi_to_freq(low_midi)
        config.fmax = config.midi_to_freq(high_midi)
        rec.update_config(config)
        # mirror into the Practice tab's recording config
        self.practice_panel.set_range(low_midi, high_midi)
        # re-run pitch detection on the existing audio (if any) with the new range
        self.perform_panel.detect_pitches()

    def on_tuning_applied(self, tuning: float):
        """Set the active recording's Config tuning (A4 Hz), then re-detect."""
        if not self._has_recording():
            return
        rec = self.active_recording
        rec.config.tuning = tuning
        rec.update_config(rec.config)
        # mirror into the Practice tab's recording config
        self.practice_panel.set_tuning(tuning)
        self.perform_panel.detect_pitches()

    def on_tolerance_applied(self, tolerance: float):
        """Set the active recording's Config tolerance, mirror it into Practice,
        then re-run just the mistake step (if the take is already analyzed)."""
        if not self._has_recording():
            return
        rec = self.active_recording
        rec.config.tolerance = tolerance
        rec.update_config(rec.config)
        # mirror into the Practice tab (drives its live pitch match)
        self.practice_panel.set_tolerance(tolerance)
        self.perform_panel.reanalyze_if_analyzed()

    def on_score_renamed(self, title: str):
        """The Score Title was edited in the RecordingTree (the source of truth).
        Push it to BOTH tabs' (independent) scores and re-render their viewers."""
        self.score_data.set_title(title)
        self.practice_panel.score_data.set_title(title)
        self.perform_panel.refresh_score_viewer()
        self.practice_panel.refresh_score_viewer()

    def on_score_viewer_loaded(self):
        """Perform tab's score viewer is ready: load the demo score once."""
        DEMO_SCORE_PATH = Path(__file__).resolve().parent / "resources" / "scores" / "c_major_scale.mxl"
        self.load_score(str(DEMO_SCORE_PATH))

    def on_user_audio_toggled(self, checked: bool):
        """The toolbar 'User' checkbox (Perform-tab playback of the user audio)."""
        self.perform_panel.set_user_audio_enabled(checked)

    def on_tempo_changed(self, new_bpm: int):
        """Change the tempo of the ACTIVE tab's score only (the two tabs keep
        independent scores, so changing one never affects the other). Ignored
        (reverted) while anything is playing/recording, to avoid mid-stream drift."""
        sd = self._active_score_data()
        if self._any_transport_active():
            self.toolbar.set_tempo(sd.bpm)  # revert UI
            return
        sd.change_tempo(new_bpm)
        # refresh just the active tab's views (Verovio keeps its original-tempo
        # render, remapped via the panel's _score_viewer_time).
        self._active_panel().on_tempo_changed()
        # re-range the shared slider for the active tab
        self._sync_slider_range()

    def _active_score_data(self) -> ScoreData:
        """The score the active tab is showing — Perform's is the app's; Practice
        keeps its own independent copy."""
        return self.practice_panel.score_data if self._practice_active() else self.score_data

    def _on_analyzed(self):
        """Analyze resized the Perform score: show its new BPM in the toolbar (it
        only runs on the Perform tab, whose score is self.score_data)."""
        self.toolbar.set_tempo(self.score_data.bpm)

    def on_practice_toggled(self):
        """Toolbar 'Practice' button: jump to the Practice tab."""
        self.center_tabs.setCurrentWidget(self.practice_panel)

    # --- TAB SWITCHING / SLIDER SYNC ---
    def _active_slider_recording(self):
        """The recording whose length should size the shared slider, picked by the
        active tab (Practice keeps its own take separate from Perform's)."""
        if self._practice_active():
            return self.practice_panel.recording
        return self.active_recording

    def _sync_slider_range(self):
        """Re-range the shared slider to the active tab's score/recording length."""
        self.slider.update_range(
            score_data=self._active_score_data(), recording=self._active_slider_recording()
        )

    def _any_transport_active(self) -> bool:
        """True if either tab is playing/recording, or a count-in is running."""
        return (self.perform_panel.is_playing or self.perform_panel.is_recording
                or self.practice_panel.is_playing or self.practice_panel.is_recording
                or self.is_counting_in)

    def _halt_transport(self):
        """Stop playback/recording/count-in on BOTH tabs, so only the tab being
        switched to drives the shared transport (and the audio device isn't
        grabbed twice)."""
        self.perform_panel.stop_playback()
        self.perform_panel.stop_recording()
        self.practice_panel.stop_playback()
        self.practice_panel.stop_recording()
        if self.is_counting_in:
            self.countdown_timer.cancel()
            self.is_counting_in = False

    def on_center_tab_changed(self, index: int):
        """The shared transport drives whichever tab is active. On switch: halt the
        old tab, hide the Perform-only Analyze button on Practice, reset the
        transport buttons, then re-range the shared slider to the new tab and line
        its views up with the current time (the two tabs can have different total
        lengths, so the slider position is clamped/re-rendered to stay consistent)."""
        practice = self.center_tabs.widget(index) is self.practice_panel
        self._halt_transport()

        # Analyze is a Perform-only action
        self.analyze_button.setVisible(not practice)
        # reset the shared transport to a stopped state for the new tab
        self.play_button.setIcon(self.play_icon)
        self.record_button.setIcon(self.record_icon)
        self.status_bar.update_status("")
        # the tempo display reflects the active tab's (independent) score tempo
        self.toolbar.set_tempo(self._active_score_data().bpm)

        # preserve the current time, re-range for the new tab, clamp, and render.
        t = self.slider.get_time()
        self._sync_slider_range()
        t = min(t, self.slider.get_total_time())
        self.slider.handle_timer_update(t)   # move the shared handle to clamped t
        self._active_panel().render_at(t)


if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = Attune()
    window.show()
    sys.exit(app.exec())
