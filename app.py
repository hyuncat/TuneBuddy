import os
# force QtWebEngine's chromium GPU process to use software rasterization
# eg, run on the CPU. set before importing
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
from ui.time.CountdownTimer import CountdownTimer

from ui.info.Toolbar import Toolbar
from ui.info.StatusBar import StatusBar
from ui.info.RecordingTree import RecordingTree
from ui.info.InstrumentWidget import InstrumentWidget
from ui.info.MistakeWidget import MistakeWidget
from ui.info.ToleranceWidget import ToleranceWidget
from ui.info.Settings import SettingsDialog

# app logic imports
from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth

# the two center "mode" tabs, each in its own file
from perform import PerformTab
from practice import PracticeTab


class Attune(QMainWindow):
    """Each Attune instance is associated with a single score and multiple
    recordings associated to that score, each with its own analysis and settings. 
    This top level class controls shared ScoreData, manages signals + timekeeping
    + settings. 
    
    Owns two center tabs:
        - PerformTab: time-invariant performance mode, contains mistake analysis
        - PracticeTab: live pitch-matching, blocks until you play in tune
    """
    def __init__(self):
        super().__init__()
        self.score_data = ScoreData()
        self.recordings: dict[str, Recording] = {}  # name -> Recording
        self.active_recording: Recording | None = None

        self.DEMO_SCORE_PATH = str(Path(__file__).resolve().parent / "resources" / "scores" / "c_major_scale.mxl")

        # shared transport engines. The synth + wall clock are shared by both
        # tabs; each tab builds its OWN MidiPlayer on them (in attach_timekeeping)
        # so it plays its own independent score.
        self.wall_clock = WallClock(hz=10)
        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        # the metronome count-in is shared by both tabs
        self.is_counting_in = False
        # full-score vs active-part view toggle, shared across both tabs: app.py
        # owns it (single source of truth) and pushes it into each tab via
        # set_show_full, like the other side-panel settings.
        self.viewer_show_full = False

        # IMPORTANT COMPONENTS
        # left column
        self.recordings_tree = RecordingTree(self.recordings)
        self.instrument_widget = InstrumentWidget()
        # center tabs
        self.perform_tab = PerformTab(self.score_data)
        self.practice_tab = PracticeTab()  # owns its own independent score
        # right column
        self.mistake_widget = MistakeWidget()
        self.tolerance_widget = ToleranceWidget()

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

        # --- splitter: RECORDINGS TREE | [PERFORM / PRACTICE] | MISTAKES ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # ------> LEFT COLUMN
        # recordings tree <-splitter-> instrument/range panel
        self.left_column = QSplitter(Qt.Orientation.Vertical)
        self.left_column.addWidget(self.recordings_tree)
        self.left_column.addWidget(self.instrument_widget)
        self.left_column.setStretchFactor(0, 1)  # tree takes the slack
        self.left_column.setStretchFactor(1, 0)  # instrument panel fixed-ish
        self.left_column.setMinimumWidth(180)
        self.left_column.setMaximumWidth(320)

        # ------> CENTER COLUMN
        # perform tab - record and analyze mistakes
        # practice tab - blocking mode for off-pitches
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self.perform_tab, "Perform")
        self.center_tabs.addTab(self.practice_tab, "Practice")

        # ------> RIGHT COLUMN
        # mistake widget - list mistakes 
        # tolerance slider
        self.right_column = QWidget()
        self.right_column_layout = QVBoxLayout(self.right_column)
        self.right_column_layout.setContentsMargins(0, 0, 0, 0)
        self.right_column_layout.addWidget(self.mistake_widget)
        self.right_column_layout.addWidget(self.tolerance_widget)
        self.right_column_layout.setStretch(0, 1)  # yes MistakeWidget stretch!
        self.right_column_layout.setStretch(1, 0)  # no ToleranceWidget stretch
        self.tolerance_widget.setFixedHeight(self.tolerance_widget.sizeHint().height())
        # keep the column as narrow as the fixed-width MistakeWidget
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
        self.perform_tab.attach_timekeeping(
            wall_clock=self.wall_clock,
            slider=self.slider,
            status_bar=self.status_bar,
            midi_synth=self.midi_synth,
            mistake_widget=self.mistake_widget,
        )
        self.practice_tab.attach_timekeeping(
            wall_clock=self.wall_clock,
            slider=self.slider,
            status_bar=self.status_bar,
            midi_synth=self.midi_synth,
        )
        # --- DIALOGS ---
        self.settings_dialog = SettingsDialog()
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
        self.analyze_button.clicked.connect(self.perform_tab.analyze)
        self.slider_layout.addWidget(self.analyze_button)

        self._layout.addWidget(self.transport_widget)

    def init_signals(self):
        """Connect all signals and slots for UI / app logic."""
        # --- TOOLBAR SIGNALS ---
        # score/audio uploaded
        self.toolbar.score_uploaded.connect(self.load_score)
        self.toolbar.audio_uploaded.connect(self.load_audio)
        # show settings
        self.toolbar.show_settings.connect(self.settings_dialog.show)
        # clip stuff
        self.toolbar.clip_requested.connect(self.on_clip_requested)
        self.toolbar.clip_reset.connect(self.on_clip_reset)
        # the clip is GLOBAL: a clip set in one tab mirrors onto the other (note
        # indices are tab-independent, so the same clip means the same measures).
        self.perform_tab.clip_changed.connect(self.practice_tab.sync_clip)
        self.practice_tab.clip_changed.connect(self.perform_tab.sync_clip)
        # user audio playback
        self.toolbar.user_audio_toggled.connect(self.perform_tab.set_user_audio_enabled)
        # bpm / tempo handling
        self.toolbar.tempo_changed.connect(self.on_tempo_changed)

        # --- CENTRAL TABS ---
        self.center_tabs.currentChanged.connect(self.on_center_tab_changed)
        self.perform_tab.viewer_ready.connect( # triggers demo load
            lambda: self.load_score(self.DEMO_SCORE_PATH)
        )
        self.perform_tab.analyzed.connect( # reflect new bpm / length
            lambda: self.toolbar.set_tempo(self.score_data.bpm)
        ) 

        # --- TIMEKEEPING ---
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._on_countdown_finished)

        # --- SIDE PANELS ---
        # recordings tree
        self.recordings_tree.selected.connect(self.on_recording_selected)
        self.recordings_tree.score_renamed.connect(self.on_score_renamed)
        # instrument panel
        self.instrument_widget.instrument_applied.connect(self.on_instrument_applied)
        self.instrument_widget.range_applied.connect(self.on_range_applied)
        self.instrument_widget.tuning_applied.connect(self.on_tuning_applied)
        self.instrument_widget.full_score_toggled.connect(self.on_full_score_toggled)
        # tolerance panel
        self.tolerance_widget.tolerance_applied.connect(self.on_tolerance_applied)

    # --- ACTIVE-TAB HELPERS ---
    def _practice_active(self) -> bool:
        """True when the Practice tab is the one currently shown."""
        return self.center_tabs.currentWidget() is self.practice_tab

    def _active_tab(self):
        """The mode panel that the shared transport should drive right now."""
        return self.practice_tab if self._practice_active() else self.perform_tab

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
        """Load the score into the app.
        Also push into toolbar's playback menu, BPM display, instrument panel, update slider range,
        and initialize recordings tree wrt. that score. Load into both performance and practice panels.
        """
        self.unalive()  # stop any running playback/recording and reset the clock

        filepath = Path(filepath)
        self.score_data.load(filepath)

        # load into UTILITIES
        self.toolbar.populate_instrument_menu()
        self.toolbar.set_tempo(self.score_data.bpm)  # reflect the score's tempo
        self.instrument_widget.load_score(self.score_data)
        self.slider.update_range(score_data=self.score_data)

        # load into RECORDINGS TREE
        self.recordings_tree.init_score(filepath=filepath, score_data=self.score_data)
        # adding (and selecting) a recording fires on_recording_selected, which
        # sets active_recording and hands it to the Perform tab.
        self.recordings_tree._add_recording(name="untitled_recording")

        # cleanup panels and load into both tabs
        self.perform_tab.cleanup()
        self.perform_tab.load_score(self.score_data)
        self.practice_tab.cleanup()
        self.practice_tab.load_score(filepath)

    def load_audio(self, filepath: str):
        if not self._has_recording():
            return
        rec = self.active_recording
        rec.load_audio(filepath)  # loads the raw waveform only
        # default the recording's name to the uploaded audio file's name
        self.recordings_tree.set_recording_name(Path(filepath).stem)
        self.sync_slider()
        # make it playable + (re-)detect pitches in the background (the Perform
        # tab owns the audio player + detection pipeline)
        self.perform_tab.refresh_audio()

    # --- SHARED TRANSPORT DISPATCH ---
    def toggle_playback(self):
        """Play/pause the active tab via the shared play button."""
        playing = self._active_tab().toggle_playback()
        self.play_button.setIcon(self.pause_icon if playing else self.play_icon)

    def toggle_recording(self):
        panel = self._active_tab()
        # Perform needs an active recording to capture into; Practice always has
        # its own, so only guard on the Perform tab.
        if panel is self.perform_tab and not self._has_recording(warn=True):
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
        self._active_tab().start_recording()

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
        self._active_tab().on_clock_tick(t)

    def slider_changed(self, t: float):
        """Shared slider moved: refresh the time label, then drive the active
        tab's plots (each panel guards against moving while it's playing)."""
        self.update_time_label(t)
        self._active_tab().on_slider_changed(t)

    def slider_end(self, t: float):
        """Shared slider hit its end (or the clip end). On the Practice tab, stop
        the take and reset the transport. On the Perform tab, stop playback only
        when a clip is active — its end is the natural stop; an unclipped Perform
        tab still plays to the true end as before."""
        if self._practice_active():
            self.practice_tab.on_slider_end()
            self.play_button.setIcon(self.play_icon)
            self.record_button.setIcon(self.record_icon)
            self.status_bar.update_status("")

        elif self.slider.clip_window is not None and self.perform_tab.is_playing:
            self.perform_tab.stop_playback()
            self.play_button.setIcon(self.play_icon)

    # --- SIDE-PANEL HANDLERS ---
    # ------> RECORDINGS TREE
    def on_recording_selected(self, recording_name: str):
        """
        Triggers when recording is selected in RecordingTree.
        Updates it as the active_recording, pushes its config into side panels,
        and updates Performance Tab with it.
        """
        if recording_name not in self.recordings.keys():
            print(f"No recording named '{recording_name}' was found.")
            return
        
        # set the active recording here!
        self.active_recording = self.recordings[recording_name]
        rec = self.active_recording

        # push config into side panels
        self.instrument_widget.set_active_instrument(rec.active_instrument)
        self.instrument_widget.set_tuning(rec.config.tuning)
        self.tolerance_widget.set_tolerance(rec.config.tolerance)
        self.status_bar.update_name(recording_name)
        # also pass to Performance Tab
        self.perform_tab.set_active_recording(rec)
        self.sync_slider()
    
    def on_score_renamed(self, title: str):
        """The Score Title was edited in the RecordingTree (the source of truth).
        Push it to BOTH tabs' (independent) scores and re-render their viewers."""
        self.score_data.set_title(title)
        self.practice_tab.score_data.set_title(title)
        self.perform_tab.refresh_score_viewer()
        self.practice_tab.refresh_score_viewer()

    # ------> SELECT INSTRUMENT PANEL
    def on_instrument_applied(self, channel: int):
        """Make the selected `channel` the active instrument for the 
        active recording. Keep both tabs (sharing score_data) in sync."""
        if not self._has_recording():
            return
        # the range defaults follow the newly selected instrument's note range
        self.instrument_widget.populate_range_from_score(self.score_data, channel)
        # update the panels
        self.perform_tab.set_active_instrument(channel)
        self.practice_tab.set_active_instrument(channel)

    def on_full_score_toggled(self, show_full: bool):
        """Toggle BOTH tabs' score viewers between the full score and the active
        part. app.py owns the flag (single source of truth) and pushes it into
        each tab via set_show_full, so the two stay in sync like the other
        side-panel settings."""
        self.viewer_show_full = show_full
        self.perform_tab.set_show_full(show_full)
        self.practice_tab.set_show_full(show_full)

    def on_range_applied(self, low_midi: int, high_midi: int):
        """Set the active recording's Config frequency range, then re-detect."""
        if not self._has_recording():
            return
        rec = self.active_recording
        config = rec.config
        fmin = config.midi_to_freq(low_midi)
        fmax = config.midi_to_freq(high_midi)
        config.fmin = fmin
        config.fmax = fmax
        rec.update_config(config)
        # mirror into the Practice tab's recording config
        self.practice_tab.set_freq_range(fmin, fmax)
        # re-run pitch detection on the existing audio (if any) with the new range
        self.perform_tab.detect_pitches()

    def on_tuning_applied(self, tuning: float):
        """Set the active recording's Config tuning (A4 Hz), then re-detect."""
        if not self._has_recording():
            return
        # update the active recording's appropriate config
        rec = self.active_recording
        rec.config.tuning = tuning
        rec.update_config(rec.config)
        # also update in practice panel
        self.practice_tab.set_tuning(tuning)
        self.perform_tab.detect_pitches()

    # ------> TOLERANCE PANEL
    def on_tolerance_applied(self, tolerance: float):
        """Set the active recording's Config tolerance, mirror it into Practice,
        then re-run just the mistake step (if the take is already analyzed)."""
        if not self._has_recording():
            return
        rec = self.active_recording
        rec.config.tolerance = tolerance
        rec.update_config(rec.config)
        # mirror into the Practice tab (drives its live pitch match)
        self.practice_tab.set_tolerance(tolerance)
        self.perform_tab.reanalyze_if_analyzed()

    # --- TOOLBAR THINGS ---
    def _is_live(self) -> bool:
        """True if either tab is playing/recording, or a count-in is running."""
        return (self.perform_tab.is_playing or self.perform_tab.is_recording
                or self.practice_tab.is_playing or self.practice_tab.is_recording
                or self.is_counting_in)
    
    def on_tempo_changed(self, new_bpm: int):
        """Triggers when tempo changes, pushes changes to the active tab's score.
        Rejects when recording/playing back."""
        sd = self._active_score_data()
        if self._is_live():
            self.toolbar.set_tempo(sd.bpm)  # revert UI
            return
        sd.change_tempo(new_bpm)
        self._active_tab().guitar_hero.update_view_items()
        self.sync_slider() # re-range the shared slider for the active tab

    def _active_score_data(self) -> ScoreData:
        """The score the active tab is showing — Perform's is the app's; Practice
        keeps its own independent copy."""
        return self.practice_tab.score_data if self._practice_active() else self.score_data

    def _on_analyzed(self):
        """Analyze resized the Perform score: show its new BPM in the toolbar (it
        only runs on the Perform tab, whose score is self.score_data)."""
        self.toolbar.set_tempo(self.score_data.bpm)

    # --- TAB SWITCHING / SLIDER SYNC ---
    def _active_slider_recording(self):
        """The recording whose length should size the shared slider, picked by the
        active tab (Practice keeps its own take separate from Perform's)."""
        if self._practice_active():
            return self.practice_tab.recording
        return self.active_recording

    def sync_slider(self):
        """Re-range the shared slider to the active tab's score/recording length."""
        self.slider.update_range(
            score_data=self._active_score_data(), recording=self._active_slider_recording()
        )

    def unalive(self):
        """Go 'un-live': stop playback/recording/count-in on both tabs"""
        self.perform_tab.stop_playback()
        self.perform_tab.stop_recording()
        self.practice_tab.stop_playback()
        self.practice_tab.stop_recording()
        if self.is_counting_in:
            self.countdown_timer.cancel()
            self.is_counting_in = False

    def on_center_tab_changed(self, index: int):
        """The shared transport drives whichever tab is active. On switch: halt the
        old tab, hide the Perform-only Analyze button on Practice, reset the
        transport buttons, then re-range the shared slider to the new tab and line
        its views up with the current time (the two tabs can have different total
        lengths, so the slider position is clamped/re-rendered to stay consistent)."""
        practice = self.center_tabs.widget(index) is self.practice_tab
        self.unalive()

        self.analyze_button.setVisible(not practice)
        # reset the shared transport to a stopped state for the new tab
        self.play_button.setIcon(self.play_icon)
        self.record_button.setIcon(self.record_icon)
        self.status_bar.update_status("")
        # the tempo display reflects the active tab's (independent) score tempo
        self.toolbar.set_tempo(self._active_score_data().bpm)

        # preserve the current time, re-range for the new tab, clamp, and render.
        t = self.slider.get_time()
        self.sync_slider()
        t = min(t, self.slider.get_total_time())
        self.slider.handle_timer_update(t)   # move the shared handle to clamped t
        self._active_tab().render_at(t)

    # --- CLIP (measure-range focus) ---
    def on_clip_requested(self):
        """Clip menu 'Clip': clip the active tab to the measures it has selected."""
        self._active_tab().apply_clip()

    def on_clip_reset(self):
        """Clip menu 'Reset': restore the active tab's full score."""
        self._active_tab().reset_clip()


if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = Attune()
    window.show()
    sys.exit(app.exec())
