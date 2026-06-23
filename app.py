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
    QStatusBar, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QSplitter,
    QInputDialog, QMenu, QMessageBox, QStackedLayout
)
from PyQt6.QtCore import Qt, QSize, QTimer, QPoint, pyqtSignal, QObject
from PyQt6.QtGui import QIcon
import qdarktheme

from ui.ScoreViewer import ScoreViewer
from ui.GuitarHero import GuitarHero
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
from app_logic.user.ds.PitchData import PitchConfig
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData

from algorithms.Config import Config
from practice import PracticeAttune

class Attune(QMainWindow):
    """each attune instance is associated with a single score (midi/musicxml)
    and allows you to create multiple recordings associated to that score
    each with its own analysis and settings"""
    def __init__(self):
        super().__init__()
        self.score_data = ScoreData()
        self.recordings: dict[str, Recording] = {}  # name -> Recording
        self.active_recording: Recording | None = None
        # pitch_detectors we've already wired signals for (one per recording),
        # so init_pitch_detector_signals never double-connects the same detector
        self._wired_detectors: set = set()
        # rk: each recording comes with their own algorithms

        # PLAYBACK stuff
        self.wall_clock = WallClock(hz=10)
        self.metronome = None # TODO later

        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        self.midi_player = MidiPlayer(self.midi_synth, self.wall_clock)
        self.audio_player = AudioPlayer(None)
        self.audio_recorder = AudioRecorder(self.active_recording)
        # --> playback state variables
        self.is_playing = False
        self.is_recording = False
        self.is_counting_in = False
        self.user_playback_enabled = True
        # set when Analyze is pressed while offline pitch detection is still
        # running; _on_detection_finished runs the deferred analyze once the
        # smoothed pitch track is ready (analyzing raw pitches gives garbage)
        self._pending_analyze = False

        # instrument control
        self.displayed_instruments: set[int] = set() # programs to display
        self.playing_instruments: set[int] = set() # channels to play
        # score viewer: render only the active instrument's part (default) or
        # the full score (toggled via the instrument panel).
        self.viewer_show_full = False

        # initialize other important stuff
        self.init_ui()
        self.init_signals()

    def init_ui(self):
        """
        Initialize all UI components for the window.
            - main window (title + geom)
            - splitter
                - recordings file tree
                - center column (vertical splitter)
                    - score viewer (top)
                    - guitar hero (bottom)
                - right column (vertical splitter)
                    - mistake widget (top)
                    - tolerance panel (bottom)
            - slider layout
                - play/pause button
                - record button
                - time label
                - slider
                - analyze button
            - status bar
                - countdown timer
            - toolbar
            - dialogs (settings, clipper)
        """
        self.setWindowTitle("Attune")
        self.setGeometry(100, 100, 1300, 800)

        # --- CENTRAL LAYOUT WIDGET ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = QVBoxLayout(self.central_widget)

        # --- (splitter stuff) RECORDINGS TREE | [SCORE VIEWER / GUITAR HERO] | MISTAKES ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal) # allows horizontal resizing
        self.recordings_tree = RecordingTree(self.recordings)
        self.instrument_panel = InstrumentPanel()
        ABSOLUTE_PROJECT_ROOT = Path(__file__).resolve().parent

        # left column: recordings tree (top) and the instrument/range panel
        # below it, in a vertical splitter so each section's height is
        # user-adjustable. (Playback controls live on the Toolbar.)
        self.left_column = QSplitter(Qt.Orientation.Vertical)
        self.left_column.addWidget(self.recordings_tree)
        self.left_column.addWidget(self.instrument_panel)
        self.left_column.setStretchFactor(0, 1)  # tree takes the slack
        self.left_column.setStretchFactor(1, 0)  # instrument panel fixed-ish
        self.left_column.setMinimumWidth(180)
        self.left_column.setMaximumWidth(320)
        
        # score viewer requires a loading screen
        self.score_viewer_container = QWidget()
        stack = QStackedLayout(self.score_viewer_container)
        self.score_viewer = ScoreViewer(project_root=ABSOLUTE_PROJECT_ROOT)
        loading = QLabel("Loading...")
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stack.addWidget(loading)
        stack.addWidget(self.score_viewer)
        stack.setCurrentWidget(loading) # show loading screen until viewer is ready
        self.score_viewer.load_finished.connect(lambda ok: stack.setCurrentIndex(1) if ok else 0)

        self.guitar_hero = GuitarHero(self.active_recording)
        self.mistake_widget = MistakeWidget()
        self.tolerance_panel = TolerancePanel()

        # center column: score viewer stacked ON TOP of the guitar hero, with a
        # vertical splitter between them so both are adjustable in height.
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.center_splitter.addWidget(self.score_viewer_container)
        self.center_splitter.addWidget(self.guitar_hero)
        self.center_splitter.setStretchFactor(0, 1)  # score viewer grows
        self.center_splitter.setStretchFactor(1, 1)  # guitar hero grows
        # start the score viewer compact so its single white page roughly fills
        # the box (still user-resizable via the handle below it).
        self.center_splitter.setSizes([180, 520])    # initial heights (resizable)

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
        self.splitter.addWidget(self.center_splitter)
        self.splitter.addWidget(self.right_column)
        # set behavior controls
        self.splitter.setStretchFactor(0, 0)  # recordings tree is fixed-ish
        self.splitter.setStretchFactor(1, 1)  # center column grows
        self.splitter.setStretchFactor(2, 0)  # mistake widget is fixed-ish
        # default split: the side panels start compact (still user-resizable) so
        # the center column fills the full width between them.
        self.splitter.setSizes([240, 764, 296])

        self._layout.addWidget(self.splitter)

        # --- INIT SLIDER LAYOUT ---
        self.init_slider_layout()

        # --- UTILITIES --- 
        self.status_bar = StatusBar(name="untitled_recording") # with default recording name
        self.setStatusBar(self.status_bar)
        self.countdown_timer = CountdownTimer(self.status_bar, midi_synth=self.midi_synth)
        self.toolbar = Toolbar(score_data=self.score_data)
        self.addToolBar(self.toolbar)
        
        # --- DIALOGS ---
        self.settings_dialog = SettingsDialog()
        self.clipper_dialog = ClipperDialog()
        self.practice_attune = PracticeAttune(self.score_data, self.midi_synth) # practice mode window, initialized but not shown yet

        self.show() # run the show :)
        
    def init_slider_layout(self):
        """
        Initialize the layout containing the play/pause, 
        record button and the slider.
        """
        self.slider_layout = QHBoxLayout()

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
        self._layout.addLayout(self.slider_layout)

        # analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)
        self.slider_layout.addWidget(self.analyze_button)

    def init_signals(self):
        """Connect all signals and slots for UI / app logic"""
        # toolbar signals
        self.toolbar.score_uploaded.connect(self.load_score)
        self.toolbar.audio_uploaded.connect(self.load_audio)
        self.toolbar.show_settings.connect(self.settings_dialog.show)
        self.toolbar.show_clipper.connect(self.clipper_dialog.show)
        self.toolbar.user_audio_toggled.connect(self.on_user_audio_toggled)
        self.toolbar.practice_toggled.connect(self.on_practice_toggled)
        self.toolbar.tempo_changed.connect(self.on_tempo_changed)

        # timekeeping signals
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._start_recording)

        self.recordings_tree.selected.connect(self.on_recording_selected)
        self.recordings_tree.score_renamed.connect(self.on_score_renamed)
        self.instrument_panel.instrument_applied.connect(self.on_instrument_applied)
        self.instrument_panel.range_applied.connect(self.on_range_applied)
        self.instrument_panel.tuning_applied.connect(self.on_tuning_applied)
        self.instrument_panel.full_score_toggled.connect(self.on_full_score_toggled)
        self.score_viewer.load_finished.connect(self.on_score_viewer_loaded)
        self.mistake_widget.selected.connect(self.on_mistake_selected)
        self.mistake_widget.override_toggled.connect(self.on_mistake_override_toggled)
        self.tolerance_panel.tolerance_applied.connect(self.on_tolerance_applied)
        self.guitar_hero.plot_moved.connect(self.slider.handle_timer_update)

        self.init_pitch_detector_signals()

        # settings dialog signals
    
    def init_pitch_detector_signals(self):
        """Wire the active recording's pitch detector signals. Each recording
        owns its own detector, so we connect each only once (tracked in
        `_wired_detectors`) to avoid duplicate slot invocations."""
        rec = self.active_recording
        if rec is None or rec.pitch_detector in self._wired_detectors:
            return
        rec.pitch_detector.status_changed.connect(self.status_bar.update_status)
        rec.pitch_detector.detection_finished.connect(self._on_detection_finished)
        self._wired_detectors.add(rec.pitch_detector)

    # --- SHARED HELPERS / CLEANUP ---
    def _require_recording(self, message: str = "Please select a recording first.") -> "Recording | None":
        """Return the active recording, or warn (with `message`) and return None
        when there isn't one. Centralizes the repeated no-recording guard."""
        if self.active_recording is None:
            QMessageBox.warning(self, "No recording selected", message)
            return None
        return self.active_recording

    def cleanup(self):
        """Reset all analysis-derived state so no stale artifacts survive an
        input change (new score/audio, re-run detection, re-analyze). Clears the
        active recording's notes/alignment/overrides, the mistake list, and the
        GuitarHero overlays (user notes, alignment, highlight)."""
        rec = self.active_recording
        if rec is not None:
            rec.note_data = NoteData()
            rec.alignment = Alignment(config=rec.config)
            rec.overridden_mistake_indices = set()
            # reloading the now-empty recording wipes the user-note + alignment
            # overlays and the highlight bar from the GuitarHero
            self.guitar_hero.load_user(rec)
        self.mistake_widget.clear()

    # --- LOAD SCORE / AUDIO ---
    def load_score(self, filepath: str):
        """Load the score into the app."""
        # reload same score_data so all modules which reference it 
        # get updated data without needing a manual refresh
        filepath = Path(filepath)
        self.score_data.load(filepath)

        # default to the first real (non-metronome) instrument channel
        first_ch = next(
            (ch for ch in self.score_data.instruments
             if ch != self.score_data.metronome_channel),
            0,
        )
        self.score_data.active_instrument = first_ch

        # load into important ui components
        self.toolbar.populate_instrument_menu()
        self.instrument_panel.load_score(self.score_data)
        self.slider.update_range(score_data=self.score_data)
        self.recordings_tree.init_score(filepath=filepath, score_data=self.score_data)
        self.recordings_tree._add_recording(name="untitled_recording") # dummy init
        # fresh score => wipe any analysis/artifacts left over from the previous one
        self.cleanup()
        self.guitar_hero.load_score(self.score_data)
        self.practice_attune.load_score(self.score_data)
        # render the score viewer (active instrument only, unless "full" is on)
        self.refresh_score_viewer()

        # load into playback engines
        self.midi_player.load_score(self.score_data)

    def load_audio(self, filepath: str):
        rec = self._require_recording("Please select a recording to load the audio into.")
        if rec is None:
            return
        rec.load_audio(filepath)  # loads the raw waveform only
        # default the recording's name to the uploaded audio file's name
        self.recordings_tree.set_recording_name(Path(filepath).stem)
        # UI that only needs the raw audio can refresh right away
        self.slider.update_range(score_data=self.score_data, recording=rec)
        self.audio_player.load_audio(rec.audio_data)
        # clear stale analysis, then detect pitches on the new audio in the
        # background (phase text appears in the status bar, cleared in
        # _on_detection_finished, which loads the fresh pitch data)
        self.detect_pitches()

    def detect_pitches(self):
        """Clear stale analysis, then (re-)run offline pitch detection on the
        active recording's audio in the background. Used whenever the pitch track
        must be recomputed (new audio, range/tuning change). When detection
        finishes, _on_detection_finished loads the fresh pitch data into view."""
        rec = self.active_recording
        if rec is None or rec.audio_data.end_index <= 0:
            return
        self.cleanup()
        self.init_pitch_detector_signals() # just in case
        rec.pitch_detector.detect_pitches_async()

    def detect_notes(self):
        """Run note detection on the active recording's current pitch data, then
        refresh the user-note view."""
        rec = self.active_recording
        if rec is None:
            return
        rec.detect_notes()
        self.guitar_hero.load_user(rec)

    def _detection_in_flight(self) -> bool:
        """True while the active recording's offline pitch detection+smoothing
        thread is still running (so its pitch_data isn't ready to analyze)."""
        rec = self.active_recording
        if rec is None:
            return False
        thread = getattr(rec.pitch_detector, "offline_thread", None)
        return bool(thread and thread.is_alive())

    def _on_detection_finished(self):
        """Offline pitch detection finished (queued onto the main thread): clear
        the status message and load the now-ready pitch data into the view. If an
        Analyze press was deferred while detection ran, run it now."""
        self.status_bar.update_status("")
        self.guitar_hero.load_user(self.active_recording)
        if self._pending_analyze:
            self._pending_analyze = False
            self.analyze()

    # --- PLAYBACK / RECORDING TOGGLES ---
    def toggle_playback(self):
        t = self.slider.get_time()

        if not self.is_playing:
            self.is_playing = True
            self.wall_clock.start(t)
            self.midi_player.play(start_time=t)
            if self.user_playback_enabled:
                self.audio_player.play(start_time=t)
            # update UI
            self.play_button.setIcon(self.pause_icon)

        elif self.is_playing:
            self.is_playing = False
            self.wall_clock.pause()
            self.midi_player.stop()
            self.audio_player.stop()
            # update UI
            self.play_button.setIcon(self.play_icon)

    def toggle_recording(self):
        if self._require_recording("Please select a recording to record into.") is None:
            return
        if self.is_counting_in:
            # clicking again during the count-in cancels it (un-arm recording)
            self.countdown_timer.cancel()
            self.is_counting_in = False
            self.record_button.setIcon(self.record_icon)
            return
        if not self.is_recording:
            # play a one-measure metronome count-in; _start_recording on finish
            self.is_counting_in = True
            self.record_button.setIcon(self.pause_icon)
            self.countdown_timer.start(
                beats=self.score_data.count_in_beats(),
                channel=self.score_data.metronome_channel,
            )
        else:
            self._stop_recording()

    def _start_recording(self):
        """Called when the countdown timer finishes, to start the
        recording and playback."""
        # update UI
        self.record_button.setIcon(self.pause_icon)
        # stuff
        self.is_counting_in = False
        t = self.slider.get_time()
        self.is_recording = True
        self.audio_player.stop()
        self.wall_clock.start(t)
        self.audio_recorder.run(start_time=t)
        self.active_recording.pitch_detector.run(start_time=t)
        self.midi_player.play(start_time=t) # play whatever audio the user has enabled

    def _stop_recording(self):
        """Called when user clicks the record button while already recording, 
        to stop the recording and playback."""
        # update UI
        self.record_button.setIcon(self.record_icon)
        # stuff
        self.is_recording = False
        self.wall_clock.pause()
        self.audio_recorder.stop()
        self.midi_player.stop()
        self.active_recording.pitch_detector.stop()

    def _find_best_w2(self):
        """refactor later but essentially
        finds the best note detection frame size by minimizing mistakes
        then sets the config to that
        """
        # parameter sweep for ND
        W2_SIZES = [33, 31, 29, 27, 25, 23, 21, 19, 17]

        min_mistake, best_w2 = float('inf'), None
        for w2 in W2_SIZES:
            self.active_recording.config.w2 = w2
            self.active_recording.config.h2 = w2 - 2
            self.active_recording.update_config(self.active_recording.config)
            # rec.detect_pitches()
            self.active_recording.detect_notes()
            self.active_recording.detect_mistakes()
            # print(self.active_recording.alignment)

            if len(self.active_recording.alignment.mistakes) < min_mistake:
                min_mistake = len(self.active_recording.alignment.mistakes)
                best_w2 = w2
        
        print(f"best ND frame-size: {best_w2}, min mistakes: {min_mistake}")

        self.active_recording.config.w2 = best_w2
        self.active_recording.config.h2 = best_w2 - 2
        self.active_recording.update_config(self.active_recording.config)

    def mistake_correction_loop(self):
        """run mistake correction until it stops improving"""
        if not self.active_recording or not self.active_recording.alignment:
            print("No active recording or alignment to correct mistakes on.")
            return
        n_mistakes = len(self.active_recording.alignment.mistakes)
        print(f"initial mistakes: {n_mistakes}")
        while True:
            # keep the current (best-so-far) state in case this pass regresses
            prev_nd = self.active_recording.note_data
            prev_alignment = self.active_recording.alignment

            self.active_recording.correct_mistakes()
            new_n_mistakes = len(self.active_recording.alignment.mistakes)
            print(f" > mistakes after correction: {new_n_mistakes}")

            if new_n_mistakes >= n_mistakes:
                # correction stopped helping (plateaued or got worse) -> revert
                # to the better previous result and stop
                self.active_recording.note_data = prev_nd
                self.active_recording.alignment = prev_alignment
                print("no improvement after correction, breaking loop")
                break
            n_mistakes = new_n_mistakes  # made progress -> raise the baseline

    def analyze(self):
        rec = self._require_recording()
        if rec is None:
            return
        # don't analyze raw/partial pitches: if offline detection+smoothing is
        # still running in the background, defer until it finishes (the smoothed
        # track has the octave errors / noise cleaned up). _on_detection_finished
        # will re-call analyze() once the pitch_data is ready.
        if self._detection_in_flight():
            self._pending_analyze = True
            self.status_bar.update_status("Detecting pitches… will analyze when ready")
            return
        print("analyzing... ")
        self.cleanup() # clear stale notes/alignment/mistakes before recomputing

        # detect notes (at the best ND frame size), stretch the score to match
        # the take's length (also rescales p.distances), then string-edit align
        self._find_best_w2()
        self.detect_notes()
        length = rec.get_length(raw=True)
        rec.resize(new_length=length)
        rec.detect_mistakes()
        self.mistake_correction_loop()

        # color the user pitches by the final alignment (insertions red, others by
        # distance to their *aligned* score note) instead of the live per-frame
        # distance. Runs after the correction loop so it reflects the final pairs.
        rec.update_alignment_distances()

        # reload every view with the fresh analysis (note/alignment may have been
        # overwritten by the correction loop)
        self.guitar_hero.load_alignment(rec.alignment)
        self.guitar_hero.load_user(rec)
        self.guitar_hero.update_view_items()
        self.slider.update_range(score_data=self.score_data, recording=rec)
        self.mistake_widget.load_mistakes(rec.alignment.mistakes)
        
    # --- SIGNAL-RELATED ACTIONS ---
    def _score_viewer_time(self, t: float) -> float:
        """Map a wall-clock time `t` (in the *current* tempo's timeframe) back
        into the *original* score tempo's timeframe.

        The Verovio render is never reloaded on a tempo change, so its internal
        timemap (used by getElementsAtTime) stays in the original tempo. Playing
        at a different tempo stretches/squeezes wall-clock time by
        bpm_og / bpm, so we undo that here before driving the score cursor.
        """
        bpm_og = self.score_data.bpm_og or self.score_data.bpm
        if not bpm_og:
            return t
        return t * self.score_data.bpm / bpm_og

    def update_time_label(self, t: float):
        """Update the time label based on current time t."""
        def format_time(seconds: float) -> str:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins:02}:{secs:04.1f}"

        current_time_str = format_time(t)
        total_length = self.slider.get_total_time()
        total_time_str = format_time(total_length)
        self.time_label.setText(f"{current_time_str} / {total_time_str}")

    def time_changed(self, t: float):
        """Called when the wall clock time changes. Update the time label and
        move the score viewer and guitar hero plots IF currently playing."""
        self.update_time_label(t)
        if not self.is_playing:
            return
        # else, move the score and guitar hero plots
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)

    def slider_changed(self, t: float):
        """Called when slider is moved, to handle case when we are not in playback
        or recording mode but still want to see our plots move."""
        self.update_time_label(t)
        if self.is_playing:
            return
        # else, move the score and guitar hero plots
        self.score_data.update_time(t)
        self.score_viewer.set_playback_time(self._score_viewer_time(t))
        self.guitar_hero.move_plot(t)

    def slider_end(self, t: float):
        pass

    def on_recording_selected(self, recording_name: str):
        """When a recording is selected from the recordings tree, update the active 
        recording and refresh the score viewer and other relevant UI components."""
        if recording_name not in self.recordings.keys():
            print(f"No recording named '{recording_name}' was found.")
            return
        self.active_recording = self.recordings[recording_name]
        self.init_pitch_detector_signals()
        # print(f"Setting active recording to '{recording_name}'")
        self.instrument_panel.set_active_instrument(self.active_recording.active_instrument)
        # reflect this recording's Config in the tunable inputs
        self.instrument_panel.set_tuning(self.active_recording.config.tuning)
        self.tolerance_panel.set_tolerance(self.active_recording.config.tolerance)
        self.status_bar.update_name(recording_name)
        # show this recording's own analysis (load_user draws its GuitarHero
        # overlays; keep the mistake list in sync rather than blanking it)
        self.mistake_widget.load_mistakes(self.active_recording.alignment.mistakes)
        self.guitar_hero.load_user(self.active_recording)
        self.audio_player.load_audio(self.active_recording.audio_data)
        self.audio_recorder.load_recording(self.active_recording)
        self.slider.update_range(score_data=self.score_data, recording=self.active_recording)

    def on_instrument_applied(self, channel: int):
        """Make `channel` the active instrument for the active recording. Resets
        all analysis-derived data (notes, alignment, mistakes/overrides), re-inits
        the algorithms from the current Config, and refreshes the views."""
        rec = self._require_recording()
        if rec is None:
            return

        # the active instrument drives both display (score_data) and analysis
        # (rec.active_instrument is what detect_mistakes / pitch distances key on)
        self.score_data.active_instrument = channel
        rec.active_instrument = channel

        # wipe the analysis-derived data + views, then re-init the algorithm
        # stages from the (unchanged) Config
        self.cleanup()
        rec.update_config(rec.config)

        # re-render the score viewer for the new instrument (no-op extra cost
        # during playback: this only runs here, on the instrument change)
        self.refresh_score_viewer()

        # the range defaults follow the newly selected instrument's note range
        self.instrument_panel.populate_range_from_score(self.score_data, channel)

    def on_full_score_toggled(self, show_full: bool):
        """Toggle the score viewer between the full score and the active
        instrument's part. Re-render lag here is acceptable."""
        self.viewer_show_full = show_full
        self.refresh_score_viewer()

    def refresh_score_viewer(self):
        """(Re)render the score viewer to match the current instrument/full-score
        state. The single expensive Verovio layout step — never called per tick."""
        if self.score_data.score is None:
            return
        channel = None if self.viewer_show_full else self.score_data.active_instrument
        self.score_viewer.load_score(self.score_data, channel=channel)

    def on_range_applied(self, low_midi: int, high_midi: int):
        """Set the active recording's Config frequency range from the chosen
        lowest/highest notes, then re-compute pitches with the new Config."""
        rec = self._require_recording()
        if rec is None:
            return
        config = rec.config

        # half-semitone padding so the boundary notes sit comfortably inside the
        # detectable range. fmin/fmax drive the detector's tau_max/tau_min.
        MARGIN = 0.5
        config.fmin = config.midi_to_freq(low_midi - MARGIN)
        config.fmax = config.midi_to_freq(high_midi + MARGIN)
        rec.update_config(config)

        # re-run pitch detection on the existing audio (if any) with the new range
        self.detect_pitches()

    def on_tuning_applied(self, tuning: float):
        """Set the active recording's Config tuning (A4 reference, Hz), then
        re-compute pitches with it (tuning drives the freq<->MIDI conversion)."""
        rec = self._require_recording()
        if rec is None:
            return
        rec.config.tuning = tuning
        rec.update_config(rec.config)

        # re-run pitch detection on the existing audio (if any) with the new tuning
        self.detect_pitches()

    def on_tolerance_applied(self, tolerance: float):
        """Set the active recording's Config tolerance (semitone slack for a note
        to count as correct), then re-run just the mistake-detection (string-edit)
        step and refresh the mistake list + GuitarHero overlays."""
        rec = self._require_recording()
        if rec is None:
            return
        rec.config.tolerance = tolerance
        rec.update_config(rec.config)

        self.analyze()

    def on_score_renamed(self, title: str):
        """The Score Title was edited in the RecordingTree (the source of truth).
        Push it to the score and re-render so Verovio shows the new title."""
        self.score_data.set_title(title)
        self.refresh_score_viewer()

    def on_score_viewer_loaded(self):
        """Called after score viewer is done loading JS and ready to receive data."""
        DEMO_SCORE_PATH = Path(__file__).resolve().parent / "resources" / "scores" / "c_major_scale.mxl"
        self.load_score(str(DEMO_SCORE_PATH))

    def on_user_audio_toggled(self, checked: bool):
        """Called when user toggles the user audio playback option in the toolbar."""
        self.user_playback_enabled = checked
        if not checked:
            self.audio_player.stop()

    def on_tempo_changed(self, new_bpm: int):
        """Called when user changes the tempo in the toolbar. Update the score data and 
        midi player accordingly."""
        if self.is_playing: # revert the ui back to old value
            self.toolbar.tempo_spinbox.setValue(self.score_data.bpm)
            return
        self.score_data.change_tempo(new_bpm)
        self.guitar_hero.update_view_items()
        # don't reload the score viewer: Verovio keeps its original-tempo render
        # and _score_viewer_time() maps the new wall-clock time back into it.
        self.slider.update_range(score_data=self.score_data, recording=self.active_recording)

    def on_mistake_override_toggled(self, idx: int):
        if self.active_recording is None:
            return
        self.active_recording.toggle_mistake_override(idx)
        mistake = self.active_recording.alignment.mistakes[idx]
        self.mistake_widget.refresh_override(idx)
        self.guitar_hero.update_highlight_override(mistake.is_overridden())
        self.guitar_hero.update_view_items()

    def on_mistake_selected(self, idx: int):
        """When mistake selected, highlight corresponding note in guitar hero plot"""
        if self.active_recording is None:
            return
        mistakes = self.active_recording.alignment.mistakes
        if 0 <= idx < len(mistakes):
            self.guitar_hero.highlight_mistake(mistakes[idx])

    def on_practice_toggled(self):
        """Called when user clicks the practice mode button in the toolbar."""
        # open a confirmation popup
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Enter Practice Mode?")
        msg_box.setText("Are you sure you want to enter practice mode? This will open a new window.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes)
        result = msg_box.exec()

        if result == QMessageBox.StandardButton.Yes:
            # open new window with just guitar hero
            # implement later
            print("Entering practice mode...")
            self.practice_attune.show()
            self.practice_attune.raise_() # bring to front
            self.practice_attune.activateWindow() # focus
        else:
            # close popup
            msg_box.close()


if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = Attune()
    window.show()
    sys.exit(app.exec())