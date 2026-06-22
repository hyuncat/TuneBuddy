# code for practice mode
from __future__ import annotations
import os

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QStatusBar, QPushButton, QLabel, QTreeWidget, QTreeWidgetItem, QSplitter,
    QInputDialog, QMenu, QMessageBox, QStackedLayout
)
from PyQt6.QtCore import Qt, QSize, QTimer, QPoint, pyqtSignal, QObject
from PyQt6.QtGui import QIcon

from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.Recording import Recording

from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.midi.ScoreData import ScoreData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer

# adjust this import to wherever your GuitarHero widget lives
from ui.GuitarHero import GuitarHero
from ui.info.Toolbar import Toolbar
from ui.info.StatusBar import StatusBar
from ui.time.CountdownTimer import CountdownTimer
from ui.time.WallClock import WallClock
from ui.time.Slider import Slider


class PracticeAttune(QMainWindow):
    """
    Standalone practice-mode window.

    For now this is just a thin shell around a GuitarHero widget.
    Later it can own its own playback controls, countdown, scoring,
    user recording logic, note-hit detection, etc.
    """

    def __init__(self, score_data: ScoreData, midi_synth: MidiSynth=None, parent=None):
        super().__init__(parent)
        self.score_data = score_data
        self.recording = Recording(score_data=self.score_data)
        self.wall_clock = WallClock(hz=10)

        self.is_playing = False
        self.is_recording = False
        self.is_counting_in = False

        self.audio_recorder = AudioRecorder(self.recording)
        self.midi_synth = midi_synth if midi_synth is not None else MidiSynth("resources/MuseScore_General.sf3")
        self.midi_player = MidiPlayer(self.midi_synth, self.wall_clock)
        self.midi_player.load_score(self.score_data)

        self.init_ui()
        self.init_slider_layout()
        self.init_signals()

    def init_ui(self):
        """Create the basic window layout."""
        self.setWindowTitle("Practice Mode")
        self.resize(1200, 700)

        central = QWidget(self)
        self.setCentralWidget(central)

        self._layout = QVBoxLayout(central)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)

        self.guitar_hero = GuitarHero(self.recording)
        self.guitar_hero.load_score(self.score_data)
        self._layout.addWidget(self.guitar_hero)

        # --- UTILITIES --- 
        self.status_bar = StatusBar() # with default recording name
        self.status_bar.update_status("Ready...")
        self.setStatusBar(self.status_bar)
        self.countdown_timer = CountdownTimer(self.status_bar, midi_synth=self.midi_synth)

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

    def init_signals(self):
        # timekeeping signals
        self.wall_clock.time_changed.connect(self.time_changed)
        self.slider.slider_changed.connect(self.slider_changed)
        self.slider.slider_end.connect(self.slider_end)
        self.countdown_timer.finished.connect(self._start_recording)
        self.recording.pitch_detector.pitch_detected.connect(self.pitch_detected)

    # --- PLAYBACK / RECORDING TOGGLES ---
    def toggle_playback(self):
        if not self.is_playing:
            self._start_playback()

        elif self.is_playing:
            self._stop_playback()

    def _start_playback(self):
        t = self.slider.get_time()
        self.is_playing = True
        self.wall_clock.start(t)
        self.midi_player.play(start_time=t)
        # update UI
        self.play_button.setIcon(self.pause_icon)

    def _stop_playback(self):
        self.is_playing = False
        self.wall_clock.pause()
        self.midi_player.stop()
        # update UI
        self.play_button.setIcon(self.play_icon)

    def toggle_recording(self):
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
        self.is_counting_in = False
        self.midi_player.stop() # stop things we don't want
        # reset stall variables
        self.wall_clock.stall = False
        self.recording.pitch_detector.block = False
        # stuff
        t = self.slider.get_time()
        self.is_recording = True
        self.wall_clock.start(t)
        self.audio_recorder.run(start_time=t)
        self.recording.pitch_detector.run(start_time=t)


    def _stop_recording(self):
        """Called when user clicks the record button while already recording, 
        to stop the recording and playback."""
        # update UI
        self.record_button.setIcon(self.record_icon)
        # stuff
        self.is_recording = False
        self.wall_clock.pause()
        self.audio_recorder.stop()
        self.recording.pitch_detector.stop()
    
    def time_changed(self, t: float):
        """Called when the wall clock time changes. Update the time label and
        move the score viewer and guitar hero plots IF currently playing."""
        if not (self.is_playing or self.is_recording):
            return
        self.guitar_hero.move_plot(t)

    def pitch_detected(self, t: float):
        """t = time at which a new pitch was detected"""
        # else, have the following logic:
        p = self.recording.pitch_data.read_pitch(t)
        if p is None or not p.candidates:  # no frame, or unvoiced (empty after smoothing)
            return
        u = p.candidates[0][0]
        m = self.score_data.note_datas[self.score_data.active_instrument].read_current_note(t)
        if m is None:
            return
        m = m.midi_num[0]
        print(f"detected pitch: {u}, target pitch: {m}")
        self.guitar_hero.update_view_items()

        if abs(u - m) > 1:
            self.status_bar.update_status(f"Off! Detected pitch: {u:.1f} Hz, Target pitch: {m:.1f} Hz")
            self.recording.pitch_detector.block = True
            self.wall_clock.stall = True
        else:
            self.status_bar.update_status(f"On! Detected pitch: {u:.1f} Hz, Target pitch: {m:.1f} Hz")
            self.recording.pitch_detector.block = False
            # move the guitar hero plot
            self.wall_clock.stall = False


    def slider_changed(self, t: float):
        """Called when slider is moved, to handle case when we are not in playback
        or recording mode but still want to see our plots move."""
        if self.is_playing:
            return
        # else, move guitar hero plot
        self.guitar_hero.move_plot(t)

    def slider_end(self, t: float):
        self._stop_recording()
        self._stop_playback()

    def load_score(self, score_data: ScoreData):
        """Load a score into the practice mode, initializing the recording and guitar hero with the new score data."""
        self.score_data = score_data
        self.recording = Recording(score_data=self.score_data)
        self.guitar_hero.load_score(self.score_data)
        self.guitar_hero.load_user(self.recording)
        self.midi_player.load_score(self.score_data)
        self.audio_recorder.load_recording(self.recording)
        self.recording.pitch_detector.pitch_detected.connect(self.pitch_detected)
        self.slider.update_range(score_data=self.score_data)

    def closeEvent(self, event):
        """Hook for cleanup later."""
        # future: stop timers / playback / recording threads here
        super().closeEvent(event)