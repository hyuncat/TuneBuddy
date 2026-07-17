import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStatusBar, QPushButton, QLabel
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon
import os
import qdarktheme

from ui.info.Toolbar import Toolbar
from ui.GuitarHero import ScorePlot
from ui.time.Slider import Slider
from ui.time.WallClock import WallClock
from notebooks.archive.Settings import SettingsDialog
from ui.time.Clipper import ClipperDialog

# app logic imports
from app_logic.user.ds.Recording import UserData
from app_logic.user.ds.PitchData import PitchConfig
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from notebooks.archive.MidiData import MidiData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.Alignment import Alignment

from algorithms.Config import Config
from algorithms.PitchDetector import PitchDetector
from algorithms.NoteDetector import NoteDetector
from algorithms.StringEditor import StringEditor

class TuneBuddy(QMainWindow):
    def __init__(self):
        super().__init__()
        # algorithms
        config = {
            'sr': 44100,    # sample rate
            'w1': 1024 * 2,  # frame size
            'h1': 128,       # hop size
            'fmin': 196.0,
            'fmax': 3000.0,
            'tuning': 440.0,
            'unv_thresh': 0.9, # if unvoiced_prob > unv_thresh, consider the frame unvoiced

            # --- NOTE DETECTION PARAMETERS ---
            'w2': 21, # frame size (NOTE: should always be odd)
            'h2': 19, # hop size
            'pitch_thresh': 0.5,
            'slope_thresh': 0.75 / 21,
            'unv_ratio': 0.8, # proportion of unvoiced pitches in a window to consider the window unvoiced

            # --- STRING EDIT PARAMETERS ---
            'ins_cost': 1.5,
            'del_cost': 2,
            'sub_cost': 1,
            'tolerance': 1,
            # tiger-mom parameter
            'tiger_level': 1
        }
        self.config = Config(**config)
        self.pitch_detector = PitchDetector(self.config)
        self.note_detector = NoteDetector(self.config)
        self.string_editor = StringEditor(self.config)

        self.user_data: UserData = UserData(self.pitch_detector, self.note_detector)
        self.pitch_detector.init_user_data(self.user_data)
        self.midi_data: MidiData = None
        self.alignment: Alignment = None

        # very important master clock for timekeeping
        self.wall_clock = WallClock(hz=10) # 10 updates per second

        # important midi playback things
        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        self.midi_player = MidiPlayer(self.midi_synth, self.wall_clock)

        # important audio record/playback things
        self.audio_player = AudioPlayer(self.user_data.audio_data)
        self.audio_recorder = AudioRecorder(self.user_data)
        # --> playback state variables
        self.is_playing = False
        self.is_recording = False
        self.user_playback_enabled = True

        # initialize important stuff
        self.init_ui()
        self.init_signals()

    def init_ui(self):
        self.setWindowTitle("TuneBuddy")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = QVBoxLayout(self.central_widget)

        # setup essential widgets
        self.score_plot = ScorePlot(self.midi_data, self.user_data)
        self._layout.addWidget(self.score_plot)
        self.init_slider_layout()

        # side utilities
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("")
        self.status_bar.addWidget(self.status_label)

        # countdown timer
        self.countdown_timer = QTimer(self)
        self.countdown_ms = 0 # current countdown time in ms
        self.countdown_timer.timeout.connect(self._update_countdown)

        self.toolbar = Toolbar()
        self.addToolBar(self.toolbar)
        self.settings_dialog = SettingsDialog()
        self.clipper_dialog = ClipperDialog()

        self.show() # run the show

    def init_slider_layout(self):
        """Initialize the layout containing the 
        play/pause, record button and the slider.
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
        """Initialize all the signals and slots."""
        # toolbar signals
        self.toolbar.midi_uploaded.connect(self.load_midi)
        self.toolbar.audio_uploaded.connect(self.load_audio)
        self.toolbar.show_settings.connect(self.settings_dialog.show)
        self.toolbar.show_clipper.connect(self.clipper_dialog.show)

        self.pitch_detector.pitch_detected.connect(self.score_plot.update_view_items)

        # timekeeping signals
        self.wall_clock.time_changed.connect(self.score_plot.move_plot)
        self.slider.slider_changed.connect(self.move_plot_from_slider)
        self.slider.slider_changed.connect(self.update_time_label)
        self.slider.slider_end.connect(self._slider_end_reached)

        # settings dialog signals
        self.settings_dialog.settings_panel.settings_applied.connect(self.apply_settings)
        self.clipper_dialog.clipper_panel.clip_triggered.connect(self.handle_clip)

    # --- LOADING FUNCTIONS ---
    def load_midi(self, filepath):
        """Load a MIDI or musicXML file."""
        print(f"MIDI uploaded: {filepath}")
        # load into data structures
        self.midi_data= MidiData(filepath)
        self.midi_player.load_midi(self.midi_data)

        # update UI + settings panel
        self.slider.update_slider_range(midi_data=self.midi_data)
        self.score_plot.load_midi(self.midi_data)
        self.settings_dialog.settings_panel.load_midi(self.midi_data)
        self.clipper_dialog.clipper_panel.load_midi(self.midi_data)

    def load_audio(self, filepath):
        """Load an audio file."""
        print(f"Audio uploaded: {filepath}")
        # load the audio into the corresponding user data structures
        self.user_data = UserData(self.pitch_detector, self.note_detector)
        self.user_data.load_audio(filepath)
        self.audio_player.load_audio(self.user_data.audio_data)

        # update UI
        self.slider.update_slider_range(user_data=self.user_data) # update slider range
        # update the score plot with user notes and pitches
        self.score_plot.load_user(self.user_data)

    # --- KEY FUNCTIONALITIES ---
    # playback, record, analyze
    def toggle_playback(self):
        """Toggle playback of MIDI and/or audio."""
        if not self.is_playing: # currently not playing (either midi or user audio)
            print("Starting playback...")
            self.is_playing = True

            self.play_button.setIcon(self.pause_icon)
            t = self.slider.get_time()

            # ensure wall clock is started
            self.wall_clock.start(t=t) 

            # start midi playback
            self.midi_player.play(start_time=t)
            # alternatively, start audio playback
            if self.user_playback_enabled:
                self.audio_player.play(start_time=t)

        else:
            print("Pausing playback...")
            self.is_playing = False
            self.play_button.setIcon(self.play_icon)
            # pause our shit
            self.wall_clock.pause()
            # rk: the following don't explicitly stop wall_clock
            self.midi_player.pause() 
            self.audio_player.pause() 

    def toggle_recording(self):
        """Toggle recording of audio."""
        if not self.is_recording:
            print("Starting recording...")
            self.is_recording = True
            # start recording logic here
            # WAIT FOR 2 SEC BEFORE STARTING
            self._start_countdown(2.0)
            # the above also starts in the _start_countdown function
        else:
            print("Stopping recording...")
            self.is_recording = False
            self.audio_recorder.stop()
            self.pitch_detector.stop()
            self.wall_clock.pause()
            # stop recording logic here

    def analyze(self):
        # print("analyze() placeholder...")
        assert self.midi_data is not None, "No MIDI data loaded!"
        assert self.user_data is not None, "No user audio data loaded!"

        self.user_data.detect_notes()
        self.midi_data.resize(new_length=self.user_data.audio_data.get_length())
        self.slider.update_slider_range(
            midi_data=self.midi_data, 
            user_data=self.user_data
        )

        self.alignment = self.string_editor.string_edit(
            self.user_data.note_data, 
            self.midi_data.note_data
        )
        self.score_plot.load_alignment(self.alignment)

    # --- UTILITIES ---
    def apply_settings(self, settings: dict):
        """Apply settings from the settings dialog."""
        # apply pyin settings
        pitch_config_settings = {
            "fmin": settings.get("fmin", 80.0),
            "fmax": settings.get("fmax", 1000.0),
            "tuning": settings.get("tuning", 440.0)
        }
        self.config.fmin = pitch_config_settings["fmin"]
        self.config.fmax = pitch_config_settings["fmax"]
        self.config.tuning = pitch_config_settings["tuning"]

        self.pitch_detector.load_config(self.config)

        # apply midi channel settings
        self.midi_data.change_tempo(settings.get("tempo", 90))
        active_channels = settings.get("active_channels", [])
        self.midi_player.set_channels(active_channels)
        self.user_playback_enabled = settings.get("user_playback", True)
        self.score_plot.update_view_items()

    def handle_clip(self):
        """update slider and plot based on clip boundaries"""
        self.score_plot.update_view_items()
        self.slider.update_slider_range(midi_data=self.midi_data, user_data=self.user_data)
        # print("CLIPPED.")
    
    def _start_countdown(self, countdown_sec: float):
        """Start a countdown before recording.
        Args:
            countdown_sec (float): countdown time in seconds
        """
        self.countdown_ms = int(countdown_sec * 1000)
        self.status_label.setText(f"Starting in {countdown_sec:.1f} sec)")
        self.countdown_timer.start(100) # update every 100 ms

    def _update_countdown(self):
        """Update the countdown timer in the status bar.
        Also starts the recording when countdown reaches 0."""
        self.countdown_ms -= 100 # decrement since this is called every 100ms

        # STOP CONDITION
        if self.countdown_ms <= 0:
            self.countdown_timer.stop()
            self.status_label.setText("Recording...")
            # --> START RECORDING
            t = self.slider.get_time()
            self.wall_clock.start(t=t)
            self.audio_recorder.run(start_time=t)
            self.pitch_detector.run(start_time=t)
        
        seconds_left = self.countdown_ms / 1000
        self.status_label.setText(f"Starting in (s): {seconds_left:.2f}")
    
    def _slider_end_reached(self):
        """Handle the event when the slider reaches the end."""
        if self.is_playing:
            self.is_playing = False
            self.play_button.setIcon(self.play_icon)
            self.wall_clock.pause()
            self.midi_player.pause()
            self.audio_player.pause()



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

    def move_plot_from_slider(self, t: float):
        """Move the plot to time t (sec) from the slider."""
        if self.is_playing:
            return # ignore slider moves while playing
        self.score_plot.move_plot(t)

if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = TuneBuddy()
    window.show()
    sys.exit(app.exec())