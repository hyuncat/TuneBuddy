import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStatusBar, QPushButton
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
import os
import qdarktheme

from ui.Toolbar import Toolbar
from ui.ScorePlot import ScorePlot
from ui.Slider import Slider
from ui.WallClock import WallClock

# app logic imports
from app_logic.user.ds.UserData import UserData
from app_logic.user.ds.PitchData import PitchConfig
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.midi.MidiData import MidiData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer
from app_logic.Alignment import Alignment

from algorithms.PitchDetector import PitchDetector
from algorithms.NoteDetector import NoteDetector
from algorithms.StringEditor import StringEditor

class TuneBuddy(QMainWindow):
    def __init__(self):
        super().__init__()
        # algorithms
        self.pitch_detector = PitchDetector()
        self.note_detector = NoteDetector()
        self.string_editor = StringEditor(tiger_level=1)

        self.user_data: UserData = UserData(self.pitch_detector, self.note_detector)
        self.pitch_detector.init_user_data(self.user_data)
        self.midi_data: MidiData = None
        self.alignment: Alignment = None

        # very important master clock for timekeeping
        self.wall_clock = WallClock(hz=10) # 10 updates per second

        # important midi playback things
        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth = MidiSynth(self.SOUNDFONT)
        self.midi_player = MidiPlayer(self.midi_synth, self.wall_clock  )

        # important audio record/playback things
        self.audio_player = AudioPlayer(self.user_data.audio_data)
        self.audio_recorder = AudioRecorder(self.user_data)
        # --> playback state variables
        self.is_playing = False
        self.is_recording = False

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
        self.toolbar = Toolbar()
        self.addToolBar(self.toolbar)

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
        self.toolbar.show_settings.connect(self.show_settings_dialog)

        self.pitch_detector.pitch_detected.connect(self.pitch_detected)

        # timekeeping signals
        self.wall_clock.time_changed.connect(self.move_plot)
        self.slider.slider_changed.connect(self.move_plot_from_slider)
        # self.slider.slider_end.connect() #TODO: handle end of slider

    def load_midi(self, filepath):
        """Load a MIDI or musicXML file."""
        print(f"MIDI uploaded: {filepath}")
        self.midi_data = MidiData(filepath)
        self.midi_player.load_midi(self.midi_data)
        self.update_slider_range()
        self.score_plot.load_midi(self.midi_data)

    def load_audio(self, filepath):
        """Load an audio file."""
        print(f"Audio uploaded: {filepath}")
        self.user_data = UserData(self.pitch_detector, self.note_detector)
        self.user_data.load_audio(filepath)
        self.audio_player.load_audio(self.user_data.audio_data)
        self.update_slider_range()
        # update the score plot with user notes and pitches
        self.score_plot.load_audio(self.user_data)

    def update_slider_range(self):
        """Update the slider range based on max(MIDI.length, audio.length)"""
        l1, l2 = 0, 0
        if self.midi_data:
            l1 = self.midi_data.get_length()
        if self.user_data and self.user_data.audio_data:
            l2 = self.user_data.audio_data.get_length()

        self.slider.update_slider_range(max(l1, l2))

    def show_settings_dialog(self):
        """Show the settings dialog."""
        pass

    def toggle_playback(self):
        """Toggle playback of MIDI and/or audio."""
        if not self.is_playing:
            print("Starting playback...")
            self.is_playing = True

            self.play_button.setIcon(self.pause_icon)
            t = self.slider.get_time()
            self.wall_clock.start(t=t) # ensure wall clock is started

            # start midi playback
            self.midi_player.play(start_time=t) # also turns on the wall_clock
            # alternatively, start audio playback
            self.audio_player.play(start_time=t)

        else:
            print("Pausing playback...")
            self.play_button.setIcon(self.play_icon)

            self.wall_clock.pause()
            self.midi_player.pause() # doesn't stop wall_clock
            self.audio_player.pause() # doesn't stop wall_clock
            self.is_playing = False

    def toggle_recording(self):
        """Toggle recording of audio."""
        if not self.is_recording:
            print("Starting recording...")
            self.is_recording = True
            # start recording logic here
            t = self.slider.get_time()
            self.wall_clock.start(t=t)
            self.audio_recorder.run(start_time=t)
            self.pitch_detector.run(start_time=t)
        else:
            print("Stopping recording...")
            self.is_recording = False
            self.audio_recorder.stop()
            self.pitch_detector.stop()
            self.wall_clock.pause()
            # stop recording logic here

    def move_plot(self, t: float):
        """Move the plot to time t (sec)."""
        self.score_plot.move_plot(t)

    def move_plot_from_slider(self, t: float):
        """Move the plot to time t (sec) from the slider."""
        if self.is_playing:
            return # ignore slider moves while playing
        self.score_plot.move_plot(t)

    def analyze(self):
        print("analyze() placeholder...")
        assert self.midi_data is not None, "No MIDI data loaded!"
        assert self.user_data is not None, "No user audio data loaded!"

        self.alignment = self.string_editor.string_edit(
            self.user_data.note_data, 
            self.midi_data.note_data
        )
        # self.alignment.print_alignment()
        # self.alignment.print_mistakes_summary()
        self.score_plot.plot_alignment(self.alignment)

    def pitch_detected(self, pitch_onset: int):
        """Handle a new pitch detected event."""
        # update the score plot with the new pitch
        pitch = self.user_data.pitch_data.read_pitch(pitch_onset)
        self.score_plot


if __name__ == "__main__":
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    window = TuneBuddy()
    window.show()
    sys.exit(app.exec())