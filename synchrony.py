
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QStatusBar
)
import qdarktheme

# app logic imports
from app_logic.user.ds.UserData import UserData
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.midi.MidiData import MidiData
from app_logic.midi.MidiSynth import MidiSynth
from app_logic.midi.MidiPlayer import MidiPlayer

# ui imports
from ui.Toolbar import Toolbar
from ui.RecordTab import RecordTab

class Synchrony(QMainWindow):
    def __init__(self):
        # user / midi data
        self.user_data = UserData()
        self.midi_data = None

        # important midi playback things
        self.SOUNDFONT = "resources/MuseScore_General.sf3"
        self.midi_synth: MidiSynth = MidiSynth(self.SOUNDFONT)
        self.midi_player: MidiPlayer = MidiPlayer(self.midi_synth)

        # important audio record/playback things
        self.audio_player: AudioPlayer = AudioPlayer()
        self.audio_recorder: AudioRecorder = AudioRecorder(self.user_data)

        self.init_ui() # create the main window

        # signals
        self.toolbar.midi_uploaded.connect(self.midi_uploaded)
        self.toolbar.audio_uploaded.connect(self.audio_uploaded)

    def init_ui(self):
        # the main window atop which everything is drawn upon
        super().__init__() # init the mainwindow it inherits from
        self.setWindowTitle("Synchrony")
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        self.setGeometry(100, 100, 800, 600)

        # central widget for everything
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = QVBoxLayout(self.central_widget)

        # tab manager
        self.tab_manager = QTabWidget()
        self._layout.addWidget(self.tab_manager)

        # record tab
        self.record_tab = RecordTab(self.user_data, self.midi_data, self.audio_player, self.audio_recorder, self.midi_player)
        self.tab_manager.addTab(self.record_tab, "Record")
        self.analyze_tab = QWidget()
        self.tab_manager.addTab(self.analyze_tab, "Analyze")

        # toolbar / status bar
        self.setStatusBar(QStatusBar(self))
        self.toolbar = Toolbar()
        self.addToolBar(self.toolbar)
        

    def midi_uploaded(self, midi_filepath: str):
        """create a new midi_data based on the received midi filepath signal"""
        print(f"MIDI uploaded: {midi_filepath}")
        self.midi_data = MidiData(midi_filepath)
        self.record_tab.load_midi(self.midi_data)

    def audio_uploaded(self, audio_filepath: str):
        """create a new audio_data based on the received audio filepath signal
        uses current settings_panel's sample rate for sampling resolution

        also plots the pitches on the pitchplot once done
        """
        print(f"audio uploaded: {audio_filepath}")
        self.user_data.load_audio(audio_filepath)
        self.record_tab.load_audio(self.user_data)

if __name__ == '__main__':
    # create the pyqt app instance and run it
    app = QApplication(sys.argv)
    main_window = Synchrony()
    main_window.show()
    sys.exit(app.exec())
