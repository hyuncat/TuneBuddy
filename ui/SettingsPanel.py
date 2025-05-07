from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QDoubleSpinBox,  QGroupBox, QCheckBox, QFrame
)
import logging
from app_logic.midi.MidiData import MidiData

class SettingsPanel(QWidget):
    """basically a UI to control the following
    1. recording settings
        a. tuning
        b. instrument (fmin + fmax)
    2. playback settings
        a. midi playback tempo
        b. midi instrument control
        c. user playback dis/en-able
    """
    # recording signals
    tuning_changed = pyqtSignal(float)
    fmin_changed = pyqtSignal(float)
    fmax_changed = pyqtSignal(float)

    # channel handling signals
    midi_channels_active = pyqtSignal(dict)
    user_channel_active = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        
        # user instrument configuration variables
        self.SAMPLE_RATE: int = 44100
        self.MIN_FREQ: float = 196
        self.MAX_FREQ: float = 3000
        self.TUNING: float = 440

        # user midi config variables
        self.TEMPO: float = 90

        self.INSTRUMENTS = [
            "violin", "viola", "cello", "voice", "piano"
        ]
        self.INSTRUMENT_RANGES = {
            "violin": (196, 3500),
            "viola": (125, 1000), 
            "cello": (63, 630), 
            "voice": (80, 1000), 
            "piano": (28, 4100)
        }

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # ui components
        self.textbf = QFont()
        self.textbf.setBold(True)

        self.init_recording_settings()
        self.init_playback_settings()


    def init_recording_settings(self):
        # --- RECORDING SETTINGS ---
        self.recording_title = QLabel("Recording Settings")
        self.recording_title.setFont(self.textbf)
        self._layout.addWidget(self.recording_title)

        # instrument selection
        self._layout.addWidget(QLabel("Select instrument"))
        self.instrument_selector = QComboBox()
        self.instrument_selector.addItems(self.INSTRUMENTS)
        self.instrument_selector.currentTextChanged.connect(self.instrument_selected)
        self._layout.addWidget(self.instrument_selector)

        # fmin/fmax settings
        self.fmin_layout = QHBoxLayout()
        self.fmin_label = QLabel("Min freq: ")
        self.fmin_input = QLineEdit()
        self.fmin_input.setText(str(self.MIN_FREQ))

        self.fmin_layout.addWidget(self.fmin_label)
        self.fmin_layout.addWidget(self.fmin_input)
        self.fmin_input.textChanged.connect(lambda value: self.pyin_setting_changed(value, type="fmin"))
        self._layout.addLayout(self.fmin_layout)

        self.fmax_layout = QHBoxLayout()
        self.fmax_label = QLabel("Max freq: ")
        self.fmax_input = QLineEdit()
        self.fmax_input.textChanged.connect(lambda value: self.pyin_setting_changed(value, type="fmax"))
        self.fmax_input.setText(str(self.MAX_FREQ))

        self.fmax_layout.addWidget(self.fmax_label)
        self.fmax_layout.addWidget(self.fmax_input)
        self._layout.addLayout(self.fmax_layout)

        # tuning
        self.tuning_layout = QHBoxLayout()
        self.tuning_label = QLabel("Tuning: ")
        self.tuning_input = QLineEdit()
        self.tuning_input.setText(str(self.TUNING))
        self.tuning_input.textChanged.connect(lambda value: self.pyin_setting_changed(value, type="tuning"))

        self.tuning_layout.addWidget(self.tuning_label)
        self.tuning_layout.addWidget(self.tuning_input)
        self._layout.addLayout(self.tuning_layout)

    def init_playback_settings(self):
        """initialize the following playback settings
            - tempo
            - user playback enable
            - midi instrument playback selection
        """
        # --- MIDI SETTINGS ---
        self.midi_title = QLabel("MIDI Settings")
        self.midi_title.setFont(self.textbf)
        self._layout.addWidget(self.midi_title)

        # TEMPO
        self.midi_tempo_layout = QHBoxLayout()
        self.midi_tempo = QDoubleSpinBox()

        self.midi_tempo.setValue(self.TEMPO)
        self.midi_tempo_label = QLabel("Tempo: ")

        self.midi_tempo_layout.addWidget(self.midi_tempo_label)
        self.midi_tempo_layout.addWidget(self.midi_tempo)
        self._layout.addLayout(self.midi_tempo_layout)

        # --- PLAYBACK CHANNELS ---
        # (aka instrument playback selection)
        self.playback_title = QLabel("Playback Channels")
        self.playback_title.setFont(self.textbf)
        self._layout.addWidget(self.playback_title)

        # put all checkboxes in a groupbox
        self.playback_groupbox = QGroupBox()
        self.playback_layout = QVBoxLayout()
        self.playback_groupbox.setLayout(self.playback_layout)

        # USER PLAYBACK CHECKBOX
        self.user_checkbox = QCheckBox("user")
        self.user_checkbox.setCheckState(Qt.CheckState.Checked)
        self.user_checkbox.stateChanged.connect(self.user_checkbox_selected)
        self.playback_layout.addWidget(self.user_checkbox)

        # dividing line between
        hr = QFrame()
        hr.setFrameShape(QFrame.Shape.HLine)
        hr.setFrameShadow(QFrame.Shadow.Sunken)
        self.playback_layout.addWidget(hr)

        # MIDI INSTRUMENT PLAYBACK CHECKBOXES
        self.midi_checkboxes: dict[int, QCheckBox] = {}
        for instrument in self.INSTRUMENTS:
            checkbox = QCheckBox(instrument)
            self.midi_checkboxes[instrument] = checkbox
            self.playback_layout.addWidget(checkbox)
        self._layout.addWidget(self.playback_groupbox)
        
        self._layout.addStretch() # ensures things don't expand to fit entire vertical

    def instrument_selected(self, text):
        fmin, fmax = self.INSTRUMENT_RANGES[text]
        self.MIN_FREQ = fmin
        self.MAX_FREQ = fmax
        # update labels for user
        self.fmin_input.setText(str(self.MIN_FREQ))
        self.fmax_input.setText(str(self.MAX_FREQ))

    def load_midi(self, midi_data: MidiData):
        # clear current items
        for c in self.midi_checkboxes.values():
            self.playback_layout.removeWidget(c)
        self.midi_checkboxes = {}

        self.TEMPO = midi_data.get_tempo()
        self.midi_tempo.setValue(self.TEMPO)
        
        for i, program in enumerate(midi_data.get_programs()):
            checkbox = QCheckBox(f"channel={i}, program={program}")
            checkbox.setCheckState(Qt.CheckState.Checked)
            checkbox.stateChanged.connect(self.channels_selected)
            self.midi_checkboxes[i] = checkbox
            self.playback_layout.addWidget(checkbox)

    def user_checkbox_selected(self, s: bool):
        """called whenever the user checkbox state changes

        Args:
            s: a boolean whether it's checked or not
        """
        self.user_channel_active.emit(s)
        print(f"user_is_playing changed: {s}")

    def channels_selected(self, s):
        """emits a dictionary of active channels
        with the program and whether it's active or not
        """
        active_channels = {}
        for program, midi_checkbox in self.midi_checkboxes.items():
            active_channels[program] = midi_checkbox.checkState()
        self.midi_channels_active.emit(active_channels)

    def pyin_setting_changed(self, value, type: str):
        value = float(value)
        if type == "fmin":
            self.fmin_changed.emit(value)
        elif type == "fmax":
            self.fmax_changed.emit(value)
        elif type == "tuning":
            self.tuning_changed.emit(value)
        else:
            logging.error("invalid pyin setting type (how did this happen lol?)")
