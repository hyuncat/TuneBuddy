from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QDoubleSpinBox,  QGroupBox, QCheckBox, QFrame, QDialog, QPushButton
)
import logging
from notebooks.archive.MidiData import MidiData
from ui.combo import center_combo_items

class SettingsPanel(QWidget):
    """basically a UI to control the following
    1. recording settings
        a. tuning
        b. instrument (fmin + fmax)
        c. max / min volume (tunable via calibration)
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

    # big signal
    settings_applied = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        
        # --- RECORDING VARIABLES ---
        self.SAMPLE_RATE: int = 44100
        self.TUNING: float = 440
        # min/max freq
        self.MIN_FREQ: float = 196
        self.MAX_FREQ: float = 3000
        # instrument selection
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
        self.init_program_map()

        # MIDI
        self.TEMPO: float = 90
        self.active_channels = {}

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # ui components
        self.textbf = QFont()
        self.textbf.setBold(True)

        self.init_recording_settings()
        self.init_playback_settings()
        self.settings_applied_button = QPushButton("Apply")
        self._layout.addWidget(self.settings_applied_button)
        self.settings_applied_button.clicked.connect(self.apply_settings)

    def init_program_map(self):
        self.GM_PROGRAM_MAP = {
            0:  "Acoustic Grand Piano",
            1:  "Bright Acoustic Piano",
            2:  "Electric Grand Piano",
            3:  "Honky-tonk Piano",
            4:  "Electric Piano 1",
            5:  "Electric Piano 2",
            6:  "Harpsichord",
            7:  "Clavinet",
            8:  "Celesta",
            9:  "Glockenspiel",
            10: "Music Box",
            11: "Vibraphone",
            12: "Marimba",
            13: "Xylophone",
            14: "Tubular Bells",
            15: "Dulcimer",
            16: "Drawbar Organ",
            17: "Percussive Organ",
            18: "Rock Organ",
            19: "Church Organ",
            20: "Reed Organ",
            21: "Accordion",
            22: "Harmonica",
            23: "Tango Accordion",
            24: "Acoustic Guitar (nylon)",
            25: "Acoustic Guitar (steel)",
            26: "Electric Guitar (jazz)",
            27: "Electric Guitar (clean)",
            28: "Electric Guitar (muted)",
            29: "Overdriven Guitar",
            30: "Distortion Guitar",
            31: "Guitar Harmonics",
            32: "Acoustic Bass",
            33: "Electric Bass (finger)",
            34: "Electric Bass (pick)",
            35: "Fretless Bass",
            36: "Slap Bass 1",
            37: "Slap Bass 2",
            38: "Synth Bass 1",
            39: "Synth Bass 2",
            40: "Violin",
            41: "Viola",
            42: "Cello",
            43: "Contrabass",
            44: "Tremolo Strings",
            45: "Pizzicato Strings",
            46: "Orchestral Harp",
            47: "Timpani",
            48: "String Ensemble 1",
            49: "String Ensemble 2",
            50: "SynthStrings 1",
            51: "SynthStrings 2",
            52: "Choir Aahs",
            53: "Voice Oohs",
            54: "Synth Voice",
            55: "Orchestra Hit",
            56: "Trumpet",
            57: "Trombone",
            58: "Tuba",
            59: "Muted Trumpet",
            60: "French Horn",
            61: "Brass Section",
            62: "SynthBrass 1",
            63: "SynthBrass 2",
            64: "Soprano Sax",
            65: "Alto Sax",
            66: "Tenor Sax",
            67: "Baritone Sax",
            68: "Oboe",
            69: "English Horn",
            70: "Bassoon",
            71: "Clarinet",
            72: "Piccolo",
            73: "Flute",
            74: "Recorder",
            75: "Pan Flute",
            76: "Blown Bottle",
            77: "Shakuhachi",
            78: "Whistle",
            79: "Ocarina",
            80: "Lead 1 (square)",
            81: "Lead 2 (sawtooth)",
            82: "Lead 3 (calliope)",
            83: "Lead 4 (chiff)",
            84: "Lead 5 (charang)",
            85: "Lead 6 (voice)",
            86: "Lead 7 (fifths)",
            87: "Lead 8 (bass + lead)",
            88: "Pad 1 (new age)",
            89: "Pad 2 (warm)",
            90: "Pad 3 (polysynth)",
            91: "Pad 4 (choir)",
            92: "Pad 5 (bowed)",
            93: "Pad 6 (metallic)",
            94: "Pad 7 (halo)",
            95: "Pad 8 (sweep)",
            96: "FX 1 (rain)",
            97: "FX 2 (soundtrack)",
            98: "FX 3 (crystal)",
            99: "FX 4 (atmosphere)",
            100:"FX 5 (brightness)",
            101:"FX 6 (goblins)",
            102:"FX 7 (echoes)",
            103:"FX 8 (sci-fi)",
            104:"Sitar",
            105:"Banjo",
            106:"Shamisen",
            107:"Koto",
            108:"Kalimba",
            109:"Bagpipe",
            110:"Fiddle",
            111:"Shanai",
            112:"Tinkle Bell",
            113:"Agogo",
            114:"Steel Drums",
            115:"Woodblock",
            116:"Taiko Drum",
            117:"Melodic Tom",
            118:"Synth Drum",
            119:"Reverse Cymbal",
            120:"Guitar Fret Noise",
            121:"Breath Noise",
            122:"Seashore",
            123:"Bird Tweet",
            124:"Telephone Ring",
            125:"Helicopter",
            126:"Applause",
            127:"Gunshot",
        }

    def apply_settings(self):
        """Apply the settings from the UI to the audio processing pipeline."""
        self.TUNING = float(self.tuning_input.text())
        self.MIN_FREQ = float(self.fmin_input.text())
        self.MAX_FREQ = float(self.fmax_input.text())
        self.INSTRUMENT = self.instrument_selector.currentText()
        self.TEMPO = float(self.midi_tempo.value())

        # construct a dict with all settings to emit
        settings = {
            "tuning": self.TUNING,
            "fmin": self.MIN_FREQ,
            "fmax": self.MAX_FREQ,
            "instrument": self.INSTRUMENT,
            "tempo": self.TEMPO,
            "user_playback": self.user_checkbox.isChecked(),
            "active_channels": self.active_channels
        }
        self.settings_applied.emit(settings)
        print(f"Settings applied: {settings}")

    def init_recording_settings(self):
        # --- RECORDING SETTINGS ---
        self.recording_title = QLabel("Recording Settings")
        self.recording_title.setFont(self.textbf)
        self._layout.addWidget(self.recording_title)

        # instrument selection
        self._layout.addWidget(QLabel("Select instrument"))
        self.instrument_selector = QComboBox()
        self.instrument_selector.addItems(self.INSTRUMENTS)
        center_combo_items(self.instrument_selector)
        self.instrument_selector.currentTextChanged.connect(self.instrument_selected)
        self._layout.addWidget(self.instrument_selector)

        # fmin/fmax settings
        self.fmin_layout = QHBoxLayout()
        self.fmin_label = QLabel("Min freq: ")
        self.fmin_input = QLineEdit()
        self.fmin_input.setText(str(self.MIN_FREQ))

        self.fmin_layout.addWidget(self.fmin_label)
        self.fmin_layout.addWidget(self.fmin_input)
        # self.fmin_input.textChanged.connect(lambda value: self.pyin_setting_changed(value, type="fmin"))
        self._layout.addLayout(self.fmin_layout)

        self.fmax_layout = QHBoxLayout()
        self.fmax_label = QLabel("Max freq: ")
        self.fmax_input = QLineEdit()
        # self.fmax_input.textChanged.connect(lambda value: self.pyin_setting_changed(value, type="fmax"))
        self.fmax_input.setText(str(self.MAX_FREQ))

        self.fmax_layout.addWidget(self.fmax_label)
        self.fmax_layout.addWidget(self.fmax_input)
        self._layout.addLayout(self.fmax_layout)

        # tuning
        self.tuning_layout = QHBoxLayout()
        self.tuning_label = QLabel("Tuning: ")
        self.tuning_input = QLineEdit()
        self.tuning_input.setText(str(self.TUNING))
        # self.tuning_input.textChanged.connect(lambda value: self.pyin_setting_changed(value, type="tuning"))

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
        self.midi_tempo.setRange(20, 400)

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

        for channel, program in midi_data.get_programs().items():
            checkbox = QCheckBox(f"{self.GM_PROGRAM_MAP.get(program, program)}")
            checkbox.setCheckState(Qt.CheckState.Checked)
            checkbox.stateChanged.connect(self.channels_selected)
            self.midi_checkboxes[channel] = checkbox
            self.playback_layout.addWidget(checkbox)

    def user_checkbox_selected(self, s: bool):
        """called whenever the user checkbox state changes
        Args:
            s: a boolean whether it's checked or not
        """
        # self.user_channel_active.emit(s)
        print(f"user_is_playing changed: {s}")

    def channels_selected(self, s):
        """emits a dictionary of active channels
        with the program and whether it's active or not
        """
        self.active_channels = []
        for channel, midi_checkbox in self.midi_checkboxes.items():
            if midi_checkbox.isChecked():
                self.active_channels.append(channel)
        print(f"active midi channels changed: {self.active_channels}")
        # self.midi_channels_active.emit(self.active_channels)

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


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(300)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self.settings_panel = SettingsPanel()
        self._layout.addWidget(self.settings_panel)