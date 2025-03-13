import os
import sys
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QAction, QColor, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLineEdit, QTabWidget, QVBoxLayout, QHBoxLayout, QToolBar, QStatusBar, QSplitter, QLabel, QComboBox, QDoubleSpinBox, QFileDialog, QSlider, QPushButton
import qdarktheme
import pyqtgraph as pg
import mido
from math import ceil

from app.core.midi.MidiData import MidiData
# from app.core.audio.AudioData import AudioData
from app.core.recording.Recording import Recording
from app.core.recording.Pitch import Pitch

class SettingsPanel(QWidget):
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
        self._layout.addLayout(self.fmin_layout)

        self.fmax_layout = QHBoxLayout()
        self.fmax_label = QLabel("Max freq: ")
        self.fmax_input = QLineEdit()
        self.fmax_input.setText(str(self.MAX_FREQ))

        self.fmax_layout.addWidget(self.fmax_label)
        self.fmax_layout.addWidget(self.fmax_input)
        self._layout.addLayout(self.fmax_layout)

        # tuning
        self.tuning_layout = QHBoxLayout()
        self.tuning_label = QLabel("Tuning: ")
        self.tuning_input = QLineEdit()
        self.tuning_input.setText(str(self.TUNING))

        self.tuning_layout.addWidget(self.tuning_label)
        self.tuning_layout.addWidget(self.tuning_input)
        self._layout.addLayout(self.tuning_layout)

        # --- MIDI SETTINGS ---
        self.midi_title = QLabel("MIDI Settings")
        self.midi_title.setFont(self.textbf)
        self._layout.addWidget(self.midi_title)

        # tempo
        self.midi_tempo_layout = QHBoxLayout()
        self.midi_tempo = QDoubleSpinBox()

        self.midi_tempo.setValue(self.TEMPO)
        self.midi_tempo_label = QLabel("Tempo: ")

        self.midi_tempo_layout.addWidget(self.midi_tempo_label)
        self.midi_tempo_layout.addWidget(self.midi_tempo)
        self._layout.addLayout(self.midi_tempo_layout)

        # instrument selection
        # let's do this later.....
        
        self._layout.addStretch() # ensures things don't expand to fit entire vertical

    def instrument_selected(self, text):
        fmin, fmax = self.INSTRUMENT_RANGES[text]
        self.MIN_FREQ = fmin
        self.MAX_FREQ = fmax
        # update labels for user
        self.fmin_input.setText(str(self.MIN_FREQ))
        self.fmax_input.setText(str(self.MAX_FREQ))


class PitchPlot(QWidget):
    def __init__(self):
        super().__init__()

        self.time = 0
        self.x_range = (-1, 4)
        self.y_range = (40, 90)

        self.colors = {
            'background': '#1D1D1D',    # dark gray
            'label_text': '#A4A4A4',    # black
            'timeline': '#FF0000',      # red
            'pitch': '#F57C9C'          # pink
        }

        self.midi_config = {
            'normal_color': '#53555C',  # Grey
            'played_color': '#3A3B40',  # Darker grey
            'bar_height': 1
        }

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # init the pyqtgraph plot
        self.plot = pg.PlotWidget()
        self._layout.addWidget(self.plot)
        
        self.plot.setBackground(self.colors['background'])

        # margins (to make room around labels)
        self.plot.getPlotItem().layout.setContentsMargins(10, 10, 10, 15) 
        
        # label tick colors
        self.plot.getAxis('bottom').setPen(self.colors['label_text'])
        self.plot.getAxis('left').setPen(self.colors['label_text'])
        self.plot.getAxis('bottom').setTextPen(self.colors['label_text'])
        self.plot.getAxis('left').setTextPen(self.colors['label_text'])

        # x/y axis labels
        self.plot.setLabel('bottom', 'Time (s)', color=self.colors['label_text'])  
        self.plot.setLabel('left', 'Pitch (MIDI #)', color=self.colors['label_text'])

        self.plot.setXRange(self.x_range[0], self.x_range[1])
        self.plot.setYRange(self.y_range[0], self.y_range[1])

        # add our timeline
        self.plot_timeline(self.time)

        # plot items
        self.midi_notes = []


    def plot_timeline(self, time: float):
        """plots a thin red line at the specified time"""
        # add the TIMELINE
        self.timeline = pg.InfiniteLine(
            pos=time,
            angle=90,
            pen={'color': self.colors['timeline'], 'width': 2})
        self.plot.addItem(self.timeline)

    def plot_midi(self, midi_data: MidiData):
        """plots the given midi data as notes on the pitchplot"""

        # clear other midi notes
        for midi_note in self.midi_notes:
            self.plot.removeItem(midi_note)
        self.midi_notes = []

        # plot the new ones!
        for _, midi_note in midi_data.pitch_df.iterrows():

            start, duration, pitch = (midi_note['start'], midi_note['duration'], midi_note['pitch'])
            note_x = start + (duration / 2)
            # display colors behind the timeline as darker
            color = self.midi_config['normal_color'] if note_x >= self.time else self.midi_config['played_color']

            midi_note = pg.BarGraphItem(
                x=note_x,  
                y=pitch,
                height=self.midi_config['bar_height'],
                width=duration,
                brush=color,
                pen=None,
                name='MIDI')
            self.midi_notes.append(midi_note)
            self.plot.addItem(midi_note)

            # print(f'plotted {midi_note}: pitch={pitch}')

    def plot_pitches(self, pitches: list[Pitch]):
        """
        plots all pitches, from a list of pitches, all at once
            => not for piecemeal pitch plotting since we erase the board 
               every time before running
        """
        print("Plotting pitches...")
        # base_color = QColor(self.colors['pitch'])  # pink

        # normalize volumes (min-max normalization)
        volumes = [pitch.volume for pitch in pitches]
        min_vol = min(volumes)
        max_vol = max(volumes)
        
        # avoid division by zero in case all volumes are the same
        if max_vol == min_vol:
            normalized_volumes = [0.5] * len(pitches)
        else:
            normalized_volumes = [(v-min_vol) / (max_vol-min_vol) for v in volumes]

        brushes = []
        for i, pitch in enumerate(pitches):
            volume = normalized_volumes[i]

            # convert volume to hue:
            # 0 -> blue (HSV hue ~240), 0.5 -> yellow (HSV hue ~60) 1 -> red (HSV hue ~0)
            if volume < 0.5:
                hue = 240 - (240-180) * (volume/0.5)  # blue to yellow
            else:
                hue = 180 - (180 - 0) * ((volume-0.5) / 0.5)  # yellow to red

            # QColor uses HSV (hue, saturation, value)
            color = QColor.fromHsv(int(hue), 255, 255) 

            # set opacity based on pitch probability
            color.setAlphaF(1)  # opacity (alpha) based on pitch probability (0 to 1)
            brushes.append(pg.mkBrush(color))  # Use mkBrush to create the brush with color

        self.pitches = pg.ScatterPlotItem(
            x=[pitch.time for pitch in pitches],
            y=[pitch.midi_num for pitch in pitches],
            size=5,
            pen=None,
            brush=brushes,
            name='Pitch'
        )
        self.plot.addItem(self.pitches)
        print("Done!")

class Slider(QWidget):
    
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # slider <==> timer resolution variables
        self.TICKS_PER_SEC = 10
        self.N_TICKS_PER_CALLBACK = 1

        # current time variables
        self.tick = 0

        # init our slider!!
        self.DEFAULT_LENGTH_SEC = 30
        self.slider = QSlider(Qt.Orientation.Horizontal)
        midi_length_ticks = int(self.DEFAULT_LENGTH_SEC*self.TICKS_PER_SEC)
        self.slider.setRange(0, midi_length_ticks)

        self._layout.addWidget(self.slider)

        # slider emissions

    def update_slider_range(self, sec: float):
        """update the slider range to have [sec] amount of space"""
        ticks = int(ceil(sec * self.TICKS_PER_SEC))
        self.slider.setRange(0, ticks)


class RecordingPanel(QWidget):
    def __init__(self):
        # everything needed to record audio
        super().__init__()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._layout.addWidget(QLabel("Record"))
        self._layout.addStretch()

        self.pitch_plot = PitchPlot()
        self._layout.addWidget(self.pitch_plot)

        # the playback layout
        self.slider_layout = QHBoxLayout()

        # get the play/pause button icons
        app_directory = os.path.dirname(__file__) # directory of current file (which is root)
        play_filepath = os.path.join(app_directory, 'app', 'resources', 'icons', 'play.png')
        pause_filepath = os.path.join(app_directory, 'app', 'resources', 'icons', 'pause.png')
        record_filepath = os.path.join(app_directory, 'app', 'resources', 'icons', 'record.png')
        self.play_icon = QIcon(play_filepath)
        self.pause_icon = QIcon(pause_filepath)
        self.record_icon = QIcon(record_filepath)

        # add play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(QSize(26, 26))
        self.slider_layout.addWidget(self.play_button)

        # add record button
        self.record_button = QPushButton()
        self.record_button.setIcon(self.record_icon)
        self.record_button.setFixedSize(QSize(26, 26))
        self.slider_layout.addWidget(self.record_button)

        # init the slider!
        self.slider = Slider()
        self.slider_layout.addWidget(self.slider)
        self._layout.addLayout(self.slider_layout)
        
        self._layout.addStretch()

        # recording variables
        self.is_recording: bool = False

    def toggle_playback(self):
        """toggle the playback of whatever audio we want on/off"""

        


class RecordTab(QWidget):
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)

        # split the tab into settings / recording
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self._layout.addWidget(self.splitter)

        # settings panel
        self.settings_panel = SettingsPanel()
        
        # recording panel
        self.recording_panel = RecordingPanel()

        self.splitter.addWidget(self.settings_panel)
        self.splitter.addWidget(self.recording_panel)
        self.splitter.setSizes([200, 700])

    def update_midi(self, midi_data: MidiData):
        self.recording_panel.pitch_plot.plot_midi(midi_data)

    def update_pitches(self, recording: Recording):
        pitches = recording.pitches.get_pitches()
        self.recording_panel.pitch_plot.plot_pitches(pitches)


class AnalyzeTab(QWidget):
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)


class Toolbar(QToolBar):
    """
    Supports the following actions:
        1, uploading midi
        2. uploading audio
        3. adjusting recording settings
    """
    midi_uploaded = pyqtSignal(str)
    audio_uploaded = pyqtSignal(str)

    def __init__(self):
        super().__init__() # init the QToolBar it inherits from
        self.setOrientation(Qt.Orientation.Horizontal)

        self.buttons = {}

        self.buttons["Upload MIDI"] = QAction("Upload MIDI", self)
        self.buttons["Upload MIDI"].setStatusTip("Upload a MIDI or XML file")
        self.buttons["Upload MIDI"].triggered.connect(self.upload_midi)

        self.buttons["Upload audio"] = QAction("Upload audio", self)
        self.buttons["Upload audio"].setStatusTip("Upload an audio file")
        self.buttons["Upload audio"].triggered.connect(self.upload_audio)

        for b in self.buttons.values():
            self.addAction(b)
    
    def upload_midi(self):
        print("uploading midi")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI or XML File", "", "All Files (*)"

        )
        if file_path:
            print(f"Selected MIDI/XML file: {file_path}")
            self.midi_uploaded.emit(file_path)

    def upload_audio(self):
        print("uploading audio")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "All Files (*)"
        )
        if file_path:
            print(f"Selected audio file: {file_path}")
            self.audio_uploaded.emit(file_path)



class Synchrony(QMainWindow):
    def __init__(self):
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
        self.record_tab = RecordTab()
        self.tab_manager.addTab(self.record_tab, "Record")
        self.analyze_tab = AnalyzeTab()
        self.tab_manager.addTab(self.analyze_tab, "Analyze")

        # toolbar / status bar
        self.setStatusBar(QStatusBar(self))
        self.toolbar = Toolbar()
        self.addToolBar(self.toolbar)
        self.toolbar.midi_uploaded.connect(self.midi_uploaded)
        self.toolbar.audio_uploaded.connect(self.audio_uploaded)

        # variables
        self.recording: Recording = Recording()

    def midi_uploaded(self, midi_filepath: str):
        # create a new midi_data based on the received midi filepath signal
        self.recording.load_midi(midi_filepath)
        self.record_tab.update_midi(self.recording.midi_data)

    def audio_uploaded(self, audio_filepath: str):
        # create a new audio_data based on the received audio filepath signal
        # uses current settings_panel's sample rate for sampling resolution
        self.recording.load_audio(
            audio_filepath=audio_filepath, 
            sr=self.record_tab.settings_panel.SAMPLE_RATE
        )
        # compute + plot pitches!
        self.recording.detect_pitches()
        self.record_tab.update_pitches(self.recording)
        


if __name__ == '__main__':
    # app configs
    APP_NAME: str = "Synchrony"
    DARK_THEME: bool = True

    app = QApplication(sys.argv)
    main_window = Synchrony()
    main_window.show()
    sys.exit(app.exec())
