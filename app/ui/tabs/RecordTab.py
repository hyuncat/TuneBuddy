import os
from math import ceil
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSplitter
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSize


# App config
from app.config import AppConfig
# Audio modules
from app.core.audio.AudioData import AudioData
from app.core.audio.AudioRecorder import AudioRecorder
from app.core.audio.AudioPlayer import AudioPlayer
# MIDI modules
from app.core.midi.MidiData import MidiData
from app.core.midi.MidiSynth import MidiSynth
from app.core.midi.MidiPlayer import MidiPlayer
from notebooks.archive.PitchDf import PitchDf
# Algorithm modules
from app.algorithms.align.DTW import DTW
from app.algorithms.align.OnsetDf import UserOnsetDf
from archive.PYin import PYin
from app.core.recording.Pitch import PitchConfig
# UI
from app.ui.widgets.Slider import Slider
from app.ui.plots.PitchPlot import PitchPlot

class RecordTab(QWidget):
    """Tab for handling initial audio recording/playback"""
    def __init__(self):
        super().__init__()
        # main layout for tab
        self._layout = QHBoxLayout(self)
        self.setLayout(self._layout)

        # make side panel collapsible
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False) # prevent both widgets from being collapsed at once
        self._layout.addWidget(self._splitter)

        # side panel ui
        self._side_panel = QWidget()
        self._side_panel_layout = QVBoxLayout()
        self._side_panel.setLayout(self._side_panel_layout)
        self._side_panel_layout.addWidget(QLabel("Settings"))
        self._side_panel_layout.addStretch()
        
        # central panel ui
        self._central_panel = QWidget()
        self._central_panel_layout = QVBoxLayout()
        self._central_panel.setLayout(self._central_panel_layout)

        self._splitter.addWidget(self._side_panel)
        self._splitter.addWidget(self._central_panel)
        self._splitter.setSizes([200, 700])

        # Recording/playback variables
        self.is_recording = False
        self.is_midi_playing = False
        self.is_user_playing = False

        # Initialize the MIDI/user audio data
        # (Can be omitted later to allow custom MIDI uploads and
        # user audio recording within the app itself)
        # ---
        self.MIDI_FILE = "aligned_ultra_sally.mid" # resources/midi/...
        self.SOUNDFONT_FILE = "MuseScore_General.sf3" # resources/...
        self.USER_AUDIO_FILE = "ultra_sally.mp3" # resources/audio/...

        self.app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        self.init_midi(self.MIDI_FILE, self.SOUNDFONT_FILE)
        self.init_user_audio(self.USER_AUDIO_FILE)
        self.init_pitch_plot()
        self.init_slider()
        self.init_playback_buttons()
    
    def init_midi(self, midi_file: str, soundfont_file: str):
        """
        Initializes MIDI data, synth, and player for use in the app.
        Expects the file name as input and parses filepath as the app/resources 
        folder automatically.

        Args:
            midi_file (str): MIDI file to load
            soundfont_file (str): Soundfont file to load
        """
        print(f"Starting app with MIDI file {midi_file}...")

        # Get MIDI/soundfont file paths
        soundfont_filepath = os.path.join(self.app_directory, 'resources', soundfont_file)
        midi_filepath = os.path.join(self.app_directory, 'resources', 'midi', midi_file)

        # Initialize MIDI data, synth, and player
        self._MidiData = MidiData(midi_filepath)
        self._MidiSynth = MidiSynth(soundfont_filepath)
        self._MidiPlayer = MidiPlayer(self._MidiSynth)
        self._MidiPlayer.load_midi(self._MidiData) # Load MIDI data into the player

    def init_user_audio(self, user_audio_file: str) -> None:
        """
        Preload the app with an audio recording to perform analysis.
        """
        # Get audio file path
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        audio_filepath = os.path.join(app_directory, 'resources', 'audio', user_audio_file)

        # Initialize audio data, recorder, and player
        # self._AudioRecorder = AudioRecorder()
        
        self._AudioData = AudioData()
        self._AudioData.load_data(audio_filepath)

        self._AudioPlayer = AudioPlayer()
        self._AudioPlayer.load_audio_data(self._AudioData)
        print(f"Preloaded user audio: {user_audio_file}")

        self.pitches, self.best_prob_pitches = PYin.pyin(self._AudioData.data, mean_threshold=0.3)

        pitch_config = PitchConfig( # Defines resolution of pitch bins
            bins_per_semitone=10, tuning=440.0, fmin=196, fmax=5000
        )
        self.pitch_df = PitchDf(self._AudioData, pitch_config, self.pitches)
        self.onset_df = UserOnsetDf(self._AudioData, pitch_list=self.best_prob_pitches, pitch_df=self.pitch_df)
    
    def init_pitch_plot(self):
        self._PitchPlot = PitchPlot()
        self._PitchPlot.plot_midi(self._MidiData)

        # Plot these if user preloads in data
        self._PitchPlot.plot_onsets(self.onset_df.onset_df)
        self._PitchPlot.plot_pitches(self.best_prob_pitches)
        self._central_panel_layout.addWidget(self._PitchPlot)

    def init_slider(self):
        self._Slider = Slider(self._MidiData) # Init slider with current MIDI data
        self._Slider.slider_changed.connect(self.handle_slider_change)

        # Update the slider max value if the audio file is longer than the MIDI file
        if hasattr(self, "_AudioData"):
            audio_length = ceil(self._AudioData.get_length())
            if audio_length > self._MidiData.get_length():
                slider_ticks = audio_length * self._Slider.TICKS_PER_SEC
                self._Slider.update_slider_max(slider_ticks)
        
        self._central_panel_layout.addWidget(self._Slider)
    
    def init_playback_buttons(self):
        """Init playback/record buttons"""
        # Create a horizontal layout for the playback and recording buttons
        self.buttonLayout = QHBoxLayout()

        # Add togglePlay and toggleRecord buttons
        play_filepath = os.path.join(self.app_directory, 'resources', 'icons', 'play.png')
        pause_filepath = os.path.join(self.app_directory, 'resources', 'icons', 'pause.png')
        self.play_icon = QIcon(play_filepath)
        self.pause_icon = QIcon(pause_filepath)

        self.midi_play_button = QPushButton()
        self.midi_play_button.setIcon(self.play_icon)
        self.midi_play_button.setFixedSize(self.play_icon.actualSize(QSize(26, 26)))

        self.midi_play_button.clicked.connect(self.toggle_midi)
        self.buttonLayout.addWidget(self.midi_play_button)

        # self.record_button = QPushButton('Start recording')
        # self.record_button.clicked.connect(self.toggle_record)
        # self.buttonLayout.addWidget(self.record_button)

        # Listen back to audio
        self.user_play_button = QPushButton()
        self.user_play_button.setIcon(self.play_icon)
        self.user_play_button.setFixedSize(self.play_icon.actualSize(QSize(26, 26)))

        self.user_play_button.clicked.connect(self.toggle_user_playback)
        self.buttonLayout.addWidget(self.user_play_button)

        # DTW button
        self.dtw_button = QPushButton('Align')
        self.dtw_button.clicked.connect(self.dtw_align)
        self.buttonLayout.addWidget(self.dtw_button)

        self._side_panel_layout.addLayout(self.buttonLayout)

    def toggle_midi(self):
        """Toggle the MIDI playback on and off."""
        #TODO: Make slider not stop if at least one playback is running
        if self.is_midi_playing: # Pause timer if playing
            self._Slider.stop_timer()
            self.is_midi_playing = False
            self.midi_play_button.setIcon(self.play_icon)
            self._MidiPlayer.pause()
        else: # Unpause timer if not playing
            self._Slider.start_timer()
            self.is_midi_playing = True
            self.midi_play_button.setIcon(self.pause_icon)
            start_time = self._Slider.get_current_time()
            self._MidiPlayer.play(start_time=start_time)

    def toggle_user_playback(self):
        """Toggle user's recorded audio playback on/off"""
        if not hasattr(self, "_AudioPlayer"):
            return
        
        if self.is_user_playing:
            self._Slider.stop_timer()
            self.user_play_button.setIcon(self.play_icon)
            self._AudioPlayer.pause()
            self.is_user_playing = False
        else:
            self._Slider.start_timer()
            self.user_play_button.setIcon(self.pause_icon)
            start_time = self._Slider.get_current_time()
            self._AudioPlayer.play(start_time=start_time)
            self.is_user_playing = True

    def handle_slider_change(self, value):
        """Handle the slider change event, e.g., seeking in a MIDI playback"""
        #TODO: Handle seeking during audio/midi playback
        current_time = self._Slider.get_current_time() 
        self._PitchPlot.move_plot(current_time)

        if value < self._Slider.slider.minimum():
            self._Slider.slider.setValue(self._Slider.slider.minimum())

        if value >= self._Slider.slider.maximum():
            self._Slider.stop_timer()
            # Pause MIDI playback if it's currently playing
            self.is_midi_playing = False
            self.midi_play_button.setIcon(self.play_icon)
            
            if hasattr(self, "_MidiPlayer"):
                self._MidiPlayer.pause()

            # Pause user audio playback if it's currently playing
            self.is_user_playing = False
            self.user_play_button.setText('Play Recorded Audio')
            if hasattr(self, "_AudioPlayer"):
                self._AudioPlayer.pause()

    def change_midi_tempo(self, bpm: float=None, target_length: float=None):
        """
        Change the midi tempo to either be a target length or a target bpm.
        Also re-plots the midi file below the user.
        """



    def dtw_align(self):
        print("Align notes! (Implemented later)")
        # do we have audio data of both?
        user_audio = self._AudioData
        midi_data = self._MidiData

        # Change tempo of midi data to be same length as user audio
        midi_data.change_tempo(target_length=user_audio.get_length())
        midi_data.save_to_file() # save the tempo changed midi to the file
