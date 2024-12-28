from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QLineEdit
import os
from essentia.standard import MonoLoader
from math import ceil

from app.config import AppConfig

# Audio recording/playback modules
from app.core.audio.AudioData import AudioData
from app.core.audio.AudioRecorder import AudioRecorder
from app.core.audio.AudioPlayer import AudioPlayer
from archive.PitchAnalyzer import PitchAnalyzer

# MIDI handling / playback modules
from app.core.midi.MidiData import MidiData
from app.core.midi.MidiSynth import MidiSynth
from app.core.midi.MidiPlayer import MidiPlayer

# UI
from app.ui.widgets.Slider import Slider
from archive.PitchPlot import PitchPlot

# DTW
from app.dtw.PitchDTW import PitchDTW


class RecordTab(QWidget):
    def __init__(self):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)

        # Recording/playback variables
        self.is_recording = False
        self.is_midi_playing = False
        self.is_user_playing = False
        self.min_confidence = 0.0

        # Initialize the UI
        self.MIDI_FILE = "fugue.mid"
        self.SOUNDFONT_FILE = "MuseScore_General.sf3"
        self.USER_AUDIO_FILE = "user_fugue2.mp3"
        self.init_midi(self.MIDI_FILE, self.SOUNDFONT_FILE)
        self.init_user_audio(self.USER_AUDIO_FILE)
        self.init_ui()

    
    def init_midi(self, midi_file: str, soundfont_file: str) -> None:
        """Initialize the MIDI data, synth, and player for use in the app."""

        # Get MIDI/soundfont file paths
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        soundfont_filepath = os.path.join(app_directory, 'resources', soundfont_file)
        midi_filepath = os.path.join(app_directory, 'resources', 'midi', midi_file)

        # Initialize MIDI data, synth, and player
        self._MidiData = MidiData(midi_filepath)
        self._MidiSynth = MidiSynth(soundfont_filepath)
        self._MidiPlayer = MidiPlayer(self._MidiSynth)
        self._MidiPlayer.load_midi(self._MidiData) # Load MIDI data into the player

        # Also store the MIDI audio data for DTW
        self.midi_audio = AudioData()
        self.midi_audio.load_midi_file(midi_filepath, soundfont_filepath)

    def init_user_audio(self, user_audio_file: str) -> None:
        # Get audio file path
        app_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        audio_filepath = os.path.join(app_directory, 'resources', 'audio', user_audio_file)

        # Initialize audio data, recorder, and player
        # self._AudioRecorder = AudioRecorder()
        print(f"Starting app with {audio_filepath}...")
        
        self._AudioData = AudioData()
        self._AudioData.load_data(audio_filepath)

        self._AudioPlayer = AudioPlayer()
        self._AudioPlayer.load_audio_data(self._AudioData)

        # Load the audio file as mono
        # self.user_pitches, self.user_pitch_confidences, self.pitch_times = PitchAnalyzer().get_buffer_pitch(self._AudioData)
        self._PitchAnalyzer = PitchAnalyzer()
        self.user_onsets = self._PitchAnalyzer.detect_onsets(self._AudioData)
        self.user_pitchdf = self._PitchAnalyzer.user_pitchdf(self._AudioData)

        # Compute warping
        self.midi_features = PitchDTW.midi_features(self._MidiData, self.midi_audio)
        self.user_features = PitchDTW.user_features(self._AudioData)

        self.alignment = PitchDTW.align(self.user_features, self.midi_features)
        self.align_df = PitchDTW.align_df(self.alignment, self.user_features, self.midi_features)
        self.align_df2 = PitchDTW.align_df2(self.alignment, self.user_features, self.midi_features)
        
    def init_ui(self):
        # Add recording slider
        self._Slider = Slider(self._MidiData) # Init slider with current MIDI data
        self._Slider.slider_changed.connect(self.handle_slider_change)

        # Update the slider max value if the audio file is longer than the MIDI file
        audio_length = ceil(self._AudioData.capacity / AppConfig.SAMPLE_RATE)
        if audio_length > self._MidiData.get_length():
            slider_ticks = audio_length * self._Slider.TICKS_PER_SEC
            self._Slider.update_slider_max(slider_ticks)

        # Add pitch plot
        self._PitchPlot = PitchPlot()
        self._PitchPlot.plot_midi(self._MidiData)
        self._PitchPlot.plot_user(self.user_pitchdf, self.user_onsets, 0)
        self._PitchPlot.plot_alignment(self.user_pitchdf, self.align_df, self.align_df2)
        self._layout.addWidget(self._PitchPlot)

        self._layout.addWidget(self._Slider)
        self.init_buttons()
        self.init_confidence_input()


    def init_confidence_input(self):
        """Initialize the confidence threshold input"""
        self.plotControlLayout = QHBoxLayout()

        # Min confidence for displaying pitches
        self.confidence_label = QLabel("Min Confidence:")
        self.confidence_input = QLineEdit()
        self.confidence_input.setText(str(self.min_confidence))

        # Window size for note detection
        self.window_size_label = QLabel("Window Size:")
        self.window_size_input = QLineEdit()
        self.window_size_input.setText("15")

        # Threshold for note detection
        self.note_threshold_label = QLabel("Note Threshold:")
        self.note_threshold_input = QLineEdit()
        self.note_threshold_input.setText("0.4")

        # Threshold for harmonic grouping
        self.harmonic_range_label = QLabel("Harmonic Range:")
        self.harmonic_range_input = QLineEdit()
        self.harmonic_range_input.setText("0.75")
        
        # Create a submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.update_pitch_plot)
        
        # Add the widgets to the layout
        self.plotControlLayout.addWidget(self.confidence_label)
        self.plotControlLayout.addWidget(self.confidence_input)
        self.plotControlLayout.addWidget(self.window_size_label)
        self.plotControlLayout.addWidget(self.window_size_input)
        self.plotControlLayout.addWidget(self.note_threshold_label)
        self.plotControlLayout.addWidget(self.note_threshold_input)
        self.plotControlLayout.addWidget(self.harmonic_range_label)
        self.plotControlLayout.addWidget(self.harmonic_range_input)
        self.plotControlLayout.addWidget(self.submit_button)

        self._layout.addLayout(self.plotControlLayout)

    def update_pitch_plot(self):
        """Update the confidence threshold and re-plot user pitches"""
        try:
            min_confidence = float(self.confidence_input.text())
            window_size = int(self.window_size_input.text())
            note_threshold = float(self.note_threshold_input.text())
            harmonic_range = float(self.harmonic_range_input.text())

            self.user_features.resegment_notes(window_size, note_threshold)
            self.alignment = PitchDTW.align(self.user_features, self.midi_features)
            self.align_df = PitchDTW.align_df(self.alignment, self.user_features, self.midi_features)
            self.align_df2 = PitchDTW.align_df2(self.alignment, self.user_features, self.midi_features)

            # Re-plot user pitches with new note estimation, harmonic grouping, confidence parameters
            print("Updating pitch plot with new parameters...\n---")
            self._PitchPlot.plot_user(
                user_pitchdf=self.user_pitchdf, 
                onsets=self.user_features.onset_times, 
                min_confidence=min_confidence,
                window_size=window_size, 
                note_threshold=note_threshold, 
                harmonic_range=harmonic_range
            )
            self._PitchPlot.plot_alignment(self.user_pitchdf, self.align_df, self.align_df2)
            print("Done!")
            
        except ValueError:
            # Print error and don't update plot
            print("Invalid input - failed to parse input as float or int.")
        

    def init_buttons(self):
        """Init playback/record buttons"""
        # Create a horizontal layout for the playback and recording buttons
        self.buttonLayout = QHBoxLayout()

        # Add togglePlay and toggleRecord buttons
        self.midi_play_button = QPushButton('Play MIDI')
        self.midi_play_button.clicked.connect(self.toggle_midi)
        self.buttonLayout.addWidget(self.midi_play_button)

        # self.record_button = QPushButton('Start recording')
        # self.record_button.clicked.connect(self.toggle_record)
        # self.buttonLayout.addWidget(self.record_button)

        # Listen back to audio
        self.user_play_button = QPushButton('Play Recorded Audio')
        self.user_play_button.clicked.connect(self.toggle_user_playback)
        self.buttonLayout.addWidget(self.user_play_button)

        self._layout.addLayout(self.buttonLayout)


    def toggle_midi(self):
        """Toggle the MIDI playback on and off."""
        #TODO: Make slider not stop if at least one playback is running
        if self.is_midi_playing: # Pause timer if playing
            self._Slider.stop_timer()
            self.is_midi_playing = False
            self.midi_play_button.setText('Play MIDI')
            self._MidiPlayer.pause()
        else: # Unpause timer if not playing
            self._Slider.start_timer()
            self.is_midi_playing = True
            self.midi_play_button.setText('Pause MIDI')
            start_time = self._Slider.get_current_time()
            # Check if MidiPlayer exists
            print("does MidiPlayer exist?", self._MidiPlayer)
            self._MidiPlayer.play(start_time=start_time)

    def toggle_user_playback(self):
        if self.is_user_playing:
            self._Slider.stop_timer()
            self.user_play_button.setText('Play Recorded Audio')
            self._AudioPlayer.pause()
            self.is_user_playing = False
        else:
            self._Slider.start_timer()
            self.user_play_button.setText('Pause Recorded Audio')
            start_time = self._Slider.get_current_time()
            self._AudioPlayer.play(start_time=start_time)
            self.is_user_playing = True


    def handle_slider_change(self, value):
        """Handle the slider change event, e.g., seeking in a MIDI playback"""
        current_time = self._Slider.get_current_time() 
        self._PitchPlot.move_plot(current_time)

        if value == self._Slider.slider.maximum():
            self._Slider.stop_timer()
            # Pause MIDI playback if it's currently playing
            self.is_midi_playing = False
            self.midi_play_button.setText('Play MIDI')
            self._MidiPlayer.pause()

            # Pause user audio playback if it's currently playing
            self.is_user_playing = False
            self.user_play_button.setText('Play Recorded Audio')
            self._AudioPlayer.pause()

        # print(f"Slider value changed to: {value}")

    def status_message(self):
        return "Recording audio..."