import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton
)
import logging

# ui imports
from ui.SettingsPanel import SettingsPanel
from ui.Slider import Slider
from ui.PitchPlot import PitchPlot

# app logic imports
from app_logic.user.ds.UserData import UserData
from app_logic.user.AudioRecorder import AudioRecorder
from app_logic.user.AudioPlayer import AudioPlayer
from app_logic.midi.MidiData import MidiData
from app_logic.midi.MidiPlayer import MidiPlayer

from algorithms.PitchDetector import PitchDetector
class RecordTab(QWidget):
    """ interface / handles ui for the recording tab
    
    contains two panels:
        - settings_panel
        - recording_panel

    also handles cross-talk between settings and recording / playback
    """
    def __init__(self, user_data: UserData, midi_data: MidiData, audio_player: AudioPlayer, audio_recorder: AudioRecorder, midi_player: MidiPlayer, pitch_detector: PitchDetector):
        super().__init__()
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # important references
        self.user_data = user_data
        self.midi_data = midi_data
        self.midi_player = midi_player
        self.audio_player = audio_player
        self.audio_recorder = audio_recorder
        self.pitch_detector = pitch_detector

        # --- UI INITIALIZATIONS ---
        # split the tab into settings / recording
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self._layout.addWidget(self.splitter)

        # SETTINGS PANEL
        self.settings_panel = SettingsPanel()
        # and its signals
        self.settings_panel.user_channel_active.connect(self.set_user_channel)
        self.settings_panel.midi_channels_active.connect(self.set_midi_channels)
        # recording settings: change signals
        self.settings_panel.tuning_changed.connect(lambda x: self.pyin_settings_changed(x, "tuning"))
        self.settings_panel.fmin_changed.connect(lambda x: self.pyin_settings_changed(x, "fmin"))
        self.settings_panel.fmax_changed.connect(lambda x: self.pyin_settings_changed(x, "fmax"))
        
        # RECORDING PANEL
        self.init_recording_panel_ui()

        self.splitter.addWidget(self.settings_panel)
        self.splitter.addWidget(self.recording_panel)
        self.splitter.setSizes([200, 700])

        # playback variables
        self.slider_is_moving = False
        self.user_can_play = True
        self.is_recording = False

        # pitch detected signal
        self.pitch_detector.pitch_detected.connect(self.on_pitch_detected)


    def init_recording_panel_ui(self):
        self.recording_panel = QWidget()
        self.recording_layout = QVBoxLayout()
        self.recording_panel.setLayout(self.recording_layout)

        # title
        self.recording_layout.addWidget(QLabel("Record"))
        self.recording_layout.addStretch()

        # pitch plot
        self.pitch_plot = PitchPlot()
        self.recording_layout.addWidget(self.pitch_plot)

        # the playback layout
        self.slider_layout = QHBoxLayout()

        # get the play/pause button icons
        app_directory = os.path.dirname(os.path.dirname(__file__)) # directory one level higher than this (root)
        play_filepath = os.path.join(app_directory, 'resources', 'icons', 'play.png')
        pause_filepath = os.path.join(app_directory, 'resources', 'icons', 'pause.png')
        record_filepath = os.path.join(app_directory, 'resources', 'icons', 'record.png')

        self.play_icon = QIcon(play_filepath)
        self.pause_icon = QIcon(pause_filepath)
        self.record_icon = QIcon(record_filepath)

        # add play button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(QSize(26, 26))
        self.play_button.clicked.connect(self.toggle_playback)
        self.slider_layout.addWidget(self.play_button)

        # add record button
        self.record_button = QPushButton()
        self.record_button.setIcon(self.record_icon)
        self.record_button.setFixedSize(QSize(26, 26))
        self.record_button.clicked.connect(self.toggle_recording)
        self.slider_layout.addWidget(self.record_button)

        # init the slider!
        self.slider = Slider()
        self.slider.slider_changed.connect(self.handle_slider_change)
        self.slider.slider_end.connect(self.handle_slider_end)
        self.slider_layout.addWidget(self.slider)
        self.recording_layout.addLayout(self.slider_layout)
        
        self.recording_layout.addStretch() 

    def update_midi(self, midi_data: MidiData):
        self.settings_panel.load_midi(midi_data)
        self.all_channels = midi_data.get_channels()
        # self.recording_panel.load_midi(midi_data)

    def load_user_audio(self, user_data: UserData):
        """load in a pre-recorded user audio"""
        self.user_data = user_data
        pitches = user_data.pitches.get_pitches()
        self.pitch_plot.plot_pitches(pitches, from_scratch=True)
        self.audio_player.load_audio(user_data.audio_data)

    def set_midi_channels(self, current_channels: dict[int, Qt.CheckState]):
        """update the current channels / reinitialize with new ones"""
        # print(f"the current channels: {current_channels}")
        active_channels = []
        for c, check_state in current_channels.items():
            if check_state == Qt.CheckState.Checked:
                active_channels.append(c)
        print(f"updating active_channels to: {active_channels}")
        self.midi_player.set_channels(active_channels)

    def set_user_channel(self, is_active: bool):
        self.user_can_play = is_active

    def pyin_settings_changed(self, value, type: str):
        """called whenever we detect the pyin settings in the settings_panel have changed"""
        if type == "fmin":
            self.pitch_detector.re_init(f0_min=value)
        elif type == "fmax":
            self.pitch_detector.re_init(f0_max=value)
        elif type == "tuning":
            self.pitch_detector.re_init(tuning=value)
        else:
            logging.error("invalid pyin setting type (how did this happen lol?)")

    def handle_slider_change(self):
        self.current_time = self.slider.get_time()
        self.pitch_plot.move_plot(self.current_time)

    def handle_slider_end(self):
        """make sure slider ends gracefully"""
        if self.slider_is_moving:
            self.toggle_playback()

    def toggle_playback(self):
        current_time = self.slider.get_time()

        if not self.slider_is_moving:
            self.slider.start_timer()
            self.slider_is_moving = True

            if self.is_recording:
                # pause everything else just in case
                if self.midi_player.midi_data:
                    self.midi_player.pause()
                if self.audio_player.audio_data:
                    self.audio_player.pause()

                self.slider_is_moving = False
                self.is_recording = False
                # run the pitch detection stuff
                self.pitch_detector.run(self.user_data, self.slider.get_time())

            if self.midi_data is not None and not self.is_recording:
                self.midi_player.play(start_time=current_time)

            if self.user_can_play and not self.is_recording:
                self.audio_player.play(start_time=current_time)
            
            else:
                self.audio_player.pause()

        else:
            self.slider.stop_timer()
            self.midi_player.pause()
            self.slider_is_moving = False
            self.is_recording = False

            # just always pause just in case
            self.audio_player.pause()
    
    def toggle_recording(self):
        current_time = self.slider.get_time()

        if not self.slider_is_moving:
            self.slider.start_timer()
            self.slider_is_moving = True
            self.recording = True

            # pause everything else just in case
            if self.midi_player.midi_data:
                self.midi_player.pause()
            if self.audio_player.audio_data:
                self.audio_player.pause()

            # run the pitch detection stuff
            print(f"starting recording at {current_time}")
            self.audio_recorder.run(current_time)
            self.pitch_detector.run(current_time)
            
        else:
            self.slider.stop_timer()
            self.audio_recorder.stop()
            self.pitch_detector.stop()
            self.slider_is_moving = False
            self.is_recording = False

    def load_midi(self, midi_data: MidiData):
        self.midi_data = midi_data
        self.midi_player.load_midi(midi_data)
        self.settings_panel.load_midi(midi_data)
        self.pitch_plot.plot_midi(midi_data)

    def load_audio(self, user_data: UserData):
        self.user_data = user_data
        self.audio_player.load_audio(user_data.audio_data)
        # pitch_line = [p[0] for p in self.user_data.pitch_data]
        self.pitch_plot.plot_pitches(self.user_data.pitch_data, from_scratch=True)

    def on_pitch_detected(self, pitch_time: float):
        """plots the newly detected pitch onto pitchplot"""
        pitch = self.user_data.pitch_data.read_pitch(pitch_time)
        self.pitch_plot.plot_pitch(pitch)