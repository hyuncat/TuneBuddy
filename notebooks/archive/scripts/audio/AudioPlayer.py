#TODO: move AudioRecorder stuff here

import sounddevice as sd
import soundfile as sf
import numpy as np
import logging

class AudioPlayer:
    def __init__(self):
        # Audio data structures
        self.audio_buffer = None
        self.samplerate = 44100 # Default 44100 hz
        self.stream = None

        # Playback variables
        self.current_time = 0
        self.is_playing = False

    def load_audio(self, audio_file_path: str):
        self.audio_buffer, self.samplerate = sf.read(audio_file_path)
        self.current_time = 0 # Reset current time
        self.is_playing = False

    def play(self, start_time=0):
        """
        Play the audio from the given start_time (in sec).
        Uses current sample rate to find index into audio_buffer, and
        uses sounddevice's Stream to play the audio.
        @param:
            - start_time: float, time (sec) to start playback from
        """
        if self.audio_buffer is None:
            logging.error("No audio data loaded. Exiting.")
            return
        
        # Find the appropriate start_time index using the samplerate
        self.current_time = start_time
        start_index = int(start_time * self.samplerate)
        current_audio_buffer = self.audio_buffer[start_index:]

        # Play the audio
        sd.play(current_audio_buffer, self.samplerate)
        self.is_playing = True

    def pause(self):
        sd.stop()
        self.is_playing = False

    def resume(self):
        # Resume playing from the current position
        if self.audio_buffer is not None:
            self.play(start_time=self.current_time)
        else:
            logging.error("No audio data loaded. Exiting.")
