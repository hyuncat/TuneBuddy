import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
import threading
from essentia.standard import MonoLoader
from app.core.audio.AudioData import AudioData
from app.config import AppConfig

class AudioPlayer:
    def __init__(self):
        # Audio data structures
        self._AudioData = None
        self.stream = None

        # Playback variables
        self.current_time = 0

        # Threading variables
        self.playback_thread = None
        self.thread_stop_event = threading.Event()

    def load_audio_file(self, audio_filepath: str):
        """
        Loads audio data from a file into the AudioPlayer.
        For prerecorded audio playback in the app.
        """
        self._AudioData = AudioData(audio_filepath=audio_filepath)
        self.current_time = 0 # Reset current time

    def load_audio_data(self, audio_data: AudioData):
        """
        Loads audio data directly from an AudioData object into the AudioPlayer.
        Used for when user records their own audio through the app.
        """
        self._AudioData = audio_data
        self.current_time = 0

    def play(self, start_time=0):
        """
        Play the audio from the given start_time (in sec).
        Uses current sample rate to find index into audio_buffer, and
        uses sounddevice's play function to play the audio.
        @param:
            - start_time: float, time (sec) to start playback from
        """
        if self._AudioData is None:
            logging.error("No audio data loaded. Exiting.")
            return
        
        # If playback_thread already exists, stop playback, then re-join thread
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.thread_stop_event.set()
            self.playback_thread.join()

        # Cleanup stop_event
        self.thread_stop_event.clear()

        # Set current time for playback
        self.current_time = start_time

        # Start playback thread
        self.playback_thread = threading.Thread(target=self._play)
        self.playback_thread.start()

    def _play(self):
        """
        Internal function for playing the audio in a separate thread.
        """
        try:
            # Read the entire audio data from the specified start time
            audio_segment = self._AudioData.read_data(self.current_time, self._AudioData.capacity / AppConfig.SAMPLE_RATE)
            if len(audio_segment) == 0:
                logging.error("No audio data available for playback.")
                return

            # Play the entire audio segment
            sd.play(audio_segment, samplerate=AppConfig.SAMPLE_RATE)
            sd.wait()  # Wait until playback is finished

        except Exception as e:
            logging.error(f"Error in playback: {e}")

    def pause(self):
        self.thread_stop_event.set()
        sd.stop()
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.playback_thread.join()
        self.is_playing = False

    def resume(self):
        # Resume playing from the current position
        if self._AudioData is not None:
            self.play(start_time=self.current_time)
        else:
            logging.error("No audio data loaded. Exiting.")
