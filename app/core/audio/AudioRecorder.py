import logging
import sounddevice as sd
from sounddevice import CallbackFlags
import numpy as np
# from ctypes import CData
from PyQt6.QtCore import pyqtSignal
import threading
from app.config import AppConfig
from app.core.audio.AudioData import AudioData
from archive.PYin import PYin

class AudioRecorder:
    """
    AudioRecorder is a class for recording user audio using sounddevice.
    It records in bursts and keeps track of 'start_time' for each recording to
    calculate where in AudioData the recording should be appended.

    @attr:
    Audio data structures:
        - buffer (np.ndarray): buffer for storing current chunk of audio data
        - start_time (float): time when recording started
        - stream (sd.InputStream): sounddevice InputStream object
        - pitch_analyzer (PitchAnalyzer): object for pitch detection

    Threading variables:
        - recording_thread (threading.Thread): thread for recording
        - thread_stop_event (threading.Event): event to stop recording when is_set()
    """
    pitch_added = pyqtSignal(float)

    def __init__(self):
        """
        Initialize the AudioRecorder object for recording audio with sounddevice's
        InputStream object.

        Notes:
            - We record mono audio for simplying pitch detection
            - We use a 'blocksize' of AppConfig.FRAME_SIZE to match the frame 
              size for Essentia's PitchYin algorithm, even though sounddevice 
              recommends a varying blocksize (default=0) for better performance
        """
        super().__init__()
        self.buffer = np.array([])
        self.current_start_time = 0

        # Threading variables
        self.recording_thread = None
        self.thread_stop_event = threading.Event()

        # Initialize the InputStream object
        self.stream = sd.InputStream(
            samplerate=AppConfig.SAMPLE_RATE,
            channels=AppConfig.CHANNELS,
            callback=self._callback,
            blocksize=AppConfig.FRAME_SIZE # Number of samples / frame
        )
        # Initialize a PitchAnalyzer object for pitch detection
        # self.pitch_analyzer = PitchAnalyzer()
        self.audio_data = AudioData()

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, 
                  time, status: CallbackFlags) -> None:
        """
        Overloaded callback function for sounddevice's InputStream.
        Adds the incoming audio data to the audio buffer and emits the pitch
        of the current audio frame.
        """
        if status:
            logging.warning(status) # Print any errors
        self.buffer = np.append(self.buffer, indata)

        # Note: Should I perform pitch analysis here or in diff class?
        # Analyze the pitch of the audio frame (indata) and emit the results
        pitch, confidence = self.pitch_analyzer.get_frame_pitch(indata)
        self.pitch_added.emit(pitch, confidence)
        self.audio_data.write_data(indata, self.current_start_time)

    def start(self, start_time: float=0):
        """
        Start recording audio from the microphone using sounddevice's InputStream.
        """
        # If recording_thread already exists, stop recording, then re-join thread
        if self.recording_thread is not None and self.recording_thread.is_alive():
            self.thread_stop_event.set()
            self.recording_thread.join()

        # Cleanup old buffer / stop_event
        self.buffer = np.array([])
        self.thread_stop_event.clear()

        # Set start_time for the recording
        self.current_start_time = start_time

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._start)
        self.recording_thread.start()

    def _start(self):
        """
        Internal function for starting the audio recording thread.
        """
        self.stream.start()

    def pause(self) -> np.ndarray:
        """
        Stop recording audio and write the data to AudioData object.
        """
        if self.recording_thread is not None and self.recording_thread.is_alive():
            self.thread_stop_event.set()
            # Wait for thread to finish (will soon, since stop_event is set)
            self.recording_thread.join() 
        self.stream.stop()
    
    def kill(self):
        """
        Kill the audio stream, stopping and closing the audio device and
        clean up all resources.
        """
        self.stream.stop()
        self.stream.close()
        self.buffer = np.array([])