import sounddevice as sd
import numpy as np
from PyQt6.QtCore import pyqtSignal
import threading

from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording

class AudioRecorder:
    def __init__(self, recording: Recording):
        # time variables
        self.t_0: float = 0
        self.t_curr: float = 0

        # important reference: to its parent user_data
        self.recording = recording
        self.sr = recording.audio_data.sr if recording else 44100 # default sr

        # threading variables
        self.recording_thread = None
        self.stop_event = threading.Event()

        # initialize inputstream object (takes in audio data block by block)
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            callback=self._callback,
            blocksize=0 # dynamic block size for best performance
        )

    def load_recording(self, recording: Recording):
        """Load a new Recording object to record into."""
        self.recording = recording
        self.sr = recording.audio_data.sr
        self.t_0 = 0
        self.t_curr = 0

    def run(self, start_time: float=0):
        """start recording from the given start_time (sec)"""
        self.stop()  # only have one thread at a time
        self.stop_event.clear()

        # keep track of the current start time
        self.t_0, self.t_curr = start_time, start_time
        self.recording.a2p_queue.init_start_time(start_time)

        # start the input stream
        self.recording_thread = threading.Thread(target=self._run, args=( ), daemon=True)
        self.recording_thread.start()

    def _run(self, start_time: float=0):
        """internal function to wait until our stop event is called to stop recording"""
        self.stream.start()
        self.stop_event.wait() # block until stop event is called
        self.stream.stop()

    def stop(self):
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_event.set()
            self.recording_thread.join() # pause the main thread until recording thread recognizes the stop event

    def _callback(self, indata, frames, time, status):
        """
        Is called every time a new audio block has been recorded and read into indata.
        This is where we write the audio data to our recording.audio_data.

        Args:
            indata: the audio data block that has been recorded
            frames: number of frames in the audio data block
            time: time information about the audio data block
            status: status information about the audio data block
        """
        if status:
            print(status, flush=True)
        
        # print(f"recording audio @ {self.t_curr}")
        indata = indata.flatten()
        self.recording.write_data(indata, start_time=self.t_curr)
        self.t_curr += len(indata) / self.sr # increment the current time
