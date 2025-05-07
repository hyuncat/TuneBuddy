import sounddevice as sd
import numpy as np
from PyQt6.QtCore import pyqtSignal
import threading

from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.UserData import UserData

class AudioRecorder:
    def __init__(self, user_data: UserData, sr: int=44100):
        self.buffer = np.array([])
        self.current_time: float = 0

        # important reference: to its parent user_data
        self.user_data = user_data

        # threading variables
        self.recording_thread = None
        self.thread_stop_event = threading.Event()

        # initialize inputstream object
        self.stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            callback=self._callback,
            blocksize=0 # dynamic block size for best performance
        )

    def _callback(self, indata, frames, time, status):
        """ callback function to write audio to our audio_data + audio_queue (+ timeref_queue?) """
        if status:
            print(status, flush=True)
        
        self.user_data.write_data(indata.flatten(), start_time=self.current_time)

    def update_current_time(self, new_time: float):
        self.current_time = new_time