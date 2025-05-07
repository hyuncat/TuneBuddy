import numpy as np
import threading
from collections import deque

from app_logic.user.ds.AudioData import AudioData
from algorithms.PitchDetector import PitchDetector


class Buffer:
    """a lock-safe queue where newly recorded audio is written, 
    and pitches are detected batch-wise"""
    def __init__(self):
        self.buffer = deque()
        self.time = 0
        self.lock = threading.Lock()

    def write(self, indata, start_time: float=0):
        """writes some iterable data (np.array, list) into our queue"""
        with self.lock:
            self.data.extend(indata)

    def read(self, n):
        """reads n samples from the audio queue then pops it off"""
        with self.lock:
            if len(self.data) < n:
                return None
            out = [self.data.popleft() for _ in range(n)]
        return out

class UserData:
    def __init__(self):
        """the user data"""
        # essential data variables
        self.audio_data = AudioData()
        self.pitch_data = None
        self.note_data = None

        # queue data structures for real time pitch + note detection + correction
        self.a2p_queue = Buffer() #audio-to-pitches
        self.p2n_queue = None #pitches-to-notes
        self.n2c_queue = None #notes-to-corrections

        # algorithms
        self.pitch_detector = PitchDetector()

    def load_audio(self, audio_filepath: str):
        """load in a pre-recorded audio file from a filepath
        also computes pitches on the entire file"""
        self.audio_data.load_data(audio_filepath)
        self.pitch_data = self.pitch_detector.detect_pitches(self.audio_data.data)

    def write_data(self, indata: np.ndarray, start_time: float):
        """write indata to the audio_data at the given start_time
        and append to our queue for pitch processing
        """
        self.audio_data.write_data(indata, start_time)
        self.a2p_queue.write(indata)