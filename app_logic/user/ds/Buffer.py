import threading
from collections import deque

import itertools
from typing import Sequence

class Buffer:
    """a lock-safe queue where newly recorded audio is written, 
    and pitches are detected batch-wise"""
    def __init__(self, sr: int=44100):
        self.buffer = deque()
        self.t_0, self.t_curr = 0, 0
        self.sr = sr
        self.lock = threading.Lock()

    def init_start_time(self, t_0: float):
        self.t_0, self.t_curr = t_0, t_0

    def push(self, indata: Sequence[float]):
        """writes some iterable data (np.array, list) into our queue"""
        with self.lock:
            self.buffer.extend(indata)

    def pop(self, frame_size: int=4096, hop_size: int=128):
        """reads frame_size samples from the audio queue then pops it off
        also returns the start time of the samples in terms of audio_data
        then increments t_curr to reflect

        returns none if not enough samples to read
        """
        if len(self.buffer) < frame_size:
            return None, -1
        
        with self.lock:
            # return frame_size worth of data
            outdata = list(itertools.islice(self.buffer, 0, frame_size))
            if len(outdata) == 0:
                return None, -1

            # only move over by hop_size
            for i in range(hop_size):
                if self.buffer:
                    self.buffer.popleft()

        # update time variables
        read_time = self.t_curr
        self.t_curr += hop_size / self.sr

        return outdata, read_time
