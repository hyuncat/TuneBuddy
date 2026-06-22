import numpy as np
import threading
import soundfile as sf
import librosa
from pydub import AudioSegment

from algorithms.Config import Config

class AudioData:
    def __init__(self, length: float=60, audio_filepath: str=None, config: Config=None):
        """
        initialize user's audio data with an array of zeros of length equal to 
        the supplied length (sec).
        """
        self.sr = config.sr if config is not None else 44100

        # initialize the audio data array with all zeros, with capacity 
        # based on the supplied length (sec)
        self.capacity = int(length * self.sr)
        self.data = np.zeros(self.capacity, dtype=np.float32)
            
        if audio_filepath is not None:
            self.load_data(audio_filepath) # (also sets capacity + sr)

        # ensure thread-safe access to the buffer
        # as AudioRecorder and AudioPlayer will be accessing it
        self.lock = threading.Lock()
        self.end_index = 0 # track end of recorded audio

    def load_data(self, audio_filepath: str, sr: float=44100):
        """
        Load audio data into the recording data array.
        Args:
            audio_filepath (str): A correct file path pointing to audio data to load
        """
        if audio_filepath.endswith(".m4a"):
            seg = AudioSegment.from_file(audio_filepath, format="m4a")
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
            data = samples / (2 ** (seg.sample_width * 8 - 1))
        else:
            data, sr = sf.read(audio_filepath, always_2d=True)
            data = data.mean(axis=1)

        self.data = data
        self.sr = sr
        self.capacity = len(data)
        self.end_index = len(data)

    def write_data(self, indata: np.ndarray, start_time: float=0):
        """
        Add a new audio chunk to the recording data, growing the self.data array if necessary.
        Args:
            buffer (np.ndarray): Temporary buffer of new audio data to be added
            start_time (float), time in seconds to start adding the new chunk
        """
        start_index = int(start_time * self.sr)
        end_index = start_index + len(indata)

        if end_index > self.capacity:
            # double the capacity
            self.capacity *= 2
            with self.lock:
                self.data = np.resize(self.data, self.capacity)

        # write the new audio data to the data array
        with self.lock:
            self.data[start_index:end_index] = indata
            self.end_index = max(self.end_index, end_index)

    def read_data(self, start_time: float=0, end_time: float=0) -> np.ndarray:
        """
        lock-safe read audio data from the recording data array.
        Args:
            start_time (float): time in seconds to start reading from
            end_time (float): time in seconds to stop reading
        Returns:
            data (np.ndarray): audio data array from start_time to end_time
        """
        start_index = int(start_time * self.sr)
        end_index = int(end_time * self.sr)

        with self.lock:
            return self.data[start_index:end_index]

    def get_length(self) -> int:
        """
        get the length of the audio data in seconds
        """
        return self.end_index / self.sr
    

