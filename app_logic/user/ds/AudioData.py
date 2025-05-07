import numpy as np
import threading
import soundfile as sf
from app_logic.midi.MidiData import MidiData


class AudioData:
    def __init__(self, midi_data: MidiData=None, audio_filepath: str=None, sr: int=44100):
        """
        initialize user's audio data with an array of zeros of length equal to the MIDI file length
        if no midi_data supplied, default to 60 seconds worth of audio
        """
        self.midi_data = midi_data
        self.sr = sr

        # initialize the audio data array with all zeros, with capacity 
        # based on MIDI file length and app's SAMPLE_RATE
        if self.midi_data is not None:
            self.capacity = int(self.midi_data.get_length() * self.sr)
            self.data = np.zeros(self.capacity, dtype=np.float32)
        
        else: # if no MIDI file is provided, use a default length of 60 seconds
            DEFAULT_LENGTH = 60
            self.capacity = int(DEFAULT_LENGTH * self.sr)
            self.data = np.zeros(self.capacity, dtype=np.float32)
            
        if audio_filepath is not None:
            self.load_data(audio_filepath) # (also sets capacity + sr)

        # ensure thread-safe access to the buffer
        # as AudioRecorder and AudioPlayer will be accessing it
        self.lock = threading.Lock()

    def load_data(self, audio_filepath: str, sr: float=44100):
        """
        Load audio data into the recording data array.
        Args:
            audio_filepath (str): A correct file path pointing to audio data to load
        """
        data, sr = sf.read(audio_filepath, always_2d=True)
        data = (data[:,0] + data[:,1]) / 2 # convert to MONO

        self.data = data
        self.sr = sr
        self.capacity = len(data)

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
        return len(self.data) / self.sr
    

