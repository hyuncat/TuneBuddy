import threading
from math import floor, ceil
import numpy as np

class PitchConfig:
    def __init__(self, tuning=440.0, fmin=196, fmax=5000):
        self.tuning = tuning
        self.fmin = fmin
        self.fmax = fmax

    def freq_to_midi(self, freq: float) -> float:
        """
        Convert a frequency to a MIDI note number (Numba optimized).
        """
        if freq <= 0:
            print("bad freq")
            return(-1)
        return 69 + 12 * np.log2(freq / self.tuning)

    def midi_to_freq(self, midi_num: float) -> float:
        """
        Convert a MIDI note number to frequency (Numba optimized).
        """
        return self.tuning * (2 ** ((midi_num - 69) / 12))

class Pitch:
    def __init__(self, time: float, candidates: list[tuple[float, float]], volume: float, config: PitchConfig):
        """
        The quintessential pitch object for the app.
        ---
        Corresponds to a given [time] in the PitchData and stores all possible 
        pitch [candidates] = [(midi_num, prob), ...] sorted from most --> least probable    
        as well as the volume and a reference to the settings (config) with which it was computed
        """
        self.config = config # tuning / fmin/fmax
        # -- essential variables --
        self.time = time
        self.candidates = candidates # [(midi_num, prob), ...]; sorted
        self.volume = volume

class PitchData:
    def __init__(self, pitch_detector):
        """
        an audio data-like pitch data
        """
        self.pd = pitch_detector # reference to the pitchdetector object which computed it
        self.time_to_index = lambda sec: floor(sec*(self.pd.SR / self.pd.HOP_SIZE))
        
        DEFAULT_LENGTH = 60 # (sec)
        self.data: list[Pitch] = [None] * ceil(self.time_to_index(DEFAULT_LENGTH))
        self.lock = threading.Lock()

    def resize(self, resize_factor=2):
        """increase the capacity of the current pitch array"""
        with self.lock:
            new_data = [None] * (len(self.data) * resize_factor)
            self.data.extend(new_data)

    def load(self, pitches: list[Pitch]):
        """load in an entire pitch array"""
        self.data = pitches

    def write(self, pitches: list[Pitch] | Pitch, start_time: float=None):
        """write the pitches to the data at the given time index"""
        if isinstance(pitches, Pitch):
            pitches = [pitches]
        if not start_time:
            start_time = pitches[0].time

        # get indices into data array
        i = self.time_to_index(start_time)
        j = i+len(pitches)

        if j > len(self.data)*0.8: # if close enough to end
            self.resize()

        with self.lock:
            self.data[i:j] = pitches

    def read(self, start_time: float=0, end_time: float=0) -> list[Pitch]:
        """returns the array of pitches corresponding to start_time <--> end_time"""
        i = self.time_to_index(start_time)
        j = min(self.time_to_index(end_time), len(self.data)-1)
        return self.data[i:j]
    
    def read_pitch(self, start_time: float=0) -> Pitch:
        """returns the closest pitch to the start_time"""
        i = self.time_to_index(start_time)
        return self.data[i]