import threading
from math import floor, ceil
import numpy as np

from algorithms.Config import Config

class PitchConfig:
    def __init__(self, tuning=440.0, fmin=196, fmax=5000):
        self.tuning = tuning
        self.fmin = fmin
        self.fmax = fmax

    def freq_to_midi(self, freq: float) -> float:
        """
        Convert a frequency to a MIDI note number.
        """
        if freq <= 0:
            # print("bad freq")
            return(-1)
        return 69 + 12 * np.log2(freq / self.tuning)

    def midi_to_freq(self, midi_num: float) -> float:
        """
        Convert a MIDI note number to frequency.
        """
        return self.tuning * (2 ** ((midi_num - 69) / 12))

    def load_config(self, config: dict):
        """load in a config dictionary"""
        self.tuning = config.get("tuning", self.tuning)
        self.fmin = config.get("fmin", self.fmin)
        self.fmax = config.get("fmax", self.fmax)

class Pitch:
    def __init__(self, time: float, candidates: list[tuple[float, float]], 
                 volume: float, unvoiced_prob: float, distance: float, config: Config):
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
        self.volume = volume # mean |amplitude| of the frame
        self.unvoiced_prob = unvoiced_prob # how messy the signal was (no clean periodicity)

        self.distance = distance # distance to target note (live / absolute-time)

        # post-analysis distance, assigned from the string-edit alignment rather
        # than the score note at this frame's absolute time. None until analyze()
        # runs; GuitarHero falls back to `distance` (live coloring) when it's None.
        # A pitch inside an insertion gets float('inf') so it always colors red.
        self.align_distance = None

class PitchData:
    def __init__(self, config: Config):
        """
        an audio data-like pitch data: an array of pitches
	       - indexable using the (SR / hop size)
        """
        # reference to the global config (for sr + hop size used in pitch detection)
        self.config = config 

        # the essential time to index lambda
        self.time_to_index = lambda sec: floor(sec*(self.config.sr / self.config.h1))
        
        DEFAULT_LENGTH = 60 # (sec)
        self.data: list[Pitch] = [None] * ceil(self.time_to_index(DEFAULT_LENGTH))
        self.lock = threading.Lock()

        self.UNVOICED_THRESHOLD = config.unv_thresh # threshold above which a pitch is considered unvoiced

    def resize(self, resize_factor=2):
        """increase the capacity of the current pitch array"""
        with self.lock:
            new_data = [None] * (len(self.data) * resize_factor)
            self.data.extend(new_data)

    def load(self, pitches: list[Pitch]):
        """load in an entire pitch array"""
        self.data = pitches

    def write(self, pitches: list[Pitch] | Pitch, start_time: float=0):
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

    def read(self, start_time: float=0, end_time: float=0, i: int=None, j: int=None, clean=False) -> list[Pitch]:
        """returns the array of pitches corresponding to start_time <--> end_time"""
        if not i and not j:
            i = max(0, self.time_to_index(start_time))
            j = min(self.time_to_index(end_time), len(self.data)-1)

        if clean:
            return [p for p in self.data[i:j] if p is not None and p.candidates!=[] and p.unvoiced_prob < self.UNVOICED_THRESHOLD]

        return self.data[i:j]
    
    def read_pitch(self, start_time: float=0) -> Pitch:
        """returns the closest pitch to the start_time"""
        i = self.time_to_index(start_time)
        return self.data[i]
