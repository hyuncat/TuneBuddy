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
    def __init__(self, time: float, frequency: float, probability: float, volume: float, config: PitchConfig):
        """just stores the different pitch attributes"""
        self.config = config # tuning / fmin/fmax

        # essential variables
        self.time = time
        self.frequency = frequency
        self.probability = probability
        self.volume = volume
        self.midi_num = self.config.freq_to_midi(frequency)

class PitchData:
    #TODO: change to store a reference to its parent userdata to handle the pitch-getting
    def __init__(self, pitches: list[Pitch], pitch_detector):
        self.pitches = pitches
        self.pitch_detector = pitch_detector

    def get_pitch(self, time: float=None, rank: int=0):
        """
        Get a pitch from the user's pitches, based on the closest time 
        to the one provided. Returns an error if the rank (0 > 1 > 2 > ... probability of pitch)
        is invalid.
        
        Args:
            time: The time of the pitch you want to query
            rank: 0 for most probable, 1 for second most probable, etc.
        """
        sample_idx = int(time * self.pitch_detector.SR)
        pitch_idx = round(sample_idx / self.pitch_detector.HOP_SIZE)
        return self.pitches[pitch_idx][rank]
    
    def get_pitches(self, start_time: float=0, end_time: float=None, rank: int=0):
        """
        Gets all the pitches from the given start to the end time.
        Tries to get the rank of the pitch provided, but defaults to what's available.
        (ie, not all times may have multiple pitch estimates)
        """
        if not end_time:
            end_time = len(self.pitches) * self.pitch_detector.HOP_SIZE / self.pitch_detector.SAMPLE_RATE
        
        start_idx = round(start_time * self.pitch_detector.SAMPLE_RATE / self.pitch_detector.HOP_SIZE)
        end_idx = round(end_time * self.pitch_detector.SAMPLE_RATE / self.pitch_detector.HOP_SIZE)

        # clamp
        start_idx = max(0, min(start_idx, len(self.pitches) - 1))
        end_idx = max(0, min(end_idx, len(self.pitches) - 1))

        pitches = self.pitches[start_idx:end_idx]
        pitches = [
            next((p[max(rank-k, 0)] for k in range(rank + 1) if len(p) > max(rank-k, 0)), None)
            for i, p in enumerate(pitches) if p
        ]
        return pitches
