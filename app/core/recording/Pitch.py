import numpy as np
from numba import jit

class PitchConfig:
    def __init__(self, bins_per_semitone=10, tuning=440.0, fmin=196, fmax=5000):
        self.bins_per_semitone = bins_per_semitone
        self.tuning = tuning
        self.fmin = fmin
        self.fmax = fmax

        self.max_midi = PitchConversion.freq_to_midi(fmax, tuning)
        self.min_midi = PitchConversion.freq_to_midi(fmin, tuning)

        # Total number of pitch bins for the given fmin, fmax, bins_per_semitone
        self.n_pitch_bins = int(
            (PitchConversion.freq_to_midi(fmax, tuning) 
             - PitchConversion.freq_to_midi(fmin, tuning)
            ) * bins_per_semitone)

    def freq_to_pitchbin(self, freq: float) -> int:
        """
        Convert a frequency to a pitch bin using the configuration.
        """
        midi_num = PitchConversion.freq_to_midi(freq, self.tuning)
        pitch_bin = int(np.round(midi_num * self.bins_per_semitone))

        # clip it to be within our desired range (just in case)
        return int(np.clip(pitch_bin, self.fmin*self.bins_per_semitone, 
                           self.fmax*self.cbins_per_semitone))

    def pitchbin_to_freq(self, pitch_bin: int) -> float:
        """Convert pitch bin to frequency"""
        midi_num = pitch_bin / self.bins_per_semitone + self.min_midi
        freq = PitchConversion.midi_to_freq(midi_num, self.tuning)
        return freq

class Pitch:
    def __init__(self, time: float, frequency: float, probability: float, volume: float, config: PitchConfig):
        self.time = time
        self.frequency = frequency
        self.probability = probability
        self.volume = volume

        self.config = config
        
        self.max_midi = PitchConversion.freq_to_midi(self.config.fmax, self.config.tuning)
        self.min_midi = PitchConversion.freq_to_midi(self.config.fmin, self.config.tuning)
        
        # Here we call the numba-optimized function for freq to pitchbin
        self.pitch_bin = self.freq_to_pitchbin(frequency)

        # self.is_voiced = is_voiced

    def freq_to_pitchbin(self, freq: float) -> int:
        """
        Convert a frequency to a pitch bin using the configuration.
        """
        self.midi_num = PitchConversion.freq_to_midi(freq, self.config.tuning)
        min_pitch_bin = int(np.round(self.min_midi * self.config.bins_per_semitone))

        # get the pitch_bin index
        pitch_bin = int(np.round(self.midi_num * self.config.bins_per_semitone)) - min_pitch_bin
        # ensure it's within the range of MIDI numbers
        return int(np.clip(pitch_bin, 0, self.config.n_pitch_bins))
    
    def bin_index_to_midi(bin_index: int, pitch_config: PitchConfig) -> float:
        """
        Convert a pitch bin index to a MIDI note number.
        """
        return bin_index / pitch_config.bins_per_semitone + pitch_config.min_midi


class PitchConversion:
    def __init__(self):
        pass

    @staticmethod
    @jit(nopython=True)
    def freq_to_midi(freq: float, tuning: float) -> float:
        """
        Convert a frequency to a MIDI note number (Numba optimized).
        """
        return 69 + 12 * np.log2(freq / tuning)

    @staticmethod
    @jit(nopython=True)
    def midi_to_freq(midi_num: float, tuning: float) -> float:
        """
        Convert a MIDI note number to frequency (Numba optimized).
        """
        return tuning * (2 ** ((midi_num - 69) / 12))


# Example usage
if __name__ == "__main__":
    config = PitchConfig(bins_per_semitone=10, tuning=440.0, fmin=196, fmax=5000)
    pitch = Pitch(time=0.0, frequency=440.0, probability=0.9, config=config)

    print(f"Time: {pitch.time}")
    print(f"Frequency: {pitch.frequency}")
    print(f"Probability: {pitch.probability}")
    print(f"Pitch Bin: {pitch.pitch_bin}")

class Pitches:
    def __init__(self, pitches: list[Pitch], pyinner: 'PYin'):
        self.pitches = pitches
        self.pyinner = pyinner

    def get_pitch(self, time: float=None, rank: int=0):
        """
        Get a pitch from the user's pitches, based on the closest time 
        to the one provided. Returns an error if the rank (0 > 1 > 2 > ... probability of pitch)
        is invalid.
        
        Args:
            time: The time of the pitch you want to query
            rank: 0 for most probable, 1 for second most probable, etc.
        """
        sample_idx = int(time * self.pyinner.SAMPLE_RATE)
        pitch_idx = round(sample_idx / self.pyinner.HOP_SIZE)
        return self.pitches[pitch_idx][rank]
    
    def get_pitches(self, start_time: float=0, end_time: float=None, rank: int=0):
        """
        Gets all the pitches from the given start to the end time.
        Tries to get the rank of the pitch provided, but defaults to what's available.
        (ie, not all times may have multiple pitch estimates)
        """
        if not end_time:
            end_time = len(self.pitches) * self.pyinner.HOP_SIZE / self.pyinner.SAMPLE_RATE
        
        start_idx = round(start_time * self.pyinner.SAMPLE_RATE / self.pyinner.HOP_SIZE)
        end_idx = round(end_time * self.pyinner.SAMPLE_RATE / self.pyinner.HOP_SIZE)

        # clamp
        start_idx = max(0, min(start_idx, len(self.pitches) - 1))
        end_idx = max(0, min(end_idx, len(self.pitches) - 1))

        pitches = self.pitches[start_idx:end_idx]
        pitches = [
            next((p[max(rank-k, 0)] for k in range(rank + 1) if len(p) > max(rank-k, 0)), None)
            for i, p in enumerate(pitches) if p
        ]
        return pitches
