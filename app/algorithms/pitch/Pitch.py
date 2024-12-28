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
        self.bin_index = int(self.pitch_bin - (self.min_midi*self.config.bins_per_semitone))

    def freq_to_pitchbin(self, freq: float) -> int:
        """
        Convert a frequency to a pitch bin using the configuration.
        """
        self.midi_num = PitchConversion.freq_to_midi(freq, self.config.tuning)
        pitch_bin = int(np.round(self.midi_num * self.config.bins_per_semitone))
        # Ensure it's within the range of MIDI numbers
        return int(np.clip(pitch_bin, self.min_midi*self.config.bins_per_semitone, 
                           self.max_midi*self.config.bins_per_semitone))
    
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
