import numpy as np
import pandas as pd
from numba import jit

from app.core.audio.AudioData import AudioData


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
    def __init__(self, time: float, frequency: float, probability: float, volume: float, audio_idx: int, config: PitchConfig):
        self.time = time
        self.frequency = frequency
        self.probability = probability
        self.volume = volume
        self.audio_idx = audio_idx

        self.config = config
        
        self.max_midi = self.freq_to_midi(self.config.fmax)
        self.min_midi = self.freq_to_midi(self.config.fmin)
        
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

    def freq_to_midi(self, freq: float) -> float:
        """
        Convert a frequency to a MIDI note number (without Numba).
        """
        return PitchConversion.freq_to_midi(freq, self.config.tuning)

    def midi_to_freq(self, midi_num: float) -> float:
        """
        Convert a MIDI note number to frequency (without Numba).
        """
        return PitchConversion.midi_to_freq(midi_num, self.config.tuning)
    
    def bin_index_to_midi(bin_index: int, pitch_config: PitchConfig) -> float:
        """
        Convert a pitch bin index to a MIDI note number.
        """
        return bin_index / pitch_config.bins_per_semitone + pitch_config.min_midi

        

class PitchDf:
    def __init__(self, audio_data: AudioData, config: PitchConfig, pitches:list[Pitch]=None):
        self.config = config
        self.audio_data = audio_data

        self.df = pd.DataFrame(
            columns=['time', 'frequency', 'midi_num', 
                     'probability', 'volume', 'audio_idx']
        )
        if pitches is not None:
            self.df = self.pitches_to_dataframe(pitches)
        
        self.df2 = self.best_prob_df()

    def pitches_to_dataframe(self, pitches: list[Pitch]) -> pd.DataFrame:
        """
        Convert a list of Pitch objects into a pandas DataFrame.

        Args:
            pitches: A list of Pitch objects.
        
        Returns:
            A pandas DataFrame containing the attributes of the Pitch objects.
        """
        # Extract the relevant data from the Pitch objects
        pitch_data = {
            'time': [pitch.time for pitch in pitches],
            'frequency': [pitch.frequency for pitch in pitches],
            'midi_num': [PitchConversion.freq_to_midi(pitch.frequency, self.config.tuning) for pitch in pitches],  # Extracted from the freq_to_midi method
            'probability': [pitch.probability for pitch in pitches],
            'volume': [pitch.volume for pitch in pitches],
            'audio_idx': [pitch.audio_idx for pitch in pitches]
        }

        # Create and return the DataFrame
        return pd.DataFrame(pitch_data)

    def append_pitch(self, time: float, freq: float, prob: float, volume: float, audio_idx: int=None):
        """
        Append a new pitch to the DataFrame, including calculating the corresponding MIDI number.

        Args:
            time: Time in seconds.
            frequency: Frequency of the pitch.
            probability: Probability of the pitch detection.
            volume: Volume of the pitch.
        """
        # Calculate the MIDI note number using the PitchConversion class
        midi_num = PitchConversion.freq_to_midi(freq, self.config.tuning)
        
        # Create a new row for the pitch
        new_row = {
            'time': time,
            'frequency': freq,
            'midi_num': midi_num,
            'probability': prob,
            'volume': volume,
            'audio_idx': audio_idx
        }

        # Append the row to the DataFrame
        self.df = pd.concat([self.df, pd.Series(new_row)], ignore_index=True)
    
    def best_prob_df(self):
        """
        Return a subset DataFrame containing only the highest probability pitch for each unique time.

        Returns:
            A DataFrame with the highest probability pitch for each unique time.
        """
        # Find the index of the highest probability pitch for each unique time
        idx = self.df.groupby('time')['probability'].idxmax()
        
        # Return a DataFrame with only the highest probability pitches
        return self.df.loc[idx].reset_index(drop=True)
    
    def find_closest_pitch(self, target_time: float):
        """
        Find the pitch closest to a given time.

        Args:
            target_time: Time in seconds for which to find the closest pitch.

        Returns:
            The row in the DataFrame corresponding to the pitch closest to the given time.
        """
        closest_idx = np.abs(self.df['time'] - target_time).idxmin()
        return self.df.iloc[closest_idx]


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


