import scipy.signal as signal
import numpy as np
from app.config import AppConfig

class Filter:
    def __init__(self):
        pass

    @staticmethod
    def high_pass_iir_filter(audio_data: np.ndarray, cutoff_freq=150, 
                             sr: int=AppConfig.SAMPLE_RATE) -> np.ndarray:
        """
        A 2nd order high pass IIR (infinite impulse response) filter to lower intensity 
        of low frequency noise below 150 Hz, based on method from McLeod's thesis.

        Args:
        audio_data (np.ndarray): The input audio signal as a 1D NumPy array.
        cutoff_freq (float, optional): The cutoff frequency (in Hz) for the high-pass filter. 
                                       Frequencies below this value will be attenuated. 
                                       Defaults to 150 Hz.
        sr (int, optional): The sampling rate (in Hz) of the audio signal. 
                            This is used to calculate the Nyquist frequency 
                            for normalizing the cutoff frequency. Defaults 
                            to the sample rate defined in `AppConfig.SAMPLE_RATE`.

        Returns:
            np.ndarray: The filtered audio signal as a 1D NumPy array, with reduced 
                        low-frequency noise.

        Example:
            >>> audio_signal = np.array([0.1, 0.2, 0.3, 0.4])  # Example input
            >>> filtered_signal = high_pass_iir_filter(audio_signal, cutoff_freq=150, sample_rate=44100)
            >>> print(filtered_signal)
        """
        nyquist_freq = sr / 2
        CUTOFF_FREQ = 150

        # normalize freq by nyquist (scipy expects Wn between 0 - 1)
        normal_cutoff = CUTOFF_FREQ / nyquist_freq
        sos = signal.iirfilter(
            N=2, Wn=normal_cutoff, rp=3, 
            btype='highpass', 
            ftype='butter', 
            output='sos', 
            fs=sr
        )
        audio_data = signal.sosfilt(sos, audio_data)
        return audio_data