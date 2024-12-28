import numpy as np
import time
from numba import njit
import scipy
import librosa
import scipy.signal as signal
from dataclasses import dataclass
from typing import Optional
from .conversions import midi_to_freq, freq_to_midi, freq_to_pitchbin
from .Transition import TransitionMatrix

@dataclass
class Pitch:
    """
    Stores pitch properties in the CMNDF function.
    """
    time: float
    frequency: float
    probability: float
    pitch_bin: Optional[int] = None

class PYin:
    def __init__(self):
        pass

    def high_pass_iir_filter(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # High pass filter to lower intensity of low frequency noise
        # Based on McLeod
        nyquist = sample_rate / 2
        CUTOFF_FREQ = 150
        normal_cutoff = CUTOFF_FREQ / nyquist
        sos = signal.iirfilter(
            N=4, Wn=normal_cutoff, rp=3, 
            btype='highpass', 
            ftype='butter', 
            output='sos', 
            fs=sample_rate
        )
        audio_data = signal.sosfilt(sos, audio_data)
        return audio_data

    
    def autocorrelation_fft(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast autocorrelation function implementation using Wiener-Khinchin theorem,
        which computes autocorrelation as the inverse FFT of the signal's power spectrum.

        Step 1 of Yin algorithm, corresponding to equation (1) in Cheveigne, Kawahara 2002.

        Args:
            audio_frame: The current frame of audio samples in Yin algorithm
            tau_max: The time-lag to check for in the autocorrelation
        """
        w = audio_frame.size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        # Zero-pad the audio signal array by the minimum power of 2 which
        # is larger than the window size + tau_max.
        # (circular instead of linear convolution, avoids errors)
        min_fft_size = w + tau_max  # (pad by >tau_max for frame end)

        p2 = (min_fft_size // 32).bit_length()
        nice_fft_sizes = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(size * (2 ** p2) for size in nice_fft_sizes if size * 2 ** p2 >= min_fft_size)

        # Decompose the signal into its frequency components
        fft_frame = np.fft.rfft(audio_frame, size_pad)  # Use only real part of the FFT (faster)

        # Compute the autocorrelation using Wiener-Khinchin theorem
        power_spectrum = fft_frame * fft_frame.conjugate()
        autocorrelation = np.fft.irfft(power_spectrum)[:tau_max]

        # Only return valid overlapping values up to window_size-tau_max
        # (type II autocorrelation)
        return autocorrelation[:w-tau_max], power_spectrum

    
    def square_difference_fct(audio_frame: np.ndarray, tau_max: int):
        """
        The square difference function implemented in the seminal YIN paper.
        """
        x = np.array(audio_frame, np.float64)  # Ensure float64 precision
        w = x.size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        autocorr, power_spec = PYin.autocorrelation_fft(x, tau_max)
        
        # Compute m'(tau) - terminology from McLeod
        m_0 = 2*np.sum(x ** 2) # initial m'(0)

        # Compute m'(tau) for each possible tau (McLeod 3.3.4)
        m_primes = np.zeros(tau_max)
        m_primes[0] = m_0
        for tau in range(1, tau_max):
            m_primes[tau] = m_primes[tau-1] - x[tau-1]**2 + x[w-tau]**2

        # Slice m_primes to only contain valid overlapping values
        m_primes = m_primes[:w-tau_max]

        # Compute the square difference function
        sdf = m_primes - 2*autocorr
        return sdf, power_spec

    def cmndf(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative Mean Normalized Difference Function (CMNDF) implementation.
        """
        x = np.array(audio_frame, np.float64)  # Ensure float64 precision
        w = x.size # window size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        # Compute the difference function
        # autocorr, _ = autocorrelation_fft(x, tau_max)
        # x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
        # diff_fct = x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * autocorr
        diff_fct, power_spec = PYin.square_difference_fct(x, tau_max)

        # Compute the Cumulative Mean Normalized Difference Function (CMNDF)
        cmndf = np.zeros(tau_max)
        cmndf[0] = 1  
        cumulative_diff = 0.0

        for tau in range(1, tau_max):
            cumulative_diff += diff_fct[tau]
            cmndf[tau] = diff_fct[tau] / (cumulative_diff/tau)

        return cmndf, power_spec


    @njit
    def parabolic_interpolation(cmndf_frame: np.ndarray, trough_index: int) -> tuple[float, float]:
        """
        Perform parabolic interpolation around a minimum point using Numba for optimization.
        
        Args:
            cmndf_frame: A 1D array of y-values (e.g., CMNDF values).
            trough_index: The index of the minimum point in cmndf_frame.

        Returns:
            A tuple with the interpolated x & y coordinates of the minimum.
        """
        x = trough_index
        if x <= 0 or x >= len(cmndf_frame) - 1:
            return float(x), cmndf_frame[x]  # No interpolation possible at the boundaries
        
        alpha = cmndf_frame[x - 1]
        beta = cmndf_frame[x]
        gamma = cmndf_frame[x + 1]

        denominator = 2 * (alpha - 2 * beta + gamma)
        if denominator == 0:
            return float(x), beta
        
        x_interpolated = x + (alpha - gamma) / denominator
        y_interpolated = beta - (alpha - gamma) * (alpha - gamma) / (4 * denominator)

        return x_interpolated, y_interpolated
  
    # Question: How is voicing determined within this function? 
    #   -> global_min? n_thresholds_below_min?
    @njit
    def _pthreshold(trough_prior: np.ndarray, trough_threshold_matrix: np.ndarray, beta_pdf, no_trough_prob: float=0.01):
        """
        Compute the probabilities of each trough using the prior distribution
        and beta distribution of thresholds, optimized with Numba.

        Args:
            trough_prior: The prior distribution of troughs.
            trough_threshold_matrix: A boolean matrix indicating threshold presence.
            beta_pdf: The probability density function of the beta distribution.
            no_trough_prob: The probability of no troughs being present.

        Returns:
            A 1D array of probabilities for each trough.
        """
        trough_probs = trough_prior.dot(beta_pdf)
        global_min = np.argmin(trough_probs)
        n_thresholds_below_min = np.count_nonzero(~trough_threshold_matrix[global_min, :])
        trough_probs[global_min] += no_trough_prob * np.sum(beta_pdf[:n_thresholds_below_min])
        return trough_probs

    def probabilistic_thresholding(cmndf_frame: np.ndarray, thresholds, beta_pdf) -> tuple[list[float], list[float]]:
        """
        Get all possible pitch candidates + probabilities for a given audio frame's CMNDF.
        Corresponds to the probabilistic thresholding step in the PYIN algorithm.
        
        Args:
            cmndf_frame: The CMNDF function for a single audio frame.
            thresholds: The thresholds to use for probabilistic thresholding.
            beta_pdf: The probability density function of the beta distribution.
        
        Returns:
            A tuple with lists of pitch candidates and their respective probabilities.
        """
        base_trough_indices = scipy.signal.argrelmin(cmndf_frame, order=1)[0]
        troughs = [PYin.parabolic_interpolation(cmndf_frame, i) for i in base_trough_indices]

        trough_x_vals = np.array([trough[0] for trough in troughs])
        trough_y_vals = np.array([trough[1] for trough in troughs])

        trough_threshold_matrix = np.less.outer(trough_y_vals, thresholds)
        trough_ranks = np.cumsum(trough_threshold_matrix, axis=0) - 1 # count how many troughs are below threshold
        n_troughs = np.count_nonzero(trough_threshold_matrix, axis=0)

        BOLTZMANN_PARAM = 2.0
        trough_prior = scipy.stats.boltzmann.pmf(trough_ranks, BOLTZMANN_PARAM, n_troughs)
        trough_prior[~trough_threshold_matrix] = 0

        trough_probabilities = PYin._pthreshold(trough_prior, trough_threshold_matrix, beta_pdf, no_trough_prob=0.01)

        SAMPLE_RATE = 44100
        trough_frequencies = SAMPLE_RATE / trough_x_vals

        return trough_frequencies, trough_probabilities

    def calculate_alpha_beta(mean_threshold, total=20):
        """
        Calculate alpha and beta for a beta distribution given a desired mean and a total sum of alpha and beta.
        """
        alpha = mean_threshold * total
        beta = total - alpha
        return alpha, beta


    @staticmethod
    def pyin(audio_data: np.ndarray, mean_threshold: float = 0.3):
        """
        The Probabilistic YIN algorithm for pitch estimation.
        """

        print("Starting pYIN algorithm...")

        sr = 44100
        frame_size = 2048*2
        hop_size = 128

        audio_data = PYin.high_pass_iir_filter(audio_data, sr)

        N_THRESHOLDS = 100
        thresholds = np.linspace(0, 1, N_THRESHOLDS) 
        cdf_thresholds = np.linspace(0, 1, N_THRESHOLDS + 1) # add one because we create using np.diff of a cdf

        # Create beta distribution centered around the desired mean threshold
        MEAN_THRESHOLD = 0.3
        alpha, beta = PYin.calculate_alpha_beta(MEAN_THRESHOLD, total=20)
        # print(f"Mean {MEAN_THRESHOLD} has alpha: {alpha}, beta: {beta}")
        beta_cdf = scipy.stats.beta.cdf(x=cdf_thresholds, a=alpha, b=beta) # How are alpha and beta calculated?
        beta_pdf = np.diff(beta_cdf) # where we know the total mass = 1

        pitches = []
        most_likely_pitches = []
        voiced_probs = []
        num_frames = (len(audio_data) - frame_size) // hop_size
        for frame_idx in range(num_frames):
            # Print the frame count in place
            print(f"\rProcessing frame {frame_idx + 1}/{num_frames}", end='')

            i = frame_idx * hop_size
            time = i / sr
            # Compute the CMNDF function for each frame
            audio_frame = audio_data[i:i+frame_size]
            cmndf_frame, _ = PYin.cmndf(audio_frame, frame_size // 2)
            
            # Compute all possible pitch candidates for the frame
            frequencies, probabilities = PYin.probabilistic_thresholding(cmndf_frame, thresholds, beta_pdf)

            # max_prob = 0
            most_likely_pitch = None

            for freq, prob in zip(frequencies, probabilities):
                pitch_bin = freq_to_pitchbin(freq, bins_per_semitone=10, tuning=440, fmin=196, fmax=5000)
                pitch = Pitch(time=time, frequency=freq, probability=prob, pitch_bin=pitch_bin)
                pitches.append(pitch)

            # Get the voiced probability for each frame as the sum of all pitch probabilities
            voiced_prob = np.clip(np.sum(probabilities), 0, 1)
            assert 0 <= voiced_prob <= 1
            voiced_probs.append(voiced_prob)

            # Create the emission matrix with 

            # Append the most likely pitch candidate for the frame to a separate list
            i = np.argmax(probabilities)
            best_prob = probabilities[i]
            best_freq = frequencies[i]
            best_pitch_bin = freq_to_pitchbin(best_freq, bins_per_semitone=10, tuning=440, fmin=196, fmax=5000)
            most_likely_pitch = Pitch(time=time, frequency=best_freq, probability=best_prob, pitch_bin=best_pitch_bin)

            most_likely_pitches.append(most_likely_pitch)
            
        print("\nDone!")
        return pitches, most_likely_pitches, voiced_probs


# def snac_fct(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
#     """
#     SNAC function implementation from McLeod's thesis.
#     """
#     x = np.array(audio_frame, np.float64)  # Ensure float64 precision
#     w = x.size
#     tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

#     # Compute the autocorrelation and power spectrum
#     sdf, power_spec = square_difference_fct(x, tau_max)

#     # Compute m'(tau)
#     m_0 = 2*np.sum(x ** 2) # initial m'(0)

#     # Compute m'(tau) for each possible tau (McLeod 3.3.4)
#     m_primes = np.zeros(tau_max)
#     m_primes[0] = m_0

#     for tau in range(1, tau_max):
#         m_primes[tau] = m_primes[tau-1] - x[tau-1]**2 + x[w-tau]**2

#     # Slice m_primes to only contain valid overlapping values
#     m_primes = m_primes[:w-tau_max]

#     # Compute the SNAC function
#     snac = 1 - sdf / m_primes

#     return snac, power_spec