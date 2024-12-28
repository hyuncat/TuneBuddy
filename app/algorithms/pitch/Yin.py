import numpy as np
from numba import njit
from app.config import AppConfig

class Yin:

    def autocorrelation_fft(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast autocorrelation function implementation using Wiener-Khinchin theorem,
        which computes autocorrelation as the inverse FFT of the signal's power spectrum.

        Step 1 of Yin algorithm, corresponding to equation (1) in Cheveigne, Kawahara 2002.

        Args:
            audio_frame: The current frame of audio samples in Yin algorithm
            tau_max: Check for all time lags up to this value for in autocorrelation

        Returns:
            autocorrelation: The similarity curve.
            amplitudes: Amplitudes of the frame.
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

        amplitudes = np.abs(fft_frame)

        # Only return valid overlapping values up to window_size-tau_max
        # (type II autocorrelation)
        return autocorrelation[:w-tau_max],amplitudes
    
    # def difference_function(audio_frame: np.ndarray, tau_max: int):
    #     """
    #     The square difference function implemented in the seminal YIN paper
    #     """
    #     x = np.array(audio_frame, np.float64)  # Ensure float64 precision
    #     w = x.size
    #     tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

    #     autocorr, power_spec, amplitudes = Yin.autocorrelation_fft(x, tau_max)
        
    #     # Compute m'(tau) - terminology from McLeod
    #     m_0 = 2*np.sum(x ** 2) # initial m'(0)

    #     # Compute m'(tau) for each possible tau (McLeod 3.3.4)
    #     m_primes = np.zeros(tau_max)
    #     m_primes[0] = m_0
    #     for tau in range(1, tau_max):
    #         m_primes[tau] = m_primes[tau-1] - x[tau-1]**2 + x[w-tau]**2

    #     # Slice m_primes to only contain valid overlapping values
    #     m_primes = m_primes[:w-tau_max]

    #     # Compute the square difference function
    #     sdf = m_primes - 2*autocorr
    #     return sdf, power_spec, amplitudes
    
    @staticmethod
    def difference_function(audio_frame: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the raw difference function for the YIN algorithm without assuming constant energy.
        
        Args:
            audio_frame: The current frame of audio samples.
            tau_max: The maximum lag to compute up to.
            
        Returns:
            diff_fct: The difference function values for lags 0 to tau_max.
            amplitudes: Amplitudes of the frame.
        """
        x = np.array(audio_frame, dtype=np.float64)  # Ensure float64 precision
        w = x.size
        tau_max = min(tau_max, w)  # Ensure tau_max is within the window size

        # compute fast autocorrelation using FFT
        autocorr, amplitudes = Yin.autocorrelation_fft(x, tau_max)

        # compute the energy (r_t(0) and r_{t+\tau}(0)) for each lag
        r_0 = np.sum(x**2)
        energy = np.full(autocorr.shape, r_0)

        diff_fct = energy[0] + energy - 2*autocorr
        diff_fct[0] = 0

        return diff_fct, amplitudes
    
    def cmndf(diff_fct: np.ndarray, tau_max: int) -> np.ndarray:
        """
        Cumulative Mean Normalized Difference Function (CMNDF).

        The idea is to normalize each d_t(tau) value for all lags based on the mean of the
        cumulative sum of all differences leading up to that point. YIN solution to not
        picking the zero-lag peak.

        Args:
            audio_frame: The current frame of audio samples in Yin algorithm
            tau_max: Check for all time lags up to this value for in autocorr

        Returns:
            cmndf: Array of values where index=tau and value=CMNDF(tau)
        """
        tau_max = min(tau_max, diff_fct.size)  # ensure tau_max is within the window size

        # Compute the Cumulative Mean Normalized Difference Function (CMNDF)
        cmndf = np.zeros(tau_max)
        cmndf[0] = 1  
        total_diff = 0.0

        for tau in range(1, tau_max):
            total_diff += diff_fct[tau]
            avg_diff = total_diff / tau
            cmndf[tau] = diff_fct[tau] / avg_diff

        return cmndf
    
    @njit
    def parabolic_interpolation(diff_fct: np.ndarray, trough_index: int) -> float:
        """
        Perform parabolic interpolation around a minimum point using Numba for optimization.
        
        Args:
            diff_fct: A 1D array of y-values (e.g., diff_fct values).
            trough_index: The index of the minimum point in cmndf_frame.

        Returns:
            A tuple with the interpolated x & y coordinates of the minimum.
        """
        x = trough_index

        # don't interpolate at boundaries - need at least 3 points
        if x <= 0 or x >= len(diff_fct) - 1:
            return float(x)
    
        y_1 = diff_fct[x - 1]
        y_2 = diff_fct[x]
        y_3 = diff_fct[x + 1]

        denominator = 2 * (y_1 - 2 * y_2 + y_3)
        if denominator == 0:
            return float(x)
        
        x_interpolated = x + (y_1 - y_3) / denominator
        # y_interpolated = y_2 - ((y_1 - y_3)**2) / (4 * denominator)

        return x_interpolated
    
    @staticmethod
    def absolute_threshold(cmndf_frame: np.ndarray, local_minima: list[int], threshold: float=0.1) -> tuple[int, bool]:
        """
        Apply the absolute thresholding step by searching for the first trough
        below a certain 'absolute threshold'

        Args:
            cmndf_frame: Output from applying CMNDF over a frame of audio samples
            local_minima: Indices of local minima of the raw diff_fct
            threshold: Take the first trough below this value of d'(tau)
        
        Returns:
            tau_0 (int): The fundamental period estimate. If possible, the first tau st. 
                         d'(tau) < threshold. Else, the x of the global minima
            is_voiced (bool): False if we return the global min
        """
        for min in local_minima:
            if cmndf_frame[min] <= threshold:
                return min, True
        
        # no min found below threshold, return the global minima
        global_min = np.argmin(cmndf_frame)
        return global_min, False