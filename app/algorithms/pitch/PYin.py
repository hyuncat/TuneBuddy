import numpy as np
import scipy.signal
import scipy.stats
from math import ceil

from app.algorithms.pitch.Yin import Yin
from app.core.recording.Pitch import PitchConfig, Pitch

class PYin:
    
    def __init__(self, sr: int=44100, f0_min: float=196.0, f0_max: float=5000.0, tuning: float=440.0):
        """
        Initialize the pitch detection parameters, like the tuning, frequency range, etc.
        Best to make it as specific as possible to your desired use case to improve accuracy of the detection.
        """
        # --- frame iteration variables ---
        self.SAMPLE_RATE = sr
        self.FRAME_SIZE = 4096
        self.HOP_SIZE = 128

        # --- pitch config variables ---
        # ensure max lag is big enough to detect lowest f0 (largest period)
        # defaults to violin min
        self.tau_max = int(sr / f0_min) 
        self.tau_min = int(sr / f0_max)
        # pitch config to associate tuning / fmin / fmax params for each pitch
        self.pitch_config = PitchConfig(tuning=tuning, fmin=f0_min, fmax=f0_max)

        # initialize beta distribution parameters
        self.UNVOICED_PROB = 0.01
        self.N_THRESHOLDS = 100
        self.beta_pdf, self.thresholds = PYin.threshold_prior(n_thresholds=self.N_THRESHOLDS)

    @staticmethod
    def threshold_prior(n_thresholds: int=100, a: float=2, b: float=34/3) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a beta distribution modeling the pdf for YIN thresholds,
        represented as a numpy array of size N_THRESHOLDS corresponding to the pdf values.
        Also returns the array of thresholds.

        Possible a,b parameters from paper:
            - mean=0.1 beta(a=2, b=18)
            - mean=0.15 beta(a=2, b=11.33)
            - mean=0.2 beta(a=2, b=8)

        Not yet sure how these parameters are determined but I got them from the paper.
        """
        # all thresholds
        thresholds = np.linspace(0, 1, n_thresholds+1)
        thresholds = thresholds[1:] # remove the 0 threshold
        beta_pdf = scipy.stats.beta.pdf(thresholds, a, b) / n_thresholds
        # print(f"initializing pyin with thresholds {thresholds}\nand beta_pdf {beta_pdf}")
        return beta_pdf, thresholds
    
    def prob_thresholding(self, troughs: np.ndarray, cmndf: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Given all relative minima of the difference curve (the 'troughs') we 
        compute the probability of all possible period = 1/f_0 estimates. Where 
        the x-position of the trough corresponds to the period estimate of the 
        audio signal (in samples).

        Args:
            troughs (np.ndarray): indices of all relative minima of the diff_fct
            cmndf (np.ndarray): cumulative mean normalized difference function, which 
                                biases smaller periods and makes us less likely to choose 
                                the zero lag

        Returns:
            tau_probs (np.ndarray): array of same shape as troughs, where corresponding
                                    indices represent associated probabilities
            unvoiced_prob (float): 1 - sum(tau_probs), eg adding up all the times we had to 
                                   take the global min because nothing was below the threshold
        """
        tau_probs = np.zeros_like(troughs)
        for i, threshold in enumerate(self.thresholds):
            tau_0, tau_idx, is_voiced = Yin.absolute_threshold(cmndf, troughs, threshold)
            # ensure the tau is within our frequency range (trying to minimize harmonic errors)
            if is_voiced and tau_0 <= self.tau_max and tau_0 >= self.tau_min:
                tau_probs[tau_idx] += self.beta_pdf[i]
            else:
                tau_probs[tau_idx] += self.beta_pdf[i] * self.UNVOICED_PROB
            
        unvoiced_prob = 1 - np.sum(tau_probs)
        return tau_probs, unvoiced_prob

    def pyin(self, audio_data: np.ndarray):
        """
        Computes the PYIN algorithm on an audio array of samples.
        Mauch, Dixon 2014
        """

        # get memory efficient frames with np pointer c++ magic
        frames = np.lib.stride_tricks.sliding_window_view(audio_data, self.FRAME_SIZE)[::self.HOP_SIZE]
        n_frames = 1 + (len(audio_data) - self.FRAME_SIZE) // self.HOP_SIZE 

        pitches = []
        # most_likely_pitches = []

        for i, frame in enumerate(frames):
            print(f"\rProcessing frame {i+1}/{n_frames}", end='')

            time = (i*self.HOP_SIZE)/self.SAMPLE_RATE # elapsed time of the frame

            diff_frame, amplitudes = Yin.difference_function(frame, self.tau_max)
            cmndf_frame = Yin.cmndf(diff_frame, self.tau_max, self.tau_min)

            prominence = ceil((np.max(cmndf_frame) - np.min(diff_frame))/2)

            # prominence picking - set prominence as 1/2 amplitude
            # and keep trying to find peaks until at least one valid peak found
            while True:
                trough_indices, _ = scipy.signal.find_peaks(-cmndf_frame, prominence=prominence, distance=5)
                if len(trough_indices) > 0 or prominence==0: # stop when we have at least 1 peak
                    break
                prominence -= 0.5
            
            # parabolic interpolation for final candidates
            tau_candidates = [Yin.parabolic_interpolation(diff_frame, t) for t in trough_indices]
            freq_estimates = [self.SAMPLE_RATE/t for t in tau_candidates]
            tau_probs, unvoiced_prob = self.prob_thresholding(trough_indices, cmndf_frame)

            volume = np.sqrt(np.mean(frame ** 2)) # get volume as mean |amplitude| of the frame
            
            # create pitch objects and format into the pitch_list
            current_pitches = []
            for freq, prob in zip(freq_estimates, tau_probs):
                pitch = Pitch(time=time, frequency=freq, probability=prob, volume=volume, config=self.pitch_config)
                current_pitches.append(pitch)

            current_pitches.sort(key=lambda p: p.probability, reverse=True)
            pitches.append(current_pitches)

            # i = np.argmax(tau_probs)
            # best_prob = tau_probs[i]
            # best_freq = freq_estimates[i]
            # most_likely_pitch = Pitch(time=time, frequency=best_freq, probability=best_prob, volume=volume, config=self.pitch_config)

            # most_likely_pitches.append(most_likely_pitch)
            
            
        print('\nDone!')

        return pitches