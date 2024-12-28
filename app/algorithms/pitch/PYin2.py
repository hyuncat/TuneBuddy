import numpy as np
import scipy.signal
import scipy.stats

from app.algorithms.pitch.Yin import Yin
from app.algorithms.pitch.Pitch import PitchConversion, PitchConfig, Pitch
from app.config import AppConfig

class PYin:
    FRAME_SIZE = 4097
    HOP_SIZE = 128
    UNVOICED_PROBABILITY = 0.01
    N_THRESHOLDS = 100

    def __init__(self):
        pass

    def probabilistic_thresholding():
        pass

    @staticmethod
    def threshold_prior(n_thresholds: int=100, a: float=2, b: float=34/3):
        """Returns a beta distribution modeling the pdf of thresholds for PYIN"""
        # all thresholds
        x = np.linspace(0, 1, n_thresholds+1)
        x = x[1:] # remove the 0 threshold
        beta_pdf = scipy.stats.beta.pdf(x, a, b) / n_thresholds
        return beta_pdf

    @classmethod
    def pyin(cls, audio_data: np.ndarray, sr: int=AppConfig.SAMPLE_RATE, f0_min: float=196.0, tuning: float=442):
        """PYIN algorithm from Mauch, Dixon"""
        # --- PARAMS ---
        # ensure max lag is big enough to detect lowest f0 (largest period)
        # defaults to violin min
        tau_max = int(sr / f0_min) 
        # pitch config to associate tuning / fmin / fmax params for each pitch
        pitch_config = PitchConfig(tuning=tuning, fmin=f0_min)

        # create beta distribution
        beta_pdf = PYin.threshold_prior(n_thresholds=cls.N_THRESHOLDS)

        # get memory efficient frames with np pointer c++ magic
        frames = np.lib.stride_tricks.sliding_window_view(audio_data, cls.FRAME_SIZE)[::cls.HOP_SIZE]
        n_frames = 1 + (len(audio_data) - cls.FRAME_SIZE) // cls.HOP_SIZE # just for printing

        pitches = []
        for i, frame in enumerate(frames):
            print(f"\rProcessing frame {i+1}/{n_frames}", end='')

            time = (i*cls.HOP_SIZE)/sr # elapsed time of the frame

            diff_frame, amplitudes = Yin.difference_function(frame, tau_max)
            cmndf_frame = Yin.cmndf(diff_frame, tau_max)

            diff_trough_indices = scipy.signal.argrelmin(diff_frame, order=1)[0]
            
            # parabolic interpolation for final candidates
            tau_candidates = [Yin.parabolic_interpolation(diff_frame, t) for t in diff_trough_indices]
            freq_estimates = [sr/t for t in tau_candidates]

            # fill in tau_probs with weighted probabilities (probabilistic thresholding)
            tau_probs = np.zeros_like(tau_candidates)
            for s_i in range(1, cls.N_THRESHOLDS):
                tau_0, is_voiced = Yin.absolute_threshold(diff_frame, diff_trough_indices, threshold=s_i)
                tau_idx = np.where(diff_trough_indices == tau_0)[0]
                if is_voiced:
                    tau_probs[tau_idx] += beta_pdf[s_i]
                else:
                    tau_probs[tau_idx] += beta_pdf[s_i] * cls.UNVOICED_PROBABILITY
            
            unvoiced_prob = 1 - np.sum(tau_probs)

            volume = np.sqrt(np.mean(frame ** 2))
            
            for freq, prob in zip(freq_estimates, tau_probs):
                pitch = Pitch(time=time, frequency=freq, probability=prob, volume=volume, config=pitch_config)
                pitches.append(pitch) 
            
            
        print('\nDone!')
        return pitches