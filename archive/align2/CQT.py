import numpy as np
import pandas as pd
import librosa
# import scipy
# from dtw import *
# import pretty_midi
# from typing import Optional, List
# from dataclasses import dataclass, field
import warnings

# from app.core.audio.AudioData import AudioData
# from app.core.midi.MidiData import MidiData
from app.config import AppConfig
# from app.algorithms.pitch.Pitch import Pitch

class CQT:
    MIN_VIOLIN_FREQ = 196
    N_BINS = 60 # 5 octaves, 12 bins / octave
    HOP_LENGTH = 1024
    FRAME_SIZE = AppConfig.FRAME_SIZE
    TUNING = 0.0

    def __init__(self):
        pass

    @staticmethod
    def extract_cqt_frame(audio_frame: np.ndarray, sample_rate=None) -> pd.DataFrame:
        """
        Compute CQT of an audio signal on a single frame, perform log-amplitude scaling,
        normalize, and return the CQT vector for the frame.

        Args:
            audio_data (np.ndarray): Frame of audio data to compute CQT of
            sample_rate (int, optional): Sample rate of the audio data. Defaults to AppConfig.SAMPLE_RATE.

        Returns:
            pd.DataFrame: A DataFrame containing the time of each frame and the averaged cqt_norm for each frame.
        """

        if sample_rate is None:
            sample_rate = AppConfig.SAMPLE_RATE

        # Suppress UserWarning and ComplexWarning
        warnings.filterwarnings('ignore')

        # Compute CQT
        cqt = librosa.cqt(
            audio_frame, 
            sr=sample_rate,
            fmin=CQT.MIN_VIOLIN_FREQ, 
            n_bins=CQT.N_BINS, 
            hop_length=len(audio_frame)+2, # used to be dtw.hop_size
            tuning=CQT.TUNING
        )

        # Convert CQT to float32 to save space
        cqt = cqt.astype(np.float32)

        # Compute log-amplitude and normalize the CQT
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt_norm = librosa.util.normalize(cqt_db, norm=2)

        return cqt_norm

    @staticmethod
    def extract_cqt(audio_data: np.ndarray, sample_rate=None) -> pd.DataFrame:
        """
        Compute CQT on an entire audio signal.
        Performs log-amplitude scaling, normalizes, and return a continuous array 
        of CQT vectors corresponding to the audio data.

        Args:
            audio_data (np.ndarray): Audio signal array to compute CQT on
            sample_rate (int, optional): Sample rate of the audio data. Defaults to AppConfig.SAMPLE_RATE.

        Returns:
            pd.DataFrame: A DataFrame containing the time of each frame and the averaged cqt_norm for each frame.
        """

        if sample_rate is None:
            sample_rate = AppConfig.SAMPLE_RATE

        # Suppress UserWarning and ComplexWarning
        warnings.filterwarnings('ignore')

        # Compute CQT
        cqt = librosa.cqt(
            audio_data, 
            sr=sample_rate,
            fmin=CQT.MIN_VIOLIN_FREQ, 
            n_bins=CQT.N_BINS, 
            hop_length=CQT.HOP_LENGTH,
            tuning=CQT.TUNING
        )

        # Convert CQT to float32 to save space
        cqt = cqt.astype(np.float32)

        # Compute log-amplitude and normalize the CQT
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt_norm = librosa.util.normalize(cqt_db, norm=2)

        return cqt_norm