import essentia.standard as es
import essentia
import numpy as np
import pandas as pd
from math import ceil

from app.core.audio.AudioData import AudioData
from app.core.midi.MidiData import MidiData 
from app.core.recording.PitchDf import PitchDf
from app.algorithms.pitch.Pitch import Pitch
from app.config import AppConfig
from app.algorithms.align.CQT import CQT


class OnsetDf:
    """
    Base class for handling onset data, containing common attributes and methods 
    like 'audio_data', 'onset_df', and the 'add_cqt_column()' method.
    """

    def __init__(self, audio_data: AudioData):
        self.audio_data = audio_data
        self.onset_df = None  # This will store the combined onset times in subclasses

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def add_cqt_column(self):
        """
        Adds a column containing the CQT norm vector at the frame of audio corresponding 
        to onset time. Requires onset_df be non-none.
        """
        if not hasattr(self, "onset_df") or self.onset_df is None:
            print("No onset_df available to add cqt column to. Please try again.")
            return

        self.onset_df['cqt_norm'] = None
        for i, onset in self.onset_df.iterrows():
            onset_time = onset['time']
            onset_idx = int(onset_time * AppConfig.SAMPLE_RATE)

            FRAME_SIZE = int(AppConfig.FRAME_SIZE)
            audio_frame = self.audio_data.data[onset_idx:onset_idx + FRAME_SIZE]

            cqt_norm = CQT.extract_cqt_frame(audio_frame)
            self.onset_df.loc[i, 'cqt_norm'] = cqt_norm

    
    # unused - not working yet / no noticeable difference
    def add_cqt_column2(self):
        """
        Adds a column containing the element-wise mean CQT norm vector at the frame of audio corresponding 
        to onset time. Chooses the two vectors with the greatest cosine similarity within the period 
        between the current onset and the next onset.
        """
        if not hasattr(self, "onset_df") or self.onset_df is None:
            print("No onset_df available to add cqt column to. Please try again.")
            return

        self.onset_df['cqt_norm'] = None

        FRAME_SIZE = int(AppConfig.FRAME_SIZE)
        SAMPLE_RATE = AppConfig.SAMPLE_RATE
        HOP_LENGTH = CQT.HOP_LENGTH

        # Step 1: Compute the CQT for the entire audio once
        full_cqt = CQT.extract_cqt(self.audio_data.data, sample_rate=SAMPLE_RATE)

        # Step 2: Compute the onset indices in terms of frames
        onset_frames = []
        for onset_time in self.onset_df['time']:
            onset_idx = int(onset_time * SAMPLE_RATE / HOP_LENGTH)  # Convert time to frame index
            onset_frames.append(onset_idx)

        # Step 3: Extract the CQT vectors at onset frames
        cqt_vectors = [full_cqt[:, frame_idx] for frame_idx in onset_frames]

        # Step 4: Compute the cosine similarity between vectors within each onset window
        for i in range(len(onset_frames) - 1):
            onset_start_idx = onset_frames[i]
            onset_end_idx = onset_frames[i + 1]

            # Extract CQT vectors within this window (between current and next onset)
            window_cqt_vectors = full_cqt[:, onset_start_idx:onset_end_idx]

            # Initialize to store the highest similarity and the corresponding vector indices
            max_similarity = -1
            idx1, idx2 = None, None

            # Compare all pairs of CQT vectors within the window
            for j in range(window_cqt_vectors.shape[1] - 1):
                for k in range(j + 1, window_cqt_vectors.shape[1]):
                    similarity = cosine_similarity(window_cqt_vectors[:, j], window_cqt_vectors[:, k])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        idx1, idx2 = j, k

            # Step 5: Compute the element-wise mean of the two vectors with the highest similarity
            if idx1 is not None and idx2 is not None:
                cqt_mean = (window_cqt_vectors[:, idx1] + window_cqt_vectors[:, idx2]) / 2
                self.onset_df.at[i, 'cqt_norm'] = cqt_mean



class UserOnsetDf(OnsetDf):
    """
    Subclass for user audio data, extending OnsetDf. This includes methods for 
    pitch change detection and combining pitch-based onsets with detected onsets.
    """

    def __init__(self, audio_data: AudioData, pitch_list: list[Pitch]):
        super().__init__(audio_data)
        self.pitch_list = pitch_list

        # Essentia onset detection helpers
        self.w = es.Windowing(type='hann')
        self.fft = es.FFT()
        self.c2p = es.CartesianToPolar()
        self.od_complex = es.OnsetDetection(method='complex')

        self.init_onset_df(audio_data, pitch_list)

    def init_onset_df(self, audio_data: AudioData, pitch_list: list[Pitch]):
        self.detect_onsets(audio_data)
        self.detect_pitch_changes(pitch_list=pitch_list, window_size=30, threshold=0.6)
        self.combine_onsets(combine_threshold=0.05)
        # self.add_cqt_column() # inherited from OnsetDf
            
    def detect_onsets(self, audio_data: AudioData) -> None:
        """
        Detects onsets in the audio data using Essentia's complex onset 
        detection algorithm. Updates self.onset_times in place (incase of recomputing onsets).

        Args:
            audio_data (AudioData): User's audio data to detect onsets from
        """
        print("Detecting onsets...", end='', flush=True)

        # Parameters from Essentia tutorial - seem optimized enough, but can experiment later.
        FRAME_SIZE = 1024
        HOP_SIZE = 512

        # Compute both ODF frame by frame. Store results to a Pool.
        pool = essentia.Pool()
        for frame in es.FrameGenerator(audio_data.data, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
            magnitude, phase = self.c2p(self.fft(self.w(frame)))
            odf_value = self.od_complex(magnitude, phase)
            pool.add('odf.complex', odf_value)
            # print(f"Frame {i}: Time {frame_time:.4f}s, ODF Value {odf_value}")

        # Detect onset locations
        onsets = es.Onsets()
        onset_times = onsets(essentia.array([pool['odf.complex']]), [1])
        print(" Done!")
        self.onset_times = onset_times

    def update_pitches(self, pitch_list: list[Pitch]):
        """Updates the current list of pitches to compute pitch changes with."""
        self.pitch_list = pitch_list
        
    def detect_pitch_changes(self, pitch_list: list[Pitch]=None, window_size: int=30, threshold: float=0.6):
        """
        Detect different-enough new pitches based on a rolling median of pitch_bin.
        We compare the current rolling median to the next one, and keep track of when 
        the difference exceeds [threshold]. 
        
        Updates self.pitch_change_times in place, creating a `np.ndarray[float]`
        with timestamps(sec) of each detected pitch change.

        Args:
            pitch_list: A list of Pitch objects (if none, uses self.pitch_list)
            window_size: Size of the rolling window to compute the median pitch. 
            threshold: Min difference between the current and next rolling median 
                       pitch needed to detect a new pitch change. 
        """
        print(f"Detecting pitch changes with rolling median window_size={window_size} and threshold={threshold}...", end='', flush=True)

        if pitch_list is not None:
            self.pitch_list = pitch_list
        elif not hasattr(self, "pitches"):
            print("Error: Didn't supply pitches and trying to detect pitch changes. Please try again with different args.")
            return

        # Rolling median on pitch_bin
        rolling_medians = pd.Series([pitch.midi_num for pitch in self.pitch_list]).rolling(window=window_size).median()

        # Prepare list for onset times
        pitch_change_times = []

        MAIN_LOWER_BOUND = window_size - 1
        MAIN_UPPER_BOUND = len(rolling_medians) - window_size - 1
        HOP_SIZE = window_size # Compare this window to next window with no overlap

        last_pitch_change = 0

        for i in range(MAIN_LOWER_BOUND, MAIN_UPPER_BOUND, HOP_SIZE):
            current_median = rolling_medians.iloc[i]
            next_median = rolling_medians.iloc[i + HOP_SIZE]

            if abs(next_median - current_median) >= threshold:
                # Subroutine to find the best onset within the window
                # based on the two pitches with the greatest difference
                #TODO: Make subroutine resistant to octave jumps
                WINDOW_LOWER_BOUND = i - ceil(window_size / 2)
                WINDOW_UPPER_BOUND = i + ceil(window_size / 2)
                largest_diff = (-1, 0)  # (index, difference)
                
                for j in range(WINDOW_LOWER_BOUND, WINDOW_UPPER_BOUND):
                    pitch_diff = abs(self.pitch_list[j].midi_num - self.pitch_list[j + 1].midi_num)
                    if pitch_diff > largest_diff[1]:
                        largest_diff = (j, pitch_diff)

                # Append the note data
                # notes.append(current_median)
                pitch_change_times.append(self.pitch_list[last_pitch_change].time)  # Time of the last onset

                # Update last pitch_change time
                last_pitch_change = largest_diff[0]

        print(" Done!")
        self.pitch_change_times = np.array(pitch_change_times)

    def combine_onsets(self, combine_threshold=0.05) -> None:
        """
        Combines self.onset_times and self.pitch_change_times into a single onset_df
        Args:
            onsets: Array of onset times from Essentia
            note_df: Note_df of segmented notes based on pitch changes
            combine_threshold: The max difference (s) between the onset time and the 
                               user note time to be considered the same onset
        """
        missing_attr = not hasattr(self, "onset_times") or not hasattr(self, "pitch_change_times")
        none_attr = self.onset_times is None or self.pitch_change_times is None

        if missing_attr or none_attr:
            print("OnsetData is missing either onset_times or pitch_change_times. Please compute them and try again.")
            return
        
        # Combine and sort the 'time' columns from both pitch_diff/onset DataFrames
        # print(f"onset_times type: {type(self.onset_times)}")
        # print(f"pitch_change_times type: {type(self.pitch_change_times)}")
        onset_df = pd.DataFrame({'time': self.onset_times})
        pitch_change_df = pd.DataFrame({'time': self.pitch_change_times})

        combined_times = pd.concat(
            [pitch_change_df['time'], 
             onset_df['time']]
        ).sort_values().reset_index(drop=True)

        # Remove duplicates within the tolerance
        unique_times = [combined_times[0]]
        for time in combined_times[1:]:
            if not any(np.isclose(time, unique_times, atol=combine_threshold)):
                unique_times.append(time)

        # Create a new DataFrame with combined + unique times
        combined_onset_df = pd.DataFrame({'time': unique_times})

        # Lambda function for finding any df.time values that are close to t
        time_is_close = lambda t, df: any(np.isclose(t, df['time'], atol=combine_threshold))

        # Create boolean columns for whether the time is close to a user note or an onset
        combined_onset_df['pitch_diff'] = combined_onset_df['time'].apply(lambda t: time_is_close(t, pitch_change_df))
        combined_onset_df['onset'] = combined_onset_df['time'].apply(lambda t: time_is_close(t, onset_df))

        self.onset_df = combined_onset_df


class MidiOnsetDf(OnsetDf):
    """
    Subclass for MIDI data. Since the onset_df and audio_data are shared, we
    inherit from OnsetDf. Onset_df is created based on midi_data.pitch_df.
    """
    def __init__(self, user_audio_data: AudioData, midi_audio_data: AudioData, midi_data: MidiData):
        super().__init__(midi_audio_data)
        self.midi_data = midi_data
        self.user_audio_data = user_audio_data
        self.init_onset_df()

    def adjust_timing(self):
        user_length = self.user_audio_data.get_length()
        midi_length = self.audio_data.get_length()
        self.time_adjust = user_length / midi_length
        
    def init_onset_df(self):
        # self.adjust_timing()
        self.onset_df = pd.DataFrame({'time': self.midi_data.pitch_df['start']})
        self.add_cqt_column()