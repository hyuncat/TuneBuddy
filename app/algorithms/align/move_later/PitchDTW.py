
import numpy as np
import pandas as pd
import librosa
import scipy
from dtw import *
from dataclasses import dataclass, field
import pretty_midi

from app.core.audio.AudioData import AudioData
from app.core.midi.MidiData import MidiData
from typing import Optional, List
from app.config import AppConfig
from archive.PitchAnalyzer import PitchAnalyzer
from app.algorithms.pitch.PYin import PYin


@dataclass
class UserFeatures():
    # Original features (pre-dtw)
    cqt_df: Optional[pd.DataFrame] = field(default=None)
    pitch_df: Optional[pd.DataFrame] = field(default=None)
    note_df: Optional[pd.DataFrame] = field(default=None)
    onset_times: Optional[List] = field(default=None)
    onset_df: Optional[pd.DataFrame] = field(default=None)
    # Alignment results (post-dtw)

    def resegment_notes(self, window_size=15, threshold=0.5):
        pitch_analyzer = PitchAnalyzer()
        self.note_df = pitch_analyzer.note_segmentation(self.pitch_df, window_size, threshold)
        self.onset_df = PitchDTW.combined_onset_df(self, onset_threshold=0.01)
    

@dataclass
class MidiFeatures():
    pitch_df: Optional[pd.DataFrame] = field(default=None)
    cqt_df: Optional[pd.DataFrame] = field(default=None)
    onset_df: Optional[pd.DataFrame] = field(default=None)


class PitchDTW:
    MIN_VIOLIN_FREQ = 196
    N_BINS = 48 # 4 octaves, 12 bins / octave
    HOP_LENGTH = 1024
    TUNING = 0.0
    pitch_analyzer = PitchAnalyzer()

    def __init__(self):
        pass

    @staticmethod
    def extract_cqt(audio_data: np.ndarray, sample_rate=None) -> pd.DataFrame:
        """
        Compute CQT of an audio signal using librosa, perform log-amplitude scaling,
        normalize, and return the processed CQT and its frame times.

        (Code adopted from Raffel, Ellis 2016)

        @param:
            - audio_data: Audio data to compute CQT of
            - sample_rate: Sample rate of the audio data (default: 44100)

        @return: a DTWFeatures object with the following attributes:
            - cqt: CQT of the supplied audio data
            - frame_times: Times, in seconds, of each frame in the CQT
        """

        if sample_rate is None:
            sample_rate = AppConfig.SAMPLE_RATE

        # Compute CQT
        cqt = librosa.cqt(
            audio_data, 
            sr = sample_rate,
            fmin = PitchDTW.MIN_VIOLIN_FREQ, 
            n_bins = PitchDTW.N_BINS, 
            hop_length = PitchDTW.HOP_LENGTH, 
            tuning = PitchDTW.TUNING
        )

        # Compute the time of each frame
        times = librosa.frames_to_time(
            np.arange(cqt.shape[1]), 
            sr=AppConfig.SAMPLE_RATE, 
            hop_length=PitchDTW.HOP_LENGTH
        )
        
        cqt = cqt.astype(np.float32) # Store CQT as float32 to save space/memory

        # Compute log-amplitude + normalize the CQT
        # (From Raffel, Ellis 2016)
        cqt = librosa.amplitude_to_db(cqt, ref=cqt.max())
        cqt = librosa.util.normalize(cqt, norm=2)

        cqt_df = pd.DataFrame({
            'time': times,
            'cqt_norm': [np.array(c) for c in cqt.T]
        })

        return cqt_df
    
    @staticmethod
    def combined_onset_df(user_features: UserFeatures, onset_threshold=0.01):
        """Onset threshold is the maximum difference between the onset time and the 
        user note time to be considered the same onset"""

        # Combine and sort the 'time' columns from both pitch_diff/onset DataFrames
        onset_df = pd.DataFrame({'time': user_features.onset_times})
        combined_times = pd.concat(
            [user_features.note_df['time'], 
             onset_df['time']]
        ).sort_values().reset_index(drop=True)

        # Remove duplicates within the tolerance
        unique_times = [combined_times[0]]
        for time in combined_times[1:]:
            if not any(np.isclose(time, unique_times, atol=onset_threshold)):
                unique_times.append(time)

        # Create a new DataFrame with combined + unique times
        combined_onset_df = pd.DataFrame({'time': unique_times})

        # Lambda function for finding any df.time values that are close to t
        time_is_close = lambda t, df: any(np.isclose(t, df['time'], atol=onset_threshold))

        # Create boolean columns for whether the time is close to a user note or an onset
        combined_onset_df['pitch_diff'] = combined_onset_df['time'].apply(lambda t: time_is_close(t, user_features.note_df))
        combined_onset_df['onset'] = combined_onset_df['time'].apply(lambda t: time_is_close(t, onset_df))

        # Find the closest CQT frame to each time
        combined_onset_df['cqt_norm'] = combined_onset_df['time'].apply(
            lambda t: PitchDTW.find_closest_cqt_norm(t, user_features.cqt_df))

        return combined_onset_df

    @staticmethod
    def find_closest_cqt_norm(time, cqt_df):
        closest_time_idx = (np.abs(cqt_df['time'] - time)).idxmin()
        return cqt_df.loc[closest_time_idx, 'cqt_norm']

    @staticmethod
    def user_features(audio_data: AudioData, sample_rate=None) -> UserFeatures:
        """
        Extract user features from the audio data, including CQT, note segmentation, and onsets.
        """
        cqt_df = PitchDTW.extract_cqt(audio_data.data, sample_rate)
        pitch_df = PitchDTW.pitch_analyzer.user_pitchdf(audio_data)
        note_df = PitchDTW.pitch_analyzer.note_segmentation(pitch_df, window_size=15, threshold=0.5)
        onset_times = PitchDTW.pitch_analyzer.detect_onsets(audio_data)

        user_features = UserFeatures(
            cqt_df=cqt_df,
            pitch_df=pitch_df,
            note_df=note_df,
            onset_times=onset_times,
        )
        onset_df = PitchDTW.combined_onset_df(user_features, onset_threshold=0.01)
        user_features.onset_df = onset_df

        return user_features

    @staticmethod
    def midi_features(midi_data: MidiData, midi_audio: AudioData, sample_rate=None) -> MidiFeatures:
        """
        Extract MIDI features from the audio data, including CQT.
        """
        cqt_df = PitchDTW.extract_cqt(midi_audio.data, sample_rate)

        onset_df = pd.DataFrame({'time': midi_data.pitch_df['start']})
        onset_df['cqt_norm'] = onset_df['time'].apply(
            lambda t: PitchDTW.find_closest_cqt_norm(t, cqt_df))
        
        return MidiFeatures(pitch_df=midi_data.pitch_df, cqt_df=cqt_df, onset_df=onset_df)
    
    @staticmethod
    def align(user_features: UserFeatures, midi_features: MidiFeatures):
        # Create distance matrix
        user_cqt = np.stack(user_features.onset_df['cqt_norm'].values)
        midi_cqt = np.stack(midi_features.onset_df['cqt_norm'].values)

        distance_matrix = scipy.spatial.distance.cdist(midi_cqt, user_cqt, metric='cosine')

        window_args = {'window_size': 100}
        alignment = dtw(
            distance_matrix,
            keep_internals=True,
            step_pattern=symmetric1,
            window_type='sakoechiba',
            window_args=window_args
        )

        # Compute the mean alignment error
        mean_error = np.mean(np.abs(alignment.index1 - alignment.index2))

        # Print some information about the alignment
        print("DTW alignment computed.")
        print(f"Distance: {alignment.distance}") # unit = cosine distance
        print(f"Mean alignment error: {mean_error}") # unit = frames

        return alignment

    @staticmethod
    def align1_from_new(user_onset_df: UserOnsetDf, midi_onset_df: MidiOnsetDf):
        """Align a sequence of CQT vectors at the onsets"""
        # Create distance matrix
        user_cqt = np.stack(user_onset_df.onset_df['cqt_norm'].values)
        midi_cqt = np.stack(midi_onset_df.onset_df['cqt_norm'].values)

        distance_matrix = scipy.spatial.distance.cdist(midi_cqt, user_cqt, metric='cosine')

        # Compute the alignment
        window_args = {'window_size': 100}
        alignment = dtw(
            distance_matrix,
            keep_internals=True,
            step_pattern=symmetric1,
            window_type='sakoechiba',
            window_args=window_args
        )

        # Print some statistics (from dtw-python documentation)
        # Compute the mean alignment error
        mean_error = np.mean(np.abs(alignment.index1 - alignment.index2))

        # Print some information about the alignment
        print("DTW alignment computed.")
        print(f"Distance: {alignment.distance}") # unit = cosine distance
        print(f"Mean alignment error: {mean_error}") # unit = frames

        return alignment

    @staticmethod
    def align_df(alignment, user_features: UserFeatures, midi_features: MidiFeatures):
        aligned_user = user_features.onset_df.iloc[alignment.index2].reset_index(drop=True)
        aligned_midi = midi_features.onset_df.iloc[alignment.index1].reset_index(drop=True)

        flat_align_df = pd.DataFrame({
            'midi_time': aligned_midi['time'],
            'user_time': aligned_user['time']
        })

        # Group user_times by midi_time
        grouped = flat_align_df.groupby('midi_time')['user_time'].apply(list).reset_index()

        # Create a new DataFrame with distinct MIDI times and corresponding user times
        align_df = pd.DataFrame({
            'midi_time': grouped['midi_time'],
            'user_times': grouped['user_time'].apply(lambda x: x if len(x) > 0 else [None])
        })

        def find_closest_pitch(time, pitch_df):
            closest_time_idx = (np.abs(pitch_df['time'] - time)).idxmin()
            return pitch_df.loc[closest_time_idx, 'midi_pitch']

        # Initialize the 'user_midi_pitches' column with empty lists
        align_df['user_midi_pitches'] = [[] for _ in range(len(align_df))]

        # Get the closest 'midi_pitch' in user_features.pitch_df to each user_time in the user_times array
        for i, user_times in align_df.iterrows():
            align_df.at[i, 'user_midi_pitches'] = [find_closest_pitch(t, user_features.pitch_df) for t in user_times['user_times']]

        # Only keep the first user_time and user_midi_pitch which is closest to the MIDI pitch
        for row_index, midi_row in midi_features.pitch_df.iterrows():
            onset_time = midi_row.start
            # find the best user onset
            # align_df_index = (np.abs(align_df['midi_time'] - onset_time)).idxmin()
            align_df_row = align_df.iloc[row_index]
            
            best_user_time = align_df_row['user_times'][0]
            best_user_pitch = align_df_row['user_midi_pitches'][0]
            onset_idx = (np.abs(aligned_user['time'] - best_user_time)).idxmin()

            for user_midi_pitch, user_time in zip(align_df_row['user_midi_pitches'], align_df_row['user_times']):
                if abs(midi_row.pitch - user_midi_pitch) < 1:
                    best_user_time = user_time
                    best_user_pitch = user_midi_pitch
                    break
            
            onset_idx = (np.abs(aligned_user['time'] - best_user_time)).idxmin()
            best_user_time = aligned_user.iloc[onset_idx].time
            # closest_time_idx = (np.abs(user_features.pitch_df['time'] - best_user_time)).idxmin()
            # best_user_pitch = user_features.pitch_df.loc[closest_time_idx, 'midi_pitch']
            
            align_df.at[row_index, 'user_times'] = [best_user_time]
            align_df.at[row_index, 'user_midi_pitches'] = [best_user_pitch]

        return align_df

    @staticmethod
    def align_df2(alignment, user_features: UserFeatures, midi_features: MidiFeatures):
        """this is what we use. not the first one lol"""
        aligned_user = user_features.onset_df.iloc[alignment.index2].reset_index(drop=True)
        aligned_midi = midi_features.onset_df.iloc[alignment.index1].reset_index(drop=True)

        flat_align_df = pd.DataFrame({
            'midi_time': aligned_midi['time'],
            'user_time': aligned_user['time']
        })

        # Group user_times by midi_time
        grouped = flat_align_df.groupby('midi_time')['user_time'].apply(list).reset_index()

        # Create a new DataFrame with distinct MIDI times and corresponding user times
        align_df = pd.DataFrame({
            'midi_time': grouped['midi_time'],
            'user_times': grouped['user_time'].apply(lambda x: x if len(x) > 0 else [None])
        })

        def find_closest_pitch(time, pitch_df):
            closest_time_idx = (np.abs(pitch_df['time'] - time)).idxmin()
            return pitch_df.loc[closest_time_idx, 'midi_pitch']

        # Initialize the 'user_midi_pitches' column with empty lists
        align_df['user_midi_pitches'] = [[] for _ in range(len(align_df))]

        # Get the closest 'midi_pitch' in user_features.pitch_df to each user_time in the user_times array
        for i, user_times in align_df.iterrows():
            align_df.at[i, 'user_midi_pitches'] = [find_closest_pitch(t, user_features.pitch_df) for t in user_times['user_times']]

        return align_df
    
    @staticmethod
    def print_aligned_times(alignment, user_features, midi_features):
        aligned_user = user_features.onset_df.iloc[alignment.index2]
        aligned_midi = midi_features.onset_df.iloc[alignment.index1]

        print("Aligned times:")
        for i, (user_time, midi_time) in enumerate(zip(aligned_user['time'], aligned_midi['time'])):
            print(f"Note {i}: User time = {user_time:.2f}, MIDI time = {midi_time:.2f}")

    # @staticmethod
    # def export_midi2(align_df, alignment, user_features, midi_features, midi_data, output_file="aligned.mid"):
    #     aligned_midi = midi_features.onset_df.iloc[alignment.index1]
    #     aligned_user = user_features.onset_df.iloc[alignment.index2]

    #     aligned_prettymidi = pretty_midi.PrettyMIDI()

    #     VIOLIN_PROGRAM = 41
    #     violin_instrument = pretty_midi.Instrument(program=VIOLIN_PROGRAM, is_drum=False, name='Violin')

    #     for row_index, note in midi_data.pitch_df.iterrows():
            
    
    @staticmethod
    def export_midi(alignment, user_features, midi_features, midi_data, output_file="aligned.mid"):
        aligned_midi = midi_features.onset_df.iloc[alignment.index1]
        aligned_user = user_features.onset_df.iloc[alignment.index2]
        align_df = PitchDTW.align_df2(alignment, user_features, midi_features)

        aligned_prettymidi = pretty_midi.PrettyMIDI()

        VIOLIN_PROGRAM = 41
        violin_instrument = pretty_midi.Instrument(program=VIOLIN_PROGRAM, is_drum=False, name='Violin')

        align_df['user_times'] = align_df['user_times'].apply(lambda x: x[0])
        for row_index, note in midi_data.pitch_df.iterrows():
            onset_time = note.start
            # onset_idx = (np.abs(aligned_user['time'] - onset_time)).idxmin()

            # find the best onset index
            align_df_index = (np.abs(align_df['user_times'] - onset_time)).argmin()

            align_df_row = align_df.iloc[align_df_index]
            
            # find the best user time to match to the midi time
            warped_onset_time = align_df_row['user_times']
            print(f"best onset time: {warped_onset_time}")

            # for user_midi_pitch, user_time in zip(align_df_row['user_midi_pitches'], align_df_row['user_times']):
            #     if abs(note.pitch - user_midi_pitch) < 1:
            #         warped_onset_time = user_time
            #         break

            # Get the next onset_time of the MIDI note
            # and its index into aligned CQT array
            next_warped_onset_time = -1
            LAST_PITCHDF_ROW_INDEX = midi_data.pitch_df.shape[0] - 1 
            if row_index == LAST_PITCHDF_ROW_INDEX:
                next_warped_onset_time = aligned_user.iloc[-1].time + .1
                next_onset_time = next_warped_onset_time
            else:
                next_align_df_row = align_df.iloc[align_df_index+1]
                next_onset_time = next_align_df_row['midi_time']

                # find the best user time to match to the midi time
                next_warped_onset_time = next_align_df_row['user_times']

            # Compute warped note duration using the ratio of the aligned / original 
            # 'internote' durations between two onset times, then use to scale the 
            # original note duration.

            original_internote_duration = next_onset_time - onset_time
            aligned_internote_duration = next_warped_onset_time - warped_onset_time
            if original_internote_duration == 0:
                warp_ratio = 0
            else:
                warp_ratio = aligned_internote_duration / original_internote_duration

            original_note_duration = note.duration
            warped_note_duration = original_note_duration * warp_ratio

            # Use the previous pitch and velocity values
            pitch = int(note.pitch)
            velocity = int(note.velocity)

            print(f"adding new note at {warped_onset_time}")
            new_note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=warped_onset_time, end=warped_onset_time+original_note_duration)
            violin_instrument.notes.append(new_note)

        aligned_prettymidi.instruments.append(violin_instrument)
        aligned_prettymidi.write(output_file)
        print(f"Aligned MIDI written to {output_file}")