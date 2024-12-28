
import numpy as np
import pandas as pd
import librosa
import scipy
from dtw import dtw, symmetric1
import pretty_midi
from typing import Optional, List
from dataclasses import dataclass, field
import warnings

from app.core.audio.AudioData import AudioData
from app.core.midi.MidiData import MidiData
from app.config import AppConfig
from app.algorithms.pitch.Pitch import Pitch
from app.algorithms.align.OnsetDf import UserOnsetDf, MidiOnsetDf
from app.algorithms.align.CQT import CQT
    
class DTW:
    def __init__(self):
        pass

    def align(user_cqt: np.ndarray, midi_cqt: np.ndarray):
        """
        Align two continuous sequences of CQT arrays corresponding to the user's audio and midi audio.
        Creates a distance matrix with the two CQT's and uses dtw-python module to align.

        Args:
            user_cqt: Array of user CQT vectors corresponding to entire audio sequence to align
            midi_cqt: Synthesized midi -> CQT vectors to align onto the user's audio
        """
        # Create distance matrix
        distance_matrix = scipy.spatial.distance.cdist(midi_cqt.T, user_cqt.T, metric='cosine')

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
    def parse_alignment(alignment, midi_data):
        """
        Convert the alignment result into a midi file. Based on the original MIDI data.
        Writes the output to 'aligned.mid' in the current working directory.

        Args:
            alignment: Result from dtw. Contains two arrays .index1 (midi cqt index)
                       and .index2 (user cqt index) which can subset into midi_CQT
            midi_data: Original midi data of the piece the user was trying to play
                       
        """
        aligned_midi = pretty_midi.PrettyMIDI()
        VIOLIN_PROGRAM = 41
        SAMPLE_RATE = 44100
        violin_instrument = pretty_midi.Instrument(program=VIOLIN_PROGRAM, is_drum=False, name='Violin')

        for i, note_row in midi_data.pitch_df.iterrows():
            start_time = note_row['start']
            start_cqt_idx = int(start_time * SAMPLE_RATE / CQT.HOP_LENGTH)
            
            align_idx = (np.abs(alignment.index1 - start_cqt_idx)).argmin()
            warped_start_time = alignment.index2[align_idx] * CQT.HOP_LENGTH / SAMPLE_RATE

            # Computing the warped end time
            if i == len(midi_data.pitch_df) - 1:
                warped_end_time = alignment.index2[-1] * CQT.HOP_LENGTH / SAMPLE_RATE
            else:
                next_start_time = midi_data.pitch_df.iloc[i+1]['start']
                next_cqt_idx = int(next_start_time * SAMPLE_RATE / CQT.HOP_LENGTH)
                next_align_idx = (np.abs(alignment.index1 - next_cqt_idx)).argmin()
                warped_end_time = alignment.index2[next_align_idx] * CQT.HOP_LENGTH / SAMPLE_RATE

            velocity = int(note_row['velocity'])
            pitch = int(note_row['pitch'])
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=warped_start_time, end=warped_end_time)
            violin_instrument.notes.append(note)

            # print(f"cqt_idx{start_cqt_idx} is at index {align_idx} in alignment")

        aligned_midi.instruments.append(violin_instrument)
        aligned_midi.write("aligned.mid")

        print("Wrote alignment result to aligned.mid.")
    
    @staticmethod
    def align_df(alignment, user_onset_df: UserOnsetDf, midi_onset_df: MidiOnsetDf):
        """Create a dataframe parsing the alignment result into more meaningful results."""
        aligned_user = user_onset_df.onset_df.iloc[alignment.index2].reset_index(drop=True)
        aligned_midi = midi_onset_df.onset_df.iloc[alignment.index1].reset_index(drop=True)

        flat_align_df = pd.DataFrame({
            'midi_time': aligned_midi['time'],
            'user_time': aligned_user['time']
        })

        grouped = flat_align_df.groupby('midi_time')['user_time'].apply(list).reset_index()
        align_df = pd.DataFrame({
            'midi_time': grouped['midi_time'],
            'user_time': grouped['user_time'].apply(lambda x: x if len(x) > 0 else [None])
        })

        # Initialize the 'user_midi_pitches' column with empty lists
        align_df['user_midi_nums'] = [[] for _ in range(len(align_df))]

        # Get the closest 'midi_pitch' in user_features.pitch_df to each user_time in the user_times array
        for i, user_times in align_df.iterrows():
            align_df.at[i, 'user_midi_nums'] = [user_onset_df.pitch_df.find_closest_pitch(t) for t in user_times['user_time']]

        return align_df