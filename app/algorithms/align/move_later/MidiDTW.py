import librosa
import pretty_midi
import soundfile as sf
import numpy as np
import pandas as pd
import scipy
from dtw import *
from dataclasses import dataclass, field
from typing import Optional

from app.config import AppConfig


@dataclass
class CQTFeatures:
    """
    Class for storing the CQT and the associated times, as well as their 
    aligned variants for a given audio signal.
    
    (e.g., For a synthesized MIDI or user recording)
    
    @attr:
        - cqt: CQT of the supplied audio data
        - times: Times, in seconds, of each frame in the CQT
        - aligned_cqt: Aligned CQT 
        - aligned_times: Aligned times
    """
    cqt: np.ndarray
    times: np.ndarray
    aligned_cqt: Optional[np.ndarray] = field(default=None)
    aligned_times: Optional[np.ndarray] = field(default=None)


class MidiDTW:
    """
    Class for preprocessing MIDI/audio files for dynamic time warping.
    """

    FS_SAMPLE_RATE = AppConfig.SAMPLE_RATE  
    SOUNDFONT = 'data/MuseScore_General.sf3'  # Default soundfont path

    MIN_VIOLIN_FREQ = 196
    N_BINS = 48 # 4 octaves, 12 bins / octave
    HOP_LENGTH = 1024
    TUNING = 0.0

    # Define the step pattern 
    STEP_PATTERN = symmetric1

    # Define the Sakoe-Chiba band
    WINDOW_TYPE = 'sakoechiba'
    WINDOW_SIZE = 9 # tunable

    @classmethod
    def update_soundfont(cls, soundfont_path):
        """
        Update the class variable for the soundfont path.
        """
        cls.SOUNDFONT = soundfont_path

    @staticmethod
    def midi_to_audio(midi_file_path: str, save_audio: bool=False) -> np.ndarray:
        """
        Convert MIDI file to np.array of audio signals using the class 
        FS_SAMPLE_RATE and SOUNDFONT variables.
        """
        midi_obj = pretty_midi.PrettyMIDI(midi_file_path)
        midi_audio = midi_obj.fluidsynth(fs=MidiDTW.FS_SAMPLE_RATE, sf2_path=MidiDTW.SOUNDFONT)

        # Write the audio to a file (optional)
        if save_audio:
            converted_midi_filepath = midi_file_path.replace('.mid', '.mp3')
            sf.write(converted_midi_filepath, midi_audio, MidiDTW.FS_SAMPLE_RATE)

        return midi_audio
    
    @staticmethod
    def load_audio(audio_file_path: str) -> np.ndarray:
        """
        Load an audio file from a file path and return it as a numpy array.
        """
        user_audio, _ = librosa.load(audio_file_path, sr=MidiDTW.FS_SAMPLE_RATE)
        return user_audio
    
    @staticmethod
    def extract_cqt(audio_data: np.ndarray, sample_rate=None) -> CQTFeatures:
        """
        Compute CQT of an audio signal using librosa, perform log-amplitude scaling,
        normalize, and return the processed CQT and its frame times.

        (Code adopted from Raffel, Ellis 2016)

        @param:
            - audio_data: Audio data to compute CQT of
            - sample_rate: Sample rate of the audio data (default: 22050)

        @return: a DTWFeatures object with the following attributes:
            - cqt: CQT of the supplied audio data
            - frame_times: Times, in seconds, of each frame in the CQT
        """

        if sample_rate is None:
            sample_rate = MidiDTW.FS_SAMPLE_RATE

        # Compute CQT
        cqt = librosa.cqt(
            audio_data, 
            sr = sample_rate,
            fmin = MidiDTW.MIN_VIOLIN_FREQ, 
            n_bins = MidiDTW.N_BINS, 
            hop_length = MidiDTW.HOP_LENGTH, 
            tuning = MidiDTW.TUNING
        )

        # Compute the time of each frame
        times = librosa.frames_to_time(
            np.arange(cqt.shape[1]), 
            sr=MidiDTW.FS_SAMPLE_RATE, 
            hop_length=MidiDTW.HOP_LENGTH
        )
        
        cqt = cqt.astype(np.float32) # Store CQT as float32 to save space/memory

        # Compute log-amplitude + normalize the CQT
        # (From Raffel, Ellis 2016)
        cqt = librosa.amplitude_to_db(cqt, ref=cqt.max())
        cqt = librosa.util.normalize(cqt, norm=2)

        cqt_features = CQTFeatures(cqt=cqt.T, times=times)
        return cqt_features
    
    @staticmethod
    def midi_dtw(midi_cqt: CQTFeatures, user_cqt: CQTFeatures) -> tuple[CQTFeatures, CQTFeatures]:
        """
        Compute the dynamic time warping distance between two CQTs.
        @param:
            - midi_cqt: CQT of the MIDI data
            - user_cqt: CQT of the user's audio data
        @return:
            - alignment: Dynamic time warping alignment object between the two CQTs
        """
        distance_matrix = scipy.spatial.distance.cdist(midi_cqt.cqt, user_cqt.cqt, metric='cosine')

        window_args = {'window_size': MidiDTW.WINDOW_SIZE}

        # Apply DTW on the distance matrix with the chosen step pattern and window
        alignment = dtw(
            distance_matrix, 
            keep_internals=True, 
            step_pattern=MidiDTW.STEP_PATTERN, 
            window_type=MidiDTW.WINDOW_TYPE, 
            window_args=window_args
        )

        # Extract aligned CQT features using the alignment path
        midi_cqt.aligned_cqt = midi_cqt.cqt[alignment.index1] 
        midi_cqt.aligned_times = midi_cqt.times[alignment.index1]

        user_cqt.aligned_cqt = user_cqt.cqt[alignment.index2] 
        user_cqt.aligned_times = user_cqt.times[alignment.index2]

        # Compute the mean alignment error
        mean_error = np.mean(np.abs(alignment.index1 - alignment.index2))

        # Print some information about the alignment
        print("DTW alignment computed.")
        print(f"Distance: {alignment.distance}") # unit = cosine distance
        print(f"Mean alignment error: {mean_error}") # unit = frames

        return midi_cqt, user_cqt


    def print_aligned_times(midi_cqt: CQTFeatures, user_cqt: CQTFeatures) -> None:
        """Utility function for sanity checking"""

        print("Comparing the aligned MIDI times to the template user times\n---")
        for midi_time, user_time in zip(midi_cqt.aligned_times, user_cqt.aligned_times):
            print(f"MIDI time: {midi_time}, User time: {user_time}")


    def align_midi(midi_cqt: CQTFeatures, user_cqt: CQTFeatures, pitch_df: pd.DataFrame, print_debug: bool=False) -> pretty_midi.PrettyMIDI:
        """
        Use the aligned MIDI / user audio times to create a new MIDI object 
        with notes aligned to the user audio.

        For now is hard-coded to create a violin instrument, but can be easily extended.

        @param:
            - midi_cqt: CQTFeatures object for the MIDI data (filled with alignments)
            - user_cqt: CQTFeatures object for the user audio data (filled with alignments)
            - pitch_df: pd.DataFrame of MIDI note information
            - print_debug: bool, whether to print debug information

        @return:
            - aligned_midi: PrettyMIDI object with the aligned violin notes
        """

        # If the CQTs are not aligned, raise an error
        if midi_cqt.aligned_cqt is None or user_cqt.aligned_cqt is None:
            raise ValueError("CQTs are not aligned. Run the DTW function first.")

        aligned_midi = pretty_midi.PrettyMIDI()
        
        VIOLIN_PROGRAM = 41
        violin_instrument = pretty_midi.Instrument(program=VIOLIN_PROGRAM, is_drum=False, name='Violin')

        for row_index, note in pitch_df.iterrows():
            # Get the original onset_time of the MIDI note 
            # and its index into aligned midi CQT array
            onset_time = note.start
            onset_idx = np.argmin(np.abs(midi_cqt.aligned_times - onset_time))
            
            # Get the next onset_time of the MIDI note
            # and its index into aligned CQT array
            LAST_PITCHDF_ROW_INDEX = pitch_df.shape[0] - 1 
            if row_index == LAST_PITCHDF_ROW_INDEX:
                next_onset_time = midi_cqt.aligned_times[-1] # Get the length of the midi_audio
            else:
                next_onset_time = pitch_df.iloc[row_index + 1].start

            next_onset_idx = np.argmin(np.abs(midi_cqt.aligned_times - next_onset_time))

            # Get the aligned onset times for the MIDI note
            # by finding the corresponding time in the user_cqt
            warped_onset_time = user_cqt.aligned_times[onset_idx]
            warped_next_onset_time = user_cqt.aligned_times[next_onset_idx]

            # Compute warped note duration using the ratio of the aligned / original 
            # 'internote' durations between two onset times, then use to scale the 
            # original note duration.

            original_internote_duration = next_onset_time - onset_time
            aligned_internote_duration = warped_next_onset_time - warped_onset_time
            warp_ratio = aligned_internote_duration / original_internote_duration

            original_note_duration = note.duration
            warped_note_duration = original_note_duration * warp_ratio

            # Use the previous pitch and velocity values
            pitch = int(note.pitch)
            velocity = int(note.velocity)

            new_note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=warped_onset_time, end=warped_onset_time+warped_note_duration)
            violin_instrument.notes.append(new_note)

            if print_debug:
                print(f'WARPING PITCH: {note.pitch} @ TIME: {note.start}\n---')

                print(f'onset_idx: {onset_idx}, next_onset_idx: {next_onset_idx}')
                print(f'onset_time: {onset_time}, next_onset_time: {next_onset_time}')
                print(f'warped_onset_time: {warped_onset_time}, warped_next_onset_time: {warped_next_onset_time}\n')
                
                print(f'original_internote_duration: {original_internote_duration} -> warped_internote_duration: {aligned_internote_duration}')
                print(f'original_note_duration: {original_note_duration} -> warped_note_duration: {warped_note_duration}\n')

                print(f'> Adding note {pitch} @ time {warped_onset_time} with duration {warped_note_duration}\n')

        aligned_midi.instruments.append(violin_instrument)

        return aligned_midi
