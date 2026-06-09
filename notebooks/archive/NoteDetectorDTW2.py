import numpy as np
import librosa
from collections import defaultdict

from app_logic.NoteData import NoteData, Note
from algorithms.Config import Config
from app_logic.user.ds.PitchData import PitchData
from app_logic.user.ds.Recording import Recording

class NoteDetectorDTW:
    def __init__(self, recording: Recording=None, config: Config=None):        
        self.recording = recording
        self.config = recording.config if recording else config

    def detect_notes(self, pitch_data: PitchData) -> NoteData:
        user_pitches = pitch_data.read(i=0, j=len(pitch_data.data), clean=True)
        # get midi notes from the active instrument note data
        active_note_data = self.recording.score_data.get_active_note_data()
        midi_notes = [n.midi_num[0] for n in active_note_data.data.values()]

        # need cost matrix
        cost_matrix = self._build_cost_matrix(user_pitches, midi_notes)

        SKIP_PENALTY = 5.0
        ALLOW_SUBSEQ = False
        _, path = librosa.sequence.dtw(
            C=cost_matrix,
            step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]]),
            weights_add=np.array([0.0, 0.0, SKIP_PENALTY]),
            weights_mul=np.array([1.0, 1.0, 1.0]),
            subseq=ALLOW_SUBSEQ,
            backtrack=True,
        )

        # parse the path
        note_idx_to_pitch_idx = defaultdict(list)
        for note_idx, pitch_idx in path:
            note_idx_to_pitch_idx[note_idx].append(pitch_idx)
        notes = {}
        for note_idx, pitch_indices in note_idx_to_pitch_idx.items():
            p_i = min(pitch_indices)
            p_j = max(pitch_indices)
            pitches = user_pitches[p_i:p_j]
            # get median pitch among the pitches aligned to this note
            median_pitch = np.median([p.candidates[0][0] for p in pitches])
            note = Note(
                i=note_idx,
                midi_num=[median_pitch],
                start_time=user_pitches[p_i].time,
                end_time=user_pitches[p_j].time
            )
            notes[user_pitches[p_i].time] = note
        
        note_data = NoteData()
        note_data.load_data(notes=notes)
        return note_data
        

    def _build_cost_matrix(self, user_pitches, midi_notes):
        """Build a cost matrix for input into Librosa DTW,
        where cost_matrix[i, j] = distance between user_pitches[i] and midi_notes[j]
        """
        T, N = len(user_pitches), len(midi_notes)
        C = np.empty((N, T), dtype=np.float64)

        for i, user_pitch in enumerate(user_pitches):
            for j, midi_pitch in enumerate(midi_notes):
                cost = self._get_cost(user_pitch, midi_pitch)
                C[j, i] = cost
        
        return C

    def _get_cost(self, user_pitch, midi_pitch):
        """Get the cost of aligning user_pitch to midi_pitch.
        Cost is defined as the average distance between each candidate pitch 
        and the midi_pitch, weighted by the candidate probabilities.
        """
        cost = 0
        for pitch, prob in user_pitch.candidates:
            cost += prob * abs(pitch - midi_pitch)
        cost = cost / len(user_pitch.candidates) # average cost over candidates
        return cost
