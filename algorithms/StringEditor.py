import numpy as np
from algorithms.Config import Config
from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import NoteData, Note
from app_logic.user.ds import Recording
import time

class StringEditor:
    def __init__(self, recording: Recording=None, config: Config=None):
        self.recording = recording
        self.config = recording.config if recording else config

        # string edit costs
        self.INSERTION_COST = self.config.ins_cost
        self.DELETION_COST = self.config.del_cost
        self.SUBSTITUTION_COST = self.config.sub_cost
        self.TOLERANCE = self.config.pitch_tolerance

        # tiger-mom parameter
        self.TIGER_LEVEL = self.config.tiger_level

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.INSERTION_COST = self.config.ins_cost
        self.DELETION_COST = self.config.del_cost
        self.SUBSTITUTION_COST = 1
        self.TOLERANCE = self.config.pitch_tolerance

        self.TIGER_LEVEL = self.config.tiger_level

    def string_edit(self, user_string: NoteData, midi_string: NoteData):
        """run string editing on the two user and midi strings.
        returns the alignment object as the result of string editing
        """
        # user_string = self.recording.NoteData
        # midi_string = self.recording.score_data.note_data

        start = time.time()
        # print("Starting string editing... ", end="", flush=True)
        user_notes = list(user_string.data.values())
        user_notes = [n for n in user_notes if n.midi_num[0] != -1]
        
        # setup dp matrix
        N = len(midi_string.times)
        M = len(user_notes)

        mat = np.zeros([N+1, M+1], dtype=np.float64)
        backpointer = np.zeros([N+1, M+1], dtype=np.int64)

        # initialize first row / column
        mat[0, :] = np.cumsum([0]+[self.INSERTION_COST]*M) # all insertions
        mat[:, 0] = np.cumsum([0]+[self.DELETION_COST]*N) # all deletions

        for i in range(1, N+1): # midi index
            for j in range(1, M+1): # user index

                top = mat[i-1, j]
                diag = mat[i-1, j-1] 
                left = mat[i, j-1]

                midi_note = midi_string.read_note(i=i-1)
                user_note = user_notes[j-1]
                # print(f"user notes: {user_note.midi_num}")
                note_distance = self.get_distance(user_note, midi_note)
                if abs(note_distance) < self.TOLERANCE: # within tolerance = same note pitch
                    SUB_COST = 0
                else:
                    # weight the substitution by how far (semitones) the user note is off
                    SUB_COST = min(abs(note_distance), 10)

                top_three = np.array([
                    top + self.DELETION_COST,
                    diag + SUB_COST,
                    left + self.INSERTION_COST
                ])
                mat[i, j] = np.min(top_three)
                backpointer[i, j] = np.argmin(top_three) # eg, 0=del, 1=sub, 2=ins

        # traceback the backpointer
        # print("starting string edit traceback...")
        i = N
        j = M

        mistakes = []
        notes = []
        mistakes_to_reverse_position = {}
        while i>0 or j>0:
            # on the boundaries the backpointer is unset (0), so force the only
            # legal move: all-insertions along the top row, all-deletions down
            # the left column. otherwise the earliest notes get silently dropped.
            if i == 0:
                mistake_type = 2  # only user notes remain -> insertion
            elif j == 0:
                mistake_type = 0  # only score notes remain -> deletion
            else:
                mistake_type = backpointer[i, j]
            midi_note = midi_string.read_note(i=i-1) if i > 0 else None
            user_note = user_notes[j-1] if j > 0 else None

            # 0: deletion
            if mistake_type==0 and i>0:
                # print(f"--> DELETION at i={i}, j={j}")
                mistake = Mistake(type="deletion", user_note=user_note, midi_note=midi_note)
                mistakes_to_reverse_position[mistake] = len(notes)
                mistakes.append(mistake)
                notes.append((None, midi_note))
                i -= 1

            # 1: substitution / no change
            elif mistake_type==1 and i>0 and j>0:
                note_distance = self.get_distance(user_note, midi_note)
                if abs(note_distance) >= self.TOLERANCE:
                    # print(f"--> SUBSTITUTION at i={i}, j={j} (distance={note_distance})")
                    mistake = Mistake(type="substitution", user_note=user_note, midi_note=midi_note)
                    mistakes_to_reverse_position[mistake] = len(notes)
                    mistakes.append(mistake)
                notes.append((user_note, midi_note))
                i -= 1
                j -= 1

            # 2: insertion
            elif mistake_type==2 and j>0:
                # print(f"--> INSERTION at i={i}, j={j}")
                mistake = Mistake(type="insertion", user_note=user_note, midi_note=midi_note)
                mistakes_to_reverse_position[mistake] = len(notes)
                mistakes.append(mistake)
                j -= 1
                notes.append((user_note, None))
            else:
                # fallback to prevent infinite loop
                # print(f"[warning] Invalid state at i={i}, j={j}, backpointer={mistake_type}")
                break

        notes = list(reversed(notes))
        mistakes = list(reversed(mistakes))
        # print(f"Done! Took {time.time() - start:.2f} seconds")
        for mistake in mistakes:
            mistake.set_pair_index(len(notes) - 1 - mistakes_to_reverse_position[mistake])
        return notes, mistakes
    
    def get_distance(self, user_note: Note, midi_note: Note):
        """return the 'distance' between the user and midi note,
        using some notion of tiger-mom-ishness.

        A score note may be a CHORD (every simultaneous pitch lives in
        midi_note.midi_num — see MidiData.make_notedatas); since pitch detection
        is monophonic it can only ever verify one pitch, so the user matches the
        NEAREST chord member. The user note's own midi_num holds detection
        candidates, so we minimise over both lists."""
        if self.TIGER_LEVEL == 1:
            return min(abs(u - m) for u in user_note.midi_num for m in midi_note.midi_num)
        return min(abs(user_note.midi_num[0] - m) for m in midi_note.midi_num)
        
