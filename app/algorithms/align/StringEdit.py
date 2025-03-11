import numpy as np
from app.core.recording.Note import Note, UserString, MidiString

class Mistake:
    def __init__(self, type, user_note, midi_note):
        self.type = type
        self.user_note = user_note
        self.midi_note = midi_note

class Alignment:
    def __init__(self, notes: list[tuple[Note, Note]], mistakes: list[Mistake]):
        """initialize the alignment class to store the user notes and mistakes"""
        self.notes: list[tuple[Note, Note]] = notes
        self.mistakes: list[Mistake] = mistakes

    def merge_notes(i: int, j: int):
        """merges notes i with j (= i+1) from the list"""
        pass

    def insert_note(note, i):
        """inserts note into position i in notes"""
        pass

    def update_pitch(note, new_pitch):
        """updates the pitch of a note"""
        pass

    def print_mistakes(self):
        print("USER MISTAKES\n---")
        for mistake in self.mistakes:
            print(f"time={mistake.user_note[0].start:.3f} | error={mistake.type}, user={mistake.user_note[0].pitch}, midi={mistake.midi_note.pitch}")

    def print_alignment(self):
        print("\nUSER ALIGNMENT\n---")
        for user_note, midi_note in self.notes:
            if not user_note:
                print(f"deletion! midi={midi_note.pitch}")
            elif not midi_note:
                print(f"insertion! user={user_note[0].pitch}")
            else:
                print(f"user={user_note[0].pitch}, midi={midi_note.pitch}")

class StringEditor:
    # string edit costs
    INSERTION_COST = 1.5
    DELETION_COST = 2
    SUBSTITUTION_COST = 1
    TOLERANCE = 1

    def __init__(self):
        pass

    @classmethod
    def string_edit(cls, user_string: UserString, midi_string: MidiString):
        """run string editing on the two user and midi strings.
        returns the alignment object as the result of string editing
        """
        # setup dp matrix
        N = len(midi_string.notes)
        M = len(user_string.notes)

        mat = np.zeros([N+1, M+1], dtype=np.float64)
        backpointer = np.zeros([N+1, M+1], dtype=np.int64)

        # initialize first row / column
        mat[0, :] = np.cumsum([0]+[StringEditor.INSERTION_COST]*M) # all insertions
        mat[:, 0] = np.cumsum([0]+[StringEditor.DELETION_COST]*N) # all deletions

        for i in range(1, N+1):
            for j in range(1, M+1):

                top = mat[i-1, j]
                diag = mat[i-1, j-1] 
                left = mat[i, j-1]

                StringEditor.SUBSTITUTION_COST = 1
                note_distance = user_string.get_distance(midi_string, j-1, i-1)
                if abs(note_distance) < StringEditor.TOLERANCE: # being generous, the StringEditor.TOLERANCE
                    StringEditor.SUBSTITUTION_COST = 0 # same note pitch

                top_three = np.array([
                    top + StringEditor.DELETION_COST,
                    diag + StringEditor.SUBSTITUTION_COST,
                    left + StringEditor.INSERTION_COST
                ])
                mat[i, j] = np.min(top_three)
                backpointer[i, j] = np.argmin(top_three)

        # traceback the backpointer
        i = N
        j = M

        mistakes = []
        notes = []
        while i>0 or j>0:

            mistake_type = backpointer[i, j]
            user_note = user_string.get_note(j-1) if j > 0 else None
            midi_note = midi_string.get_note(i-1) if i > 0 else None

            # deletion
            if mistake_type==0 and i>0:
                mistakes.append(
                    Mistake(type="deletion", user_note=user_note, midi_note=midi_note)
                )
                notes.append((None, midi_note))
                i -= 1

            # substitution / no change
            if mistake_type==1 and i>0 and j>0:
                note_distance = user_string.get_distance(midi_string, j-1, i-1)
                if abs(note_distance) >= StringEditor.TOLERANCE:
                    mistakes.append(
                        Mistake(type="substitution", user_note=user_note, midi_note=midi_note)
                    )
                notes.append((user_note, midi_note))
                i -= 1
                j -= 1

            # insertion
            if mistake_type==2 and j>0:
                mistakes.append(
                        Mistake(type="insertion", user_note=user_note, midi_note=midi_note)
                    ) 
                j -= 1
                notes.append((user_note, None))

        notes = list(reversed(notes))
        mistakes = list(reversed(mistakes))
        return Alignment(notes, mistakes)