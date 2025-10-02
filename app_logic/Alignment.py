from app_logic.NoteData import Note

class Mistake:
    def __init__(self, type: str, user_note: Note, midi_note: Note):
        self.type = type
        self.user_note = user_note
        self.midi_note = midi_note

class Alignment:
    def __init__(self, notes: list[tuple[Note, Note]], mistakes: list[Mistake]):
        """initialize the alignment class to store the user notes and mistakes"""
        self.notes: list[tuple[Note, Note]] = notes
        self.mistakes: list[Mistake] = mistakes

        # catalogue mistakes into sets for easy access (for plotting)
        self.insertions = {m.user_note.id for m in mistakes if m.type == 'insertion'}
        self.deletions = {m.midi_note.id for m in mistakes if m.type == 'deletion'}
        self.substitutions = {(m.user_note.id, m.midi_note.id) for m in mistakes if m.type == 'substitution'}

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
            user_str = (
                f"{mistake.user_note.start:.3f}, pitch={mistake.user_note.midi_num[0]}"
                if mistake.user_note else "None"
            )
            midi_str = (
                f"{mistake.midi_note.start:.3f}, pitch={mistake.midi_note.midi_num[0]}"
                if mistake.midi_note else "None"
            )
            print(f"error={mistake.type} | user=({user_str}) | midi=({midi_str})")

    def print_alignment(self):
        print("\nUSER ALIGNMENT\n---")
        for user_note, midi_note in self.notes:
            if not user_note:
                print(f"deletion! midi={midi_note.midi_num[0]}")
            elif not midi_note:
                print(f"insertion! user={user_note.midi_num[0]}")
            else:
                print(f"user={user_note.midi_num[0]}, midi={midi_note.midi_num[0]}")

    def print_mistakes_summary(self):
        print("MISTAKE SUMMARY\n---")
        print(f"Total mistakes: {len(self.mistakes)}")
        print(f"Insertions: {len(self.insertions)}")
        for n in self.insertions:
            print(f"  user_note {n}, pitch={self.notes[n][0].midi_num[0]:.3f}")
        print(f"Deletions: {len(self.deletions)}")
        for n in self.deletions:
            print(f"  midi_note {n}, pitch={self.notes[n][1].midi_num[0]:.3f}")
        print(f"Substitutions: {len(self.substitutions)}")
        for u, m in self.substitutions:
            print(f"  user_note {u}, pitch={self.notes[u][0].midi_num[0]:.3f} | midi_note {m}, pitch={self.notes[m][1].midi_num[0]:.3f}")