"""
File containing the note, and the string of notes which make up a recording.
Rolling medians --> notestring --> string edit!
"""


from dataclasses import dataclass
from math import ceil
from app.core.midi.MidiData import MidiData

class Note:
    def __init__(self, pitch: float, start: float, end: float, volume: float=None):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.volume = volume

class NoteString:
    def __init__(self):
        self.notes: list = []

    def append_note(self, note: Note):
        self.notes.append(note)

    def insert_note(self, note: Note, index: int):
        self.notes.insert(index, note)

    def get_note(self, note_index: int):
        return self.notes[note_index]

class UserString(NoteString):
    def __init__(self):
        """handles storing multiple note estimtes of different rank"""
        super().__init__()
        self.notes: list[list[Note]] = []

    def append_note(self, notes: list[Note]):
        self.notes.append(notes)

    def insert_note(self, notes: list[Note], index: int):
        self.notes.insert(index, notes)

    def get_note(self, note_index: int, rank: int=None):
        if not rank:
            return self.notes[note_index]
        return self.notes[note_index][rank]

    def get_distance(self, note_string2: 'NoteString', note_i1: int, note_i2: int):
        """returns the minimum distance between the user notes and the midi"""
        # note distance computations in here...
        note_1 = self.get_note(note_i1) # returns the list
        note_2 = note_string2.get_note(note_i2)

        return min(abs(n_1i.pitch - note_2.pitch) for n_1i in note_1)

class MidiString(NoteString):
    def __init__(self, midi_data: MidiData):
        """midi specific string operations"""
        super().__init__()
        self.load_midi(midi_data)

    def load_midi(self, midi_data: MidiData):
        """Load a midi_data and convert into a NoteString object for string editing"""
        for i, note in midi_data.pitch_df.iterrows():
            new_note = Note(
                pitch=note['pitch'],
                start=note['start'],
                end=note['start'] + note['duration'],
                volume=note['velocity']
            )
            self.append_note(new_note)