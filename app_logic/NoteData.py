import numpy as np
from collections import defaultdict
class Note:
	def __init__(self, start_time: float, end_time: float, midi_num: float):
		self.start_time = start_time
		self.end_time = end_time
		self.midi_num = midi_num

class NoteData:
    def __init__(self):
        self.data: dict[float, Note] = defaultdict(Note)

    def write_note(self, note: Note):
        """writes a single note to the note data @ the corresponding start_time"""
        self.data[note.start_time] = note

    def read(self, start_time: float, end_time: float) -> list[Note]:
        """return all notes found within the start_time - end_time boundaries"""
        times = self.data.keys()
        i = np.argmin(times-start_time)
        j = np.argmin(times-end_time) # is an overestimate, since keys indexed by start_time
        note_times = times[i:j]

        notes = []
        for t in note_times:
            n = self.data[t]
            if n.end_time <= end_time: # ensure we get notes within the boundaries
                notes.append(self.data[t])
        return notes

    def read_note(self, start_time: float) -> Note:
        """read a single note corresponding to the closest time"""
        times = self.data.keys()
        i = np.argmin(times-start_time)
        t = times[i]
        return self.data[t]