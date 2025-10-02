import numpy as np
from collections import defaultdict
from bisect import bisect_left, bisect_right
class Note:
    def __init__(self, i: int, start_time: float, end_time: float, midi_num: list[float]):
        self.id = i # used to keep track of note within the piece
        self.start_time = start_time
        self.end_time = end_time
        self.midi_num = midi_num

class NoteData:
    """Data to store and retrieve notes efficiently (indexing + binary search)
    Supports read by index and by start/end time."""
    def __init__(self):
        self.data: dict[float, Note] = defaultdict(Note)
        self.times: list[float] = [] # times are stored for binary search

    def write_note(self, note: Note):
        """writes a single note to the note data @ the corresponding start_time"""
        if note.start_time not in self.data:
            # keep times sorted for binary search
            idx = bisect_left(self.times, note.start_time)
            self.times.insert(idx, note.start_time)
        self.data[note.start_time] = note

    def read(self, start_time: float=None, end_time: float=None, i=None, j=None) -> list[Note]:        
        """return all notes found within the start_time - end_time boundaries"""
        if not self.times or (start_time is None and end_time is None and i is None and j is None):
            return []
        
        if i is not None and j is not None:
            return self._read_index(i, j)

        return self._read_time(start_time, end_time)

    def _read_index(self, i: int, j: int) -> list[Note]:
        """return all notes found within the note index boundaries i-j"""
        if i < 0 or j > len(self.times) or i >= j:
            return []
        
        notes = []
        for t in self.times[i:j]:
            notes.append(self.data[t])
            
        return notes
    
    def _read_time(self, start_time: float, end_time: float) -> list[Note]:
        """return all notes found within the start_time - end_time boundaries"""
        if not self.times or start_time is None or end_time is None:
            return []
        
        i = bisect_left(self.times, start_time)
        j = bisect_right(self.times, end_time)

        notes = []
        for t in self.times[i:j]:
            n = self.data[t]
            if n.start_time <= end_time and n.end_time >= start_time: # ensure we get notes within the boundaries
                notes.append(self.data[t])
        return notes

    def read_note(self, start_time: float=None, i: int=None) -> Note:
        """read a single note corresponding to the closest time or the note index i"""
        if not self.times or (start_time is None and i is None):
            return None
        
        if i is not None:
            if i < 0 or i >= len(self.times):
                return None
            return self.data[self.times[i]]
        
        # else, binary search for closest time
        i = bisect_left(self.times, start_time)
        if i == 0:
            closest_time = self.times[0]
        elif i == len(self.times):
            closest_time = self.times[-1]
        else:
            before = self.times[i - 1]
            after = self.times[i]
            closest_time = before if abs(before - start_time) < abs(after - start_time) else after
        return self.data[closest_time]
