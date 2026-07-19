import numpy as np
from collections import defaultdict
from bisect import bisect_left, bisect_right

from algorithms.Config import Config

class Note:
    def __init__(self, i: int, start_time: float, end_time: float, midi_num: list[float],
                 velocity: int=None, instrument: int=None):
        self.id = i # used to keep track of note within the piece

        # timing info
        self.start_time = start_time
        self.end_time = end_time
        # --> baseline times to avoid drift when transposing
        self.base_start_time = start_time
        self.base_end_time = end_time
        
        # other important values
        self.midi_num = midi_num
        self.velocity = velocity # might change to a list of peaks later?
        self.instrument = instrument
        # per-note descriptors (pitch/cents, vibrato, volume, ...) — a NoteInfo,
        # filled lazily by the UI when the note is inspected (see NoteInfo.analyze)
        self.info = None


    def get_note_name(self, prefer_flats: bool = False) -> str:
        """Convert the most-likely MIDI note to a letter name like C4 or F#3 (or
        Bb3 with prefer_flats), via Config's shared note-name method."""
        if len(self.midi_num) == 0:
            return "—"
        return Config.get_note_name(self.midi_num[0], prefer_flats=prefer_flats)

class NoteData:
    """Data to store and retrieve notes efficiently (indexing + binary search)
    Supports read by index and by start/end time."""
    def __init__(self):
        self.data: dict[float, Note] = defaultdict(Note)
        self.times: list[float] = [] # times are stored for binary search 

    # === WRITE METHODS === #
    def load_data(self, notes: dict[float, Note], times: list[float]=None, bounds: tuple=None):
        """load in note data from a dict of notes, optional times list and bounds"""
        self.data = notes
        self.times = times if times is not None else sorted(list(notes.keys())) 

    def write_note(self, note: Note):
        """writes a single note to the note data @ the corresponding start_time"""
        if note.start_time not in self.data:
            # keep times sorted for binary search
            i = bisect_left(self.times, note.start_time)
            self.times.insert(i, note.start_time)
        
        self.data[note.start_time] = note

    # === EDIT METHODS === #
    def transpose(self, dx: float=None, dy: float=None):
        """Move every note by `dx` seconds and/or `dy` semitones. All x/y-wise
        NoteData moves flow through here: a time move rekeys the dict and
        sorted-times list so time lookups stay in sync; a pitch move shifts
        every voiced chord member, clamped to the MIDI range."""
        if dy:
            for note in self.data.values():
                note.midi_num = [
                    max(0, min(127, m + dy)) if m != -1 else -1
                    for m in note.midi_num
                ]
        if not dx:
            return
        new_data = {}
        for note in self.data.values():
            note.start_time += dx
            note.end_time += dx
            new_data[note.start_time] = note
        self.data = new_data
        self.times = sorted(new_data.keys())

    # === GET METHODS === #
    # --- length / bounds getters ---
    def get_length(self, bounds: tuple[float, float]=None) -> float:
        """Return the length of the note data in seconds.
        If no bounds are supplied, returns the end_time of the last note; else
        return the length of the notes within the bounds"""
        if not self.times:
            return 0.0
        
        if bounds is None:
            first_note = self.read_note(i=0)
            last_time = self.times[-1]
            last_note = self.data[last_time]
        else:
            first_note = self.read_note(start_time=bounds[0])
            last_note = self.read_note(start_time=bounds[1])
        return last_note.end_time - first_note.start_time
    
    def get_bounds(self, clean: bool=True, use_note_end: bool=True) -> tuple[float, float]:
        """Return the (start_time, end_time) time span covered by the notes, or
        None if there are none. With clean=True (default), rests/unvoiced notes
        (midi_num[0] == -1) are skipped — the voiced span; clean=False uses every
        note. With use_note_end=False, return first note start -> last note start,
        which is useful for tempo fitting by onset span."""
        notes = self.read(i=0, j=len(self.times), clean=clean) if self.times else []
        if not notes:
            return None
        end = notes[-1].end_time if use_note_end else notes[-1].start_time
        return notes[0].start_time, end

    def get_min_note_length(self, default: float=0.0, clean: bool=True) -> float:
        """Return the minimum note length in seconds.

        When clean=True, unvoiced/rest notes are ignored. This is the score-level
        source for Config.min_note_length.
        """
        if not self.times:
            return default
        
        min_length = float('inf')
        for t in self.times:
            n = self.data[t]
            if clean and (not n.midi_num or n.midi_num[0] == -1):
                continue
            note_length = n.end_time - n.start_time
            if note_length > 0 and note_length < min_length:
                min_length = note_length
        
        return min_length if min_length != float('inf') else default

    def step_note(self, note: Note, step: int, clean: bool=True) -> Note:
        """The note `step` places after `note` in onset order (negative steps walk
        backwards), or None past either end / if `note` isn't in here. Backs the
        arrow-key walk through a take (see GuitarHero's note popup); it clamps
        rather than wraps, so the ends of the take are a dead stop."""
        notes = self.read(i=0, j=len(self.times), clean=clean)
        try:
            i = notes.index(note)
        except ValueError:
            return None
        j = i + step
        return notes[j] if 0 <= j < len(notes) else None

    def notes_by_id(self) -> dict[int, Note]:
        """Current notes keyed by their stable id (notes without one are
        skipped). Used to relink alignment refs after a score NoteData is
        rebuilt by change_tempo()/resize() — see Alignment.sync_score_notes."""
        by_id = {}
        for note in self.data.values():
            if getattr(note, "id", None) is not None:
                by_id[note.id] = note
        return by_id

    # --- note getting / reading --- #
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

    def read_current_note(self, t: float) -> Note:
        """return the note being played at time t, if any"""
        if not self.times:
            return None
        
        i = bisect_left(self.times, t)
        if i == 0 or i == len(self.times):
            return None
        note_time = self.times[i]
        note = self.data[note_time]

         # this is the one scenario which could happen
         # where the note is to the right of time t
        if t < note.start_time:
            j = i-1 if i-1 >= 0 else 0
            note_time = self.times[j]
            note = self.data[note_time]
            return note

        return note

    def note_containing(self, t: float, clean: bool=True) -> Note | None:
        """The note actually sounding at time t (strict start<=t<=end), or
        None. Unlike read_current_note — which returns the preceding note
        during rests — gaps between notes read as no note, which is what the
        note-detail panel's blank rule needs. clean=True also treats rests
        (midi_num[0] == -1) as no note."""
        if not self.times:
            return None
        i = bisect_right(self.times, t) - 1  # last note starting at or before t
        if i < 0:
            return None
        note = self.data[self.times[i]]
        if not (note.start_time <= t <= note.end_time):
            return None
        if clean and (not note.midi_num or note.midi_num[0] == -1):
            return None
        return note

    def read(self, start_time: float=None, end_time: float=None,
             i=None, j=None, clean:bool=False) -> list[Note]:
        """return all notes found within the start_time - end_time boundaries"""
        if not self.times or (start_time is None and end_time is None and i is None and j is None):
            return []
        
        if i is not None and j is not None:
            return self._read_index(i, j, clean=clean)

        return self._read_time(start_time, end_time, clean=clean)

    def _read_index(self, i: int, j: int, clean: bool=False) -> list[Note]:
        """return all notes found within the note index boundaries i-j"""
        if i < 0 or j > len(self.times) or i >= j:
            return []
        
        notes = []
        for t in self.times[i:j]:
            notes.append(self.data[t])

        if clean:
            notes = [n for n in notes if n.midi_num[0] != -1]
            
        return notes

    def _read_time(self, start_time: float, end_time: float, clean: bool=False) -> list[Note]:
        """return all notes found within the start_time - end_time boundaries"""
        if not self.times or start_time is None or end_time is None:
            return []

        j = bisect_right(self.times, end_time)

        notes = []
        for t in self.times[:j]:
            n = self.data[t]
            # ensure we get notes within the boundaries
            notes.append(n) if n.end_time >= start_time else None

        if clean:
            notes = [n for n in notes if n.midi_num[0] != -1]
            
        return notes