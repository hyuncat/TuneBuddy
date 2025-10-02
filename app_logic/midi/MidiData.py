from pathlib import Path
import mido
from music21 import converter

from app_logic.NoteData import NoteData, Note

class MidiData:
    """
    Load and store score data from MIDI or MusicXML files.
    Stores mido.Message objects which can be handled by MidiSynth,
    organized by elapsed time. Also track channels used and BPM.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.length = 0 # length of the piece in seconds

        # --- THE ESSENTIAL STUFF ---
        # store data by {elapsed_time: Message}
        self.messages: dict[float, list[mido.Message]] = {}
        self.programs: dict[float, list[mido.Message]] = {} # stores "turn on instrument" messages
        self.metas: dict[float, list[mido.MetaMessage]] = {}
        self.note_data: NoteData = NoteData()
        
        # list of all channels used in this score
        self.channels: set[int] = set()
        self.bpm: int = 100 # default tempo

        self.load(filepath)
        self.make_note_data()

    def load(self, filepath: str):
        """Load a score file, either MIDI or MusicXML."""
        ext = Path(filepath).suffix.lower()
        print(f"Loading score file: {filepath} (ext: {ext})")
        
        if ext in {'.mid', '.midi'}:
            midi_data = mido.MidiFile(filepath)
            self.handle_midi(midi_data)
        elif ext in {'.xml', '.musicxml'}:
            self.handle_musicxml(filepath)

        else:
            raise ValueError(f"Cannot handle file type: {filepath}")

    def handle_midi(self, midi_data: mido.MidiFile):
        """Load a MIDI file using mido, parse messages by elapsed time.
        Iterate through all messages, categorize by 
            - meta
            - program change
            - normal messages
        then store as dict[elapsed_time, list[Message]].
        Also track all channels used. And find the BPM if possible.
        """
        print("Handling MIDI file...")

        metas, messages, programs = {}, {}, {}
        channels = set()

        elapsed_time = 0
        for msg in midi_data:
            elapsed_time += msg.time # update time elapsed

            if msg.is_meta:
                if msg.type == "set_tempo":
                    self.bpm = round(mido.tempo2bpm(msg.tempo))
                metas.setdefault(elapsed_time, []).append(msg)
                continue

            # append MESSAGE with elapsed time into messages
            messages.setdefault(elapsed_time, []).append(msg)

            # track program changes
            if msg.type == "program_change":
                programs.setdefault(elapsed_time, []).append(msg)
                channels.add(msg.channel)

        # ---> error handling: if no channels used, add a fake one
        if not channels:
            channels.add(0)
            fake_msg = mido.Message('program_change', program=40, channel=0, time=0)
            programs[0] = [fake_msg]

        # update results
        self.messages = messages
        self.programs = programs
        self.metas = metas
        self.channels = channels
        self.length = elapsed_time # total length of the piece in seconds

    def handle_musicxml(self, filepath: str):
        """
        Basically converts MusicXML to MIDI using music21, then handles as MIDI.
        Has a slightly nicer BPM handling than with MIDI files.
        """
        print("Handling MusicXML file...")
        self.score = converter.parse(filepath)
        midi_bytes = self.score.write('midi')
        midi_data = mido.MidiFile(file=midi_bytes)
        self.handle_midi(midi_data)

        # BPM handling
        for el in self.score.recurse().getElementsByClass('MetronomeMark'):
            self.bpm = round(el.number)
            break

    def get_length(self) -> float:
        """Return the length of the piece in seconds."""
        return self.length

    def make_note_data(self):
        """Convert the stored messages into Note objects in NoteData.
        Should be called after loading a MIDI or MusicXML file.
        """
        note_onsets = {}
        notes = {}

        i = 0
        for elapsed_time, msgs in self.messages.items():
            for msg in msgs:
                # skip non-note related stuff
                if msg.type not in {'note_on', 'note_off'}:
                    continue

                key = (msg.channel, msg.note) # create unique key per note

                # velocity>0 because sometimes midi files are weird lol
                if msg.type=='note_on' and msg.velocity>0:
                    note_onsets[key] = elapsed_time

                elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                    if key not in note_onsets:
                        continue
                    # end the note we recorded in note_onsets and write to NoteData
                    start_time = note_onsets[key]
                    note = Note(
                        i=i,
                        start_time=start_time,
                        end_time=elapsed_time,
                        midi_num=[msg.note]
                    )
                    notes[start_time] = note

                    # cleanup our iteration variables
                    del note_onsets[key]
                    i += 1
                    
        self.note_data.data = notes
        self.note_data.times = sorted(notes.keys())

    def resize(self, new_length: float):
        """Resize the piece to new_length in seconds.
        Stretches or compresses all message timings.
        """
        if new_length == self.length:
            return  # no change

        # calculate the stretching/compressing factor
        factor = new_length / self.length

        # update all message timings
        messages = {}
        for t, msgs in self.messages.items():
            new_time = t * factor
            messages[new_time] = msgs
        self.messages = messages

        metas = {}
        for t, msgs in self.metas.items():
            new_time = t * factor
            metas[new_time] = msgs
        self.metas = metas

        programs = {}
        for t, msgs in self.programs.items():
            new_time = t * factor
            programs[new_time] = msgs
        self.programs = programs

        self.make_note_data() # regenerate note data
        
        # update the length + BPM
        new_bpm = self.bpm / factor
        self.bpm = round(new_bpm)
        self.length = new_length

    def change_tempo(self, new_bpm: int):
        """Change the tempo of the piece to new_bpm."""

        if new_bpm <= 0:
            print("Invalid BPM, must be > 0")
            return
        
        factor = self.bpm / new_bpm
        new_length = self.length * factor
        self.resize(new_length)
        self.bpm = new_bpm