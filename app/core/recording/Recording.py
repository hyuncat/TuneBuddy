from app.core.audio.AudioData import AudioData
from app.core.midi.MidiData import MidiData

from app.algorithms.pitch.PYin import PYin
from app.core.recording.Pitch import Pitches
from app.algorithms.align.NoteDetector import NoteDetector
from app.algorithms.align.StringEdit import StringEditor, MidiString, UserString

class Recording:
    def __init__(self):
        self.audio_data: AudioData = AudioData()
        self.midi_data: MidiData = None
        self.pyinner = PYin(sr=44100, f0_min=196, f0_max=5000, tuning=440)
        self.pitches = None
        self.note_detector = NoteDetector(self)

    def load_audio(self, audio_filepath: str=None, sr=44100):
        """Load the audio from a filepath and create AudioData for it"""
        self.audio_data.load_data(audio_filepath, sr=sr)

    def load_midi(self, midi_filepath: str=None):
        """Load the midi file from a filepath and create MidiData for it"""
        self.midi_data = MidiData(midi_filepath)

    def detect_pitches(self):
        """Detect pitches for the audio in 'audio_data' using PYIN algorithm"""
        pitches = self.pyinner.pyin(self.audio_data.data)
        self.pitches = Pitches(pitches, self.pyinner)

    def align(self):
        """Fills in self.alignment data with the user's detected notes and midi notes
        Is what gives us interpretable feedback
        """
        self.user_string = self.note_detector.detect_notes(
            start_time=0, end_time=None, rank=0, 
            pitch_thresh=0.6, slope_thresh=1)
        self.midi_string = MidiString(self.midi_data)
        self.alignment = StringEditor.string_edit(self.user_string, self.midi_string)

