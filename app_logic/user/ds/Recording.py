import numpy as np

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData, Pitch
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData
from app_logic.user.ds.Buffer import Buffer
from algorithms.Config import Config

class Recording:
    def __init__(self, score_data: ScoreData=None, config: Config=None):
        """the user data, associated with a singular recording of a score.
        each recording has its own audio data, pitch data, note data, and alignment
        as well as its own set of algorithms and parameters for processing that data"""
        self.score_data = score_data if score_data is not None else ScoreData()
        # inherit the score's current active instrument so new recordings
        # always target whichever channel was selected when they were created
        self.active_instrument = self.score_data.active_instrument
        self.update_config(config)

        # algorithms!!
        from algorithms.PitchDetector import PitchDetector
        from algorithms.PitchSmoother import PitchSmoother
        from algorithms.NoteDetector import NoteDetector
        from algorithms.StringEditor import StringEditor
        from algorithms.MistakeChecker import MistakeChecker
        self.pitch_detector = PitchDetector(recording=self)
        self.pitch_smoother = PitchSmoother(recording=self)
        self.note_detector = NoteDetector(recording=self)
        self.string_editor = StringEditor(recording=self)
        self.mistake_checker = MistakeChecker(recording=self)

        # essential data variables
        self.audio_data = AudioData(config=self.config)
        self.pitch_data = PitchData(config=self.config)
        self.note_data = NoteData()
        self.alignment: Alignment = Alignment(config=self.config) # filled in later
        self.overridden_mistake_indices = set()

        # queue data structures for real time pitch + note detection + correction
        self.a2p_queue = Buffer(self.config.sr) #audio-to-pitches
        self.p2n_queue = Buffer(sr=self.config.sr/self.config.h1) #pitches-to-notes
        self.n2c_queue = None #notes-to-corrections

    def update_config(self, config: Config=None):
        """initialize the config, either with a provided one or a default one"""
        if config is None:
            self.config = Config()
        else:
            self.config = config
            
        if hasattr(self, 'pitch_detector'):
            self.pitch_detector.load_config(self.config)
        if hasattr(self, 'pitch_smoother'):
            self.pitch_smoother.update_config(self.config)
        if hasattr(self, 'note_detector'):
            self.note_detector.update_config(self.config)
        if hasattr(self, 'string_editor'):
            self.string_editor.update_config(self.config)
        if hasattr(self, 'mistake_checker'):
            self.mistake_checker.update_config(self.config)
    # def on_pitches_detected(self, pitches):
    #     self.pitch_data.data = pitches

    def load_audio(self, audio_filepath: str):
        """load in a pre-recorded audio file from a filepath, then kick off pitch
        detection on the whole file in the background. Listen on
        `pitch_detector.detection_finished` to know when pitch_data is ready."""
        self.audio_data.load_data(audio_filepath)
        self.pitch_detector.detect_pitches_async()
        # self.detect_notes()

    def detect_pitches(self, on_phase=None):
        """run pitch detection, then smoothing, on the current audio data.
        `on_phase(text)`, if given, is called at the start of each stage so a
        caller can surface progress (e.g. a status-bar message)."""
        if on_phase: on_phase("Detecting pitches...")
        self.pitch_data.data = self.pitch_detector.detect_pitches(self.audio_data.data)
        if on_phase: on_phase("Smoothing pitches...")
        self.pitch_data.data = self.pitch_smoother.smooth(self.pitch_data.data)

    def detect_notes(self):
        """run note detection on the current pitch data"""
        self.note_data = self.note_detector.detect_notes(self.pitch_data)

    def detect_mistakes(self):
        user_notes, midi_notes = self.note_data, self.score_data.note_datas[self.active_instrument]
        notes, mistakes = self.string_editor.string_edit(user_string=user_notes, midi_string=midi_notes)
        self.alignment.load_alignment(notes, mistakes)
        self.alignment.reapply_overrides(self.overridden_mistake_indices)
    
    def correct_mistakes(self):
        nd, alignment = self.mistake_checker.check_mistakes(recording=self)
        self.note_data = nd
        self.alignment = alignment

    def write_data(self, indata: np.ndarray, start_time: float):
        """write indata to the audio_data at the given start_time
        and append to our queue for pitch processing
        """
        self.audio_data.write_data(indata, start_time)
        self.a2p_queue.push(indata)

    def write_pitch_data(self, indata: list[Pitch], start_time: float):
        """write indata to the pitch_data at the given start_time
        and append to our queue for note processing
        """
        self.pitch_data.write(indata, start_time)
        self.p2n_queue.push(indata)

    def get_length(self, raw=True):
        if raw:
            if len(self.note_data.times) > 0:
                return self.note_data.get_length()
            else:
                return self.audio_data.get_length()
        # get start time of first VOICED note, end time of last note
        start_time = self._get_first_note(voiced=True).start_time
        end_time = self._get_last_note(voiced=True).end_time
        return end_time - start_time
    
    def _get_first_note(self, voiced=True):
        if not voiced:
            return self.note_data.data[0] if self.note_data.data else None
        for n in self.note_data.data.values():
            if n.midi_num[0] != -1:
                return n
        return 0
    
    def _get_last_note(self, voiced=True):
        if not voiced:
            return self.note_data.data[-1] if self.note_data.data else None
        for n in reversed(self.note_data.data.values()):
            if n.midi_num[0] != -1:
                return n
        return 0
    
    def resize(self, new_length: float):
        """Resize the score_data to a new length by changing the BPM of the score data,
        updating the note timings and pitch distances as well."""
        # Derive the target bpm against the ORIGINAL length/tempo and let
        # change_tempo recompute the stretch factor from it (factor defaults to
        # bpm_og / new_bpm). This makes a resize behave exactly like a manual
        # tempo change, keeping self.bpm and self.length in the strict 1/bpm
        # relationship the score-viewer's bpm/bpm_og time mapping relies on.
        # (The old code anchored the factor to the *current* length and even
        # inverted the bpm, corrupting self.bpm after Analyze.)
        factor = new_length / self.score_data.midi_data.length_og
        new_bpm = round(self.score_data.bpm_og / factor)
        self.score_data.change_tempo(new_bpm)
        start_time = self._get_first_note(voiced=True).start_time
        # transpose all score notes by start_time
        self.score_data.transpose_notes(start_time)
        self._update_pitch_distances()

    def change_tempo(self, new_bpm: float):
        """Change the tempo of the recording by changing the BPM of the score data, which will automatically update the note timings and pitch distances."""
        self.score_data.change_tempo(new_bpm)
        self._update_pitch_distances()

    def _update_pitch_distances(self):
        """Update the distance to target note for all pitches in the recording, based on the current score data."""
        for note in self.score_data.note_datas[self.active_instrument].data.values():
            if note is None:
                continue
            pitches = self.pitch_data.read(start_time=note.start_time, end_time=note.end_time, clean=True)
            for p in pitches:
                p.distance = note.midi_num[0] - p.candidates[0][0]
    
    def toggle_mistake_override(self, mistake_index: int):
        #error checking
        if not (0 <= mistake_index < len(self.alignment.mistakes)):
            return
        #Toggle persisted override state for one mistake.
        mistake = self.alignment.mistakes[mistake_index]
        pair_index = mistake.get_pair_index()
        if mistake_index in self.overridden_mistake_indices:
            self.overridden_mistake_indices.remove(mistake_index)
            self.alignment.toggle_overridden_pair_indices(pair_index, False)
            overridden = False
        else:
            self.overridden_mistake_indices.add(mistake_index)
            self.alignment.toggle_overridden_pair_indices(pair_index, True)
            overridden = True

        if 0 <= mistake_index < len(self.alignment.mistakes):
            self.alignment.mistakes[mistake_index].set_override(overridden)