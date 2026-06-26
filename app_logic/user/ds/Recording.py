import numpy as np
from pathlib import Path

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData, Pitch
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData
from app_logic.user.ds.Buffer import Buffer
from app_logic.JsonHandler import JsonHandler
from algorithms.Config import Config

class Recording:
    TRAILING_AUDIO_PAD_SEC = 0.2

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

        # Persistence metadata. Folder/library entries point at files until their
        # parent score is active; live takes become dirty as soon as audio is
        # written and are clean again after save_audio().
        self.audio_filepath: Path | None = None
        self.unsaved_changes = False
        self.loaded_from_cache = False

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

    @staticmethod
    def cache_path_for(audio_filepath: str | Path) -> Path:
        return JsonHandler.cache_path_for(audio_filepath)

    @staticmethod
    def delete_cache_for(audio_filepath: str | Path) -> None:
        JsonHandler.delete_cache_for(audio_filepath)

    def cache_path(self) -> Path | None:
        return self.cache_path_for(self.audio_filepath) if self.audio_filepath else None

    def audio_file_exists(self) -> bool:
        return self.audio_filepath is not None and Path(self.audio_filepath).exists()

    def rename_files(self, new_stem: str) -> Path | None:
        return JsonHandler.rename_recording_files(self, new_stem)

    def load_audio(
        self,
        audio_filepath: str,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
        load_cache: bool = True,
    ):
        """load in a pre-recorded audio file from a filepath into audio_data.
        Pitch detection is kicked off separately by the caller (app.py runs the
        cleanup + detection together so the views reset as one)."""
        path = Path(audio_filepath)
        self.audio_data.load_data(str(path))
        self.audio_filepath = path
        self.unsaved_changes = False
        self.loaded_from_cache = False
        self.pitch_data = PitchData(config=self.config)
        self.reset_analysis()
        if load_cache:
            self.load_cache(score_filepath=score_filepath, recording_name=recording_name)

    def save_audio(self, audio_filepath: str | Path):
        """Persist this recording's current audio buffer to disk."""
        self.truncate_end(mark_unsaved=False)
        path = Path(audio_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_data.save_data(str(path))
        self.audio_filepath = path
        self.unsaved_changes = False

    def needs_save(self) -> bool:
        """True when this recording contains live edits/takes not written to disk."""
        return self.audio_data.end_index > 0 and self.unsaved_changes

    def has_pitch_data(self) -> bool:
        return any(p is not None for p in self.pitch_data.data)

    def save_cache(
        self,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> bool:
        return JsonHandler(self).save_cache(
            score_filepath=score_filepath,
            recording_name=recording_name,
        )

    def load_cache(
        self,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> bool:
        return JsonHandler(self).load_cache(
            score_filepath=score_filepath,
            recording_name=recording_name,
        )

    def cleanup(self):
        """Re-init essential data structures. Called before load_score() in app."""
        self.audio_data = AudioData(config=self.config)
        self.pitch_data = PitchData(config=self.config)
        self.reset_analysis()

    def reset_analysis(self):
        """Re-init analysis-derived data structures. Called before re-analyze() in app."""
        self.note_data = NoteData()
        self.alignment = Alignment(config=self.config)
        self.overridden_mistake_indices = set()

    def detect_pitches(self, on_phase=None):
        """run pitch detection, then smoothing, on the current audio data.
        `on_phase(text)`, if given, is called at the start of each stage so a
        caller can surface progress (e.g. a status-bar message)."""
        if on_phase:
            on_phase("Detecting pitches...")
        audio = self.audio_data.read_all()
        self.pitch_data.data = self.pitch_detector.detect_pitches(audio)
        if on_phase:
            on_phase("Smoothing pitches...")
        self.pitch_data.data = self.pitch_smoother.smooth(self.pitch_data.data)
        # the offline pass stamps frame times relative to buffer index 0, which
        # represents app-time `t_origin` (NEGATIVE for a Perform runway recorded
        # before the head). Mirror the audio buffer's origin onto the pitch data
        # and lift the frame times onto it so notes/alignment read in app-time.
        origin = self.audio_data.t_origin
        self.pitch_data.t_origin = origin
        if origin:
            for p in self.pitch_data.data:
                if p is not None:
                    p.time += origin

    def detect_notes(self):
        """run note detection on the current pitch data.

        A/B toggle lives on NoteDetector.USE_PELT (flip it to switch detectors;
        find_best_w2 reads the same flag and no-ops under PELT)."""
        if self.note_detector.USE_PELT:
            self.note_data = self.note_detector.detect_notes2(self.pitch_data)
        else:
            self.note_data = self.note_detector.detect_notes(self.pitch_data)

    def detect_transitions(self):
        """flag high-slope (pitch-transition) frames in the pitch data. Run after
        detect_notes() / onset refinement and before update_alignment_distances()
        so those biased frames are left uncolored (grey) instead of scored."""
        self.note_detector.detect_transitions(self.pitch_data)

    def recompute_note_pitches(self):
        """re-median each detected note's pitch over only its non-transition
        frames, so onset-refinement slide frames don't bias a note sharp/flat
        (e.g. a false F5->F#5). Run after detect_transitions(), before
        detect_mistakes()."""
        self.note_detector.recompute_note_pitches(self.note_data, self.pitch_data)

    def prune_transition_notes(self):
        """drop phantom notes that are almost entirely transition frames (notes
        'detected' inside a slide because the ND window is wide). Run after
        detect_transitions(), before detect_mistakes()."""
        self.note_data = self.note_detector.prune_transition_notes(self.note_data, self.pitch_data)

    def detect_mistakes(self):
        # the StringEditor only ever sees the clip's score notes (the full
        # NoteData when unclipped) — see ScoreData.clipped_note_data.
        user_notes = self.note_data
        midi_notes = self.score_data.clipped_note_data(channel=self.active_instrument)
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
        self.unsaved_changes = True
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

    def audio_bounds(self) -> tuple[float, float] | None:
        """App-time bounds for the recording audio currently considered live."""
        if self.audio_data.end_index <= 0:
            return None
        return self.audio_data.get_bounds()

    def audio_end_time(self) -> float:
        """App-time of the recording's logical audio end."""
        return self.audio_data.get_end_time()
    
    # get first/last notes (used in ___ find later)
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
    
    def shift(self, delta: float):
        """Slide the WHOLE recorded take (audio, pitches, notes) by `delta` sec on
        the app-time line. Audio/pitch frames move via their shared time origin (no
        array copy); notes are rekeyed. Used post-analysis to land the first voiced
        note on the score's clip start without translating the score."""
        if not delta:
            return
        self.audio_data.t_origin += delta
        self.pitch_data.t_origin += delta
        for p in self.pitch_data.data:
            if p is not None:
                p.time += delta
        self.note_data.shift(delta)

    def _clip_start_time(self) -> float:
        """App-time the take should align its first voiced note to: the score's
        clip start when clipped, else the score's first note."""
        cb = self.score_data.clip_bounds()
        if cb is not None:
            return cb[0]
        nd = self.score_data.note_datas.get(self.active_instrument)
        return nd.times[0] if nd and nd.times else 0.0

    def resize(self, new_length: float):
        """Stretch the score to match the take (`new_length` = the take's voiced
        span, rec.end - rec.start), and slide the take so the two line up.

        When CLIPPED, only the clipped span is matched to the take (not the whole
        score) and both are anchored at t=0 — see _resize_to_clip. Otherwise the
        whole score is stretched and the take slides onto the score's first note."""
        if self.score_data.clip is not None:
            self._resize_to_clip(new_length, self.score_data.clip)
            return

        # Derive the target bpm against the ORIGINAL length/tempo and let
        # change_tempo recompute the stretch factor from it (factor defaults to
        # bpm_og / new_bpm). This makes a resize behave exactly like a manual
        # tempo change, keeping self.bpm and self.length in the strict 1/bpm
        # relationship the score-viewer's bpm/bpm_og time mapping relies on.
        factor = new_length / self.score_data.midi_data.length_og
        new_bpm = round(self.score_data.bpm_og / factor)

        self.score_data.change_tempo(new_bpm)
        # Keep the score fixed and slide the TAKE instead, so its first voiced note
        # lands on the score's first note.
        first = self._get_first_note(voiced=True)
        if first != 0:
            self.shift(self._clip_start_time() - first.start_time)
        self._update_pitch_distances()

    def _resize_to_clip(self, new_length: float, clip: tuple[int, int]):
        """Clip-aware resize, in this exact order:
          1. stretch the WHOLE score to a bpm so the CLIPPED span matches the take
             (note(clip[1]).end - note(clip[0]).start == new_length == rec.end-rec.start),
          2. translate the score so the clip's first note starts at t=0,
          3. slide the take so its first voiced pitch also starts at t=0.
        So the clip and the take are co-anchored at the origin. The clip indices
        ride along the tempo rebuild; clip_bounds() re-derives the [0, new_length]
        window. (The GuitarHero overlay is redrawn by the caller — see analyze.)"""
        sd = self.score_data
        i0, i1 = clip
        nd = sd.note_datas[self.active_instrument]
        if new_length <= 0 or not (0 <= i0 <= i1 < len(nd.times)):
            return
        clip_span = nd.read_note(i=i1).end_time - nd.read_note(i=i0).start_time
        if clip_span <= 0:
            return

        # 1) bpm s.t. clip_span * (bpm_before / new_bpm) == new_length
        new_bpm = max(1, round(sd.bpm * clip_span / new_length))
        sd.change_tempo(new_bpm)  # rebuilds the notes; clip indices stay valid

        # 2) translate the score so clip[0] starts at t=0 (pre-clip notes go < 0)
        nd = sd.note_datas[self.active_instrument]
        sd.transpose_notes(-nd.read_note(i=i0).start_time)

        # 3) slide the take so its first voiced pitch starts at t=0
        first = self._get_first_note(voiced=True)
        if first != 0:
            self.shift(-first.start_time)

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

    # default align_distance for voiced pitches that no aligned note covers
    # (transitions / slides between notes): 0.0 => green. Don't penalize them.
    TRANSITION_DISTANCE = 0.0

    def update_alignment_distances(self):
        """Recompute every pitch's `align_distance` from the current string-edit
        alignment (call after analyze()/detect_mistakes()).

        Unlike `_update_pitch_distances`, which keys each pitch off the score note
        sitting at its absolute time, this keys off the note pairing the string
        edit chose:
          - deletion (no user note): nothing to color, skipped.
          - insertion (user note, no score match): all its pitches -> inf (red).
          - good / substitution: distance to the *aligned* score note's pitch.

        High-slope transition frames (flagged by detect_transitions) are always
        skipped — their pitch is mid-slide and unreliable, so they stay None
        (grey) rather than dragging a note's coloring or showing as a mistake.
        Other voiced pitches not covered by any aligned note default to green
        (TRANSITION_DISTANCE) rather than the live per-frame distance. Truly
        empty/unvoiced frames stay None. So in post-analysis every *drawn*,
        non-transition pitch has an align_distance; while recording (pitches
        detected fresh) they're all None -> live coloring."""
        # reset first so stale post-analysis colors never linger
        for p in self.pitch_data.data:
            if p is not None:
                p.align_distance = None

        for pair_index, (user_note, midi_note) in enumerate(self.alignment.pairs):
            if user_note is None:
                continue  # deletion: score note with no user pitches to color
            pitches = self.pitch_data.read(
                start_time=user_note.start_time,
                end_time=user_note.end_time,
                clean=True,
            )
            # transition frames are mid-slide: leave them None (grey), never score
            pitches = [p for p in pitches if not p.is_transition]
            if pair_index in self.alignment.overridden_pair_indices:
                # overridden mistake: the user dismissed it, so force its pitches
                # to distance 0 => green, regardless of the underlying mismatch.
                for p in pitches:
                    p.align_distance = 0.0
            elif midi_note is None:
                # insertion: a note that isn't in the score at all -> all red
                for p in pitches:
                    p.align_distance = float('inf')
            else:
                target = midi_note.midi_num[0]
                for p in pitches:
                    p.align_distance = target - p.candidates[0][0]

        # any remaining voiced (drawable) pitch no note covered: color green by
        # default instead of falling back to live coloring -- but skip high-slope
        # transition frames, which stay None (grey) so slides aren't penalized.
        for p in self.pitch_data.data:
            if (p is not None and p.align_distance is None and p.candidates
                    and not p.is_transition):
                p.align_distance = self.TRANSITION_DISTANCE
    
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

        # recolor the affected pitches: overridden notes -> green (distance 0),
        # un-overridden -> back to their real alignment distance.
        self.update_alignment_distances()

    def has_analysis(self):
        """Return True if this recording has been analyzed (notes detected => alignment filled in)"""
        return len(self.note_data.times) > 0

    def _last_voiced_pitch_index(self) -> int | None:
        for i in range(len(self.pitch_data.data) - 1, -1, -1):
            pitch = self.pitch_data.data[i]
            if (pitch is not None
                    and pitch.candidates
                    and pitch.unvoiced_prob < self.pitch_data.UNVOICED_THRESHOLD):
                return i
        return None

    def truncate_end(
        self,
        pad_sec: float = TRAILING_AUDIO_PAD_SEC,
        mark_unsaved: bool = True,
    ) -> bool:
        """Remove trailing silence after the last voiced pitch.

        Pitch times live in app-time, while AudioData stores samples from its
        own `t_origin`; AudioData.truncate_end handles that conversion.
        """
        last_voiced_idx = self._last_voiced_pitch_index()
        if last_voiced_idx is None:
            return False

        last_voiced = self.pitch_data.data[last_voiced_idx]
        trim_time = last_voiced.time + max(0.0, pad_sec)
        audio_changed = self.audio_data.truncate_end(trim_time)

        keep_count = last_voiced_idx + 1
        for i in range(last_voiced_idx + 1, len(self.pitch_data.data)):
            pitch = self.pitch_data.data[i]
            if pitch is None:
                continue
            if pitch.time > trim_time:
                break
            keep_count = i + 1

        pitch_changed = keep_count < len(self.pitch_data.data)
        if pitch_changed:
            with self.pitch_data.lock:
                self.pitch_data.data = self.pitch_data.data[:keep_count]

        if audio_changed and mark_unsaved:
            self.unsaved_changes = True
        return audio_changed or pitch_changed

    def trim_end(self, *args, **kwargs):
        """Compatibility alias for older callers."""
        return self.truncate_end(*args, **kwargs)
