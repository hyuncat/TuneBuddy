import numpy as np
import threading
import time
from pathlib import Path
from typing import Literal

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData, Pitch
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Alignment
from app_logic.NoteData import NoteData
from app_logic.user.ds.Buffer import Buffer
from app_logic.JsonHandler import JsonHandler
from algorithms.Config import Config

ResizeSpan = Literal["pitch", "note", "onset", "raw"]

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
        from algorithms.MistakeDetector import MistakeDetector
        from algorithms.MistakeChecker import MistakeChecker
        self.pitch_detector = PitchDetector(recording=self)
        self.pitch_smoother = PitchSmoother(recording=self)
        self.note_detector = NoteDetector(recording=self)
        self.mistake_detector = MistakeDetector(recording=self)
        self.mistake_checker = MistakeChecker(recording=self)

        # essential data variables
        self.audio_data = AudioData(config=self.config)
        self.pitch_data = PitchData(config=self.config)
        self.note_data = NoteData()
        self.onset_data = None
        self.onset_detector = None
        self.alignment: Alignment = Alignment(config=self.config) # filled in later
        self.overridden_mistake_indices = set()

        # Persistence metadata. Folder/library entries point at files until their
        # parent score is active; live takes become dirty as soon as audio is
        # written and are clean again after save_audio().
        self.audio_filepath: Path | None = None
        self.unsaved_changes = False
        self.loaded_from_cache = False

        # queue data structures for real time pitch detection
        self.a2p_queue = Buffer(self.config.sr) #audio-to-pitches

    def update_config(self, config: Config=None):
        """initialize the config, either with a provided one or a default one"""
        if config is None:
            self.config = Config()
        else:
            self.config = config

        self.sync_min_note_length_from_score()
            
        if hasattr(self, 'pitch_detector'):
            self.pitch_detector.load_config(self.config)
        if hasattr(self, 'pitch_smoother'):
            self.pitch_smoother.update_config(self.config)
        if hasattr(self, 'note_detector'):
            self.note_detector.update_config(self.config)
        if hasattr(self, 'onset_detector') and self.onset_detector is not None:
            self.onset_detector.update_config(self.config)
        if hasattr(self, 'onset_data') and self.onset_data is not None:
            self.onset_data.config = self.config
        if hasattr(self, 'mistake_detector'):
            self.mistake_detector.update_config(self.config)
        if hasattr(self, 'mistake_checker'):
            self.mistake_checker.update_config(self.config)
    # def on_pitches_detected(self, pitches):
    #     self.pitch_data.data = pitches

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
        # The pitch pipeline stamps frame times as i*h1/config.sr and converts YIN
        # lags to frequency with config.sr too. If that rate disagrees with the
        # audio file's REAL rate, BOTH drift: pitch times stretch (the track slides
        # "later and later" over its length) and detected pitches come out mistuned
        # (e.g. a 48kHz file read at 44.1k lands ~1.5 semitones flat). The audio
        # file is the source of truth, so pin config.sr to it — and re-init the
        # detectors + pitch grid — before detection runs.
        self._sync_sr_to_audio()
        self.audio_filepath = path
        self.unsaved_changes = False
        self.loaded_from_cache = False
        self.pitch_data = PitchData(config=self.config)
        self.reset_analysis()
        if load_cache:
            self.load_cache(score_filepath=score_filepath, recording_name=recording_name)

    def _sync_sr_to_audio(self):
        """Pin config.sr to the loaded audio's real sample rate and propagate it to
        the detectors/pitch grid, so pitch times + frequencies are computed on the
        audio's own sample grid rather than the 44.1k default. No-op when they
        already match. (A stale cache saved at a different sr overrides this again
        on load — see JsonHandler; such caches must be re-detected to be correct.)"""
        audio_sr = int(getattr(self.audio_data, "sr", 0) or 0)
        if audio_sr and audio_sr != self.config.sr:
            self.config.sr = audio_sr
            self.update_config(self.config)

    def save_audio(self, audio_filepath: str | Path):
        """Persist this recording's current audio buffer to disk."""
        self.trim_end(mark_unsaved=False)
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
        self, score_filepath: str | Path=None, recording_name: str=None,
    ) -> bool:
        return JsonHandler(self).save_cache(
            score_filepath=score_filepath,
            recording_name=recording_name,
        )

    def load_cache(
        self,
        score_filepath: str | Path=None,
        recording_name: str=None,
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
        self.onset_data = None
        self.alignment = Alignment(config=self.config)
        self.overridden_mistake_indices = set()

    def sync_min_note_length_from_score(self) -> float:
        """Refresh Config.min_note_length from the active score/clip NoteData."""
        if not hasattr(self, "config") or self.config is None:
            return 0.0
        try:
            note_data = self.score_data.clipped_note_data(channel=self.active_instrument)
        except (AttributeError, KeyError, TypeError):
            return self.config.min_note_length
        return self.config.set_min_note_length_from_notedata(note_data)

    def detect_pitches(self, on_phase=None):
        """run pitch detection, then smoothing, on the current audio data.
        `on_phase(text)`, if given, is called at the start of each stage so a
        caller can surface progress (e.g. a status-bar message)."""
        audio = self.audio_data.read_all()
        stop_status = self._phase_status_timer(on_phase, "Detecting pitches")
        try:
            self.pitch_data.data = self.pitch_detector.detect_pitches(
                audio,
                show_progress=False,
            )
        finally:
            stop_status()

        stop_status = self._phase_status_timer(on_phase, "Smoothing pitches")
        try:
            self.pitch_data.data = self.pitch_smoother.smooth(
                self.pitch_data.data,
                verbose=False,
            )
        finally:
            stop_status()
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

    @staticmethod
    def _phase_status_timer(on_phase, label: str):
        """Emit a per-phase elapsed-time status until the current phase finishes."""
        if on_phase is None:
            return lambda: None

        stop_event = threading.Event()
        start = time.perf_counter()

        def message() -> str:
            return f"{label}... {time.perf_counter() - start:.1f}s"

        def tick():
            while not stop_event.wait(0.25):
                on_phase(message())

        on_phase(message())
        thread = threading.Thread(target=tick, daemon=True)
        thread.start()

        def stop():
            stop_event.set()
            thread.join(timeout=0.5)
            on_phase(message())

        return stop

    def detect_notes(self):
        """Run PELT note detection on the current pitch data."""
        self.note_data = self.note_detector.detect_notes(self.pitch_data, refine_with_onsets=True)

    def detect_transitions(self):
        """Flag high-slope pitch-transition frames before PELT note detection."""
        self.note_detector.detect_transitions(self.pitch_data)

    def recompute_note_pitches(self):
        """Re-median detected notes over non-transition frames."""
        self.note_detector.recompute_note_pitches(self.note_data, self.pitch_data)

    def prune_transition_notes(self):
        """Drop detected notes that are mostly transition frames."""
        self.note_data = self.note_detector.prune_transition_notes(
            self.note_data,
            self.pitch_data,
        )

    def detect_mistakes(self, onset_aware: bool = False):
        # The MistakeDetector only ever sees the clip's score notes (the full
        # NoteData when unclipped) — see ScoreData.clipped_note_data.
        # onset_aware swaps in the time-anchored aligner (A/B against pitch-only).
        user_notes = self.note_data
        midi_notes = self.score_data.clipped_note_data(channel=self.active_instrument)
        detect = (
            self.mistake_detector.detect_pitch_mistakes_onset_aware
            if onset_aware
            else self.mistake_detector.detect_pitch_mistakes
        )
        notes, mistakes = detect(
            user_string=user_notes,
            midi_string=midi_notes,
        )
        self.alignment.load_alignment(notes, pitch_mistakes=mistakes)
        self.alignment.reapply_overrides(self.overridden_mistake_indices)

    def correct_mistakes(self):
        nd, alignment = self.mistake_checker.check_mistakes(recording=self)
        self.note_data = nd
        self.alignment = alignment
        self.reindex_mistakes()

    def reindex_mistakes(self):
        """Refresh mistake -> alignment-pair indices after note correction
        """
        if self.alignment is None:
            return
        self.alignment.reindex_mistakes()
        self.alignment.reapply_overrides(self.overridden_mistake_indices)

    def write_data(self, indata: np.ndarray, start_time: float):
        """write indata to the audio_data at the given start_time
        and append to our queue for pitch processing
        """
        self.audio_data.write_data(indata, start_time)
        self.unsaved_changes = True
        self.a2p_queue.push(indata)

    def write_pitch_data(self, indata: list[Pitch], start_time: float):
        """Write detected pitches to pitch_data at the given start_time."""
        self.pitch_data.write(indata, start_time)

    def get_length(self, raw=True):
        if raw:
            if len(self.note_data.times) > 0:
                return self.note_data.get_length()
            else:
                return self.audio_data.get_length()
        bounds = self.note_data.get_bounds(clean=True)
        if bounds is None:
            return 0.0
        start, end = bounds
        return end - start

    def audio_bounds(self) -> tuple[float, float] | None:
        """App-time bounds for the recording audio currently considered live."""
        if self.audio_data.end_index <= 0:
            return None
        return self.audio_data.get_bounds()

    def audio_end_time(self) -> float:
        """App-time of the recording's logical audio end."""
        return self.audio_data.get_end_time()
    
    def shift(self, delta: float):
        """Slide the WHOLE recorded take (audio, pitches, notes) by `delta` sec on
        the app-time line. Audio/pitch frames move via their shared time origin (no
        array copy); notes are rekeyed. General primitive; NOTE resize_score no
        longer calls it — the take is kept fixed and the score is moved onto it
        instead (so audio/pitch stay indexed in their own app-time)."""
        if not delta:
            return
        self.audio_data.t_origin += delta
        self.pitch_data.t_origin += delta
        for p in self.pitch_data.data:
            if p is not None:
                p.time += delta
        self.note_data.shift(delta)

    def resize_score(
        self,
        to_span: ResizeSpan = "pitch",
        respect_clip: bool = True,
        include_transitions: bool = True,
    ) -> bool:
        """Stretch the score to a recording span and move the score ONTO the take.

        Args:
            to_span:
                "pitch" uses first/last voiced pitch frames. This pre-note pass
                gives score-derived note-length heuristics a tempo close to the
                take before PELT computes its min_size.
                "note" uses the detected voiced note span for the final resize.
                "onset" uses the first/last voiced note ONSETS (start-to-start,
                ignoring the final note's duration), and matches the score
                onset-to-onset too. Useful when the take's last note is held
                longer or shorter than the score's — the onsets still line up.
                "raw" uses the full detected-note span, or full audio span if
                notes have not been detected yet.
            respect_clip:
                If true and a score clip exists, match the clip span to the take
                and anchor the clip's first note onto the take's first voiced
                note. Otherwise match the whole active-instrument score span and
                anchor the score's first note onto the take's. The take never
                moves — it is the app-time truth; only the score is stretched and
                shifted to fit it.
            include_transitions:
                Only used for to_span="pitch". When false, high-slope transition
                frames flagged by detect_transitions() are ignored when finding
                the first/last voiced frame.
        """
        if to_span == "pitch":
            bounds = self.pitch_data.get_voiced_range(
                include_transitions=include_transitions,
            )
        elif to_span == "note":
            bounds = self.note_data.get_bounds(clean=True, use_note_end=True)
        elif to_span == "onset":
            bounds = self.note_data.get_bounds(clean=True, use_note_end=False)
        elif to_span == "raw":
            bounds = self.audio_bounds()
        else:
            raise ValueError(f"unknown resize span: {to_span!r}")

        if not bounds or bounds[1] <= bounds[0]:
            return False

        start, end = bounds
        target_span = end - start
        take_anchor_time = start  # the take's first voiced app-time; the take STAYS here

        sd = self.score_data
        # "onset" fits the score onset-to-onset, so measure the score span the
        # same way (first note start -> last note start); every other mode fits
        # against note ends.
        score_use_note_end = to_span != "onset"
        score_bounds = sd.get_bounds(
            channel=self.active_instrument,
            respect_clip=respect_clip,
            use_note_end=score_use_note_end,
        )
        if score_bounds is None:
            return False
        score_span = score_bounds[1] - score_bounds[0]
        if score_span <= 0:
            return False

        # Stretch the relevant score span to the user's recording span. Keep the
        # bpm FRACTIONAL: rounding to an integer quantizes the scale (~0.8%/bpm at
        # 120 => ~0.5s drift over a 60s piece), which lands the score's endpoint
        # off the take's even when the spans were measured perfectly.
        new_bpm = max(1.0, sd.bpm * score_span / target_span)
        sd.change_tempo(new_bpm)

        score_bounds = sd.get_bounds(
            channel=self.active_instrument,
            respect_clip=respect_clip,
            use_note_end=score_use_note_end,
        )
        if score_bounds is None:
            return False
        # Move the SCORE onto the fixed take: land the (clip) first note ON the
        # take's first voiced note, instead of dragging the take to t=0. The take
        # never moves, so its audio/pitch/note timeline stays the single app-time
        # truth the slider and audio player index against (no negative t_origin,
        # no shift — which is what used to desync the waveform from the cursor).
        # transpose_notes takes an absolute offset from the untransposed baseline;
        # (sd.transpose_offset - score_bounds[0]) re-bases it even when change_tempo
        # no-ops, and + take_anchor_time then anchors the span onto the take.
        sd.transpose_notes(sd.transpose_offset - score_bounds[0] + take_anchor_time)

        self.sync_min_note_length_from_score()
        self._update_pitch_distances()
        return True

    def change_tempo(self, new_bpm: float):
        """Change the tempo of the recording by changing the BPM of the score data, which will automatically update the note timings and pitch distances."""
        self.score_data.change_tempo(new_bpm)
        self.sync_min_note_length_from_score()
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
        """Recompute every pitch's `align_distance` from the current pitch-mistake
        alignment (call after analyze()/detect_mistakes()) with the following coloring:
          - deletion: nothing to color, skipped
          - insertion: all pitches -> inf (red)
          - good/substitution: distance to the *aligned* score note's pitch.
        """
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
        # error checking
        if not (0 <= mistake_index < len(self.alignment.pitch_mistakes)):
            return
        # toggle persisted override state for one mistake
        mistake = self.alignment.pitch_mistakes[mistake_index]
        pair_index = mistake.get_pair_index()
        if mistake_index in self.overridden_mistake_indices:
            self.overridden_mistake_indices.remove(mistake_index)
            self.alignment.toggle_overridden_pair_indices(pair_index, False)
            overridden = False
        else:
            self.overridden_mistake_indices.add(mistake_index)
            self.alignment.toggle_overridden_pair_indices(pair_index, True)
            overridden = True

        if 0 <= mistake_index < len(self.alignment.pitch_mistakes):
            self.alignment.pitch_mistakes[mistake_index].set_override(overridden)

        # recolor the affected pitches: overridden notes -> green (distance 0)
        # un-overridden -> back to their real alignment distance
        self.update_alignment_distances()

    def has_analysis(self):
        """Return True if this recording has been analyzed (notes detected => alignment filled in)"""
        return len(self.note_data.times) > 0

    def trim_end(
        self, pad_sec: float = TRAILING_AUDIO_PAD_SEC, mark_unsaved: bool = True,
    ) -> bool:
        """Remove trailing silence after the last voiced pitch.

        Pitch times live in app-time, while AudioData stores samples from its
        own `t_origin`; AudioData.trim_end handles that conversion.
        """
        voiced_range = self.pitch_data.get_voiced_range()
        if voiced_range is None:
            return False

        _, voiced_end = voiced_range
        trim_time = voiced_end + max(0.0, pad_sec)
        audio_changed = self.audio_data.trim_end(trim_time)

        keep_count = 0
        for i, pitch in enumerate(self.pitch_data.data):
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


    # --- JSON LOADING / SAVING WRAPPERS ---
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
