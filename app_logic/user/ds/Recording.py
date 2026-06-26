import numpy as np
from pathlib import Path
from dataclasses import fields
import json
import math

from app_logic.user.ds.AudioData import AudioData
from app_logic.user.ds.PitchData import PitchData, Pitch
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import NoteData, Note
from app_logic.user.ds.Buffer import Buffer
from algorithms.Config import Config

class Recording:
    CACHE_VERSION = 1

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
        """Hidden JSON sidecar path for an audio file.

        `take.wav` -> `.take.json`, matching the requested `.filename.json`
        convention while keeping the audio file extension out of the cache name.
        """
        path = Path(audio_filepath)
        return path.with_name(f".{path.stem}.json")

    @staticmethod
    def delete_cache_for(audio_filepath: str | Path) -> None:
        path = Recording.cache_path_for(audio_filepath)
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError as e:
            print(f"Could not delete recording cache '{path}': {e}")

    def cache_path(self) -> Path | None:
        return self.cache_path_for(self.audio_filepath) if self.audio_filepath else None

    def audio_file_exists(self) -> bool:
        return self.audio_filepath is not None and Path(self.audio_filepath).exists()

    def rename_files(self, new_stem: str) -> Path | None:
        """Rename the on-disk audio file and its hidden cache sidecar now.

        Returns the new audio path, or None for unsaved in-memory recordings that
        do not have files yet. Raises on filesystem conflicts/failures so the UI
        can revert the tree edit.
        """
        if self.audio_filepath is None:
            return None
        old_audio = Path(self.audio_filepath)
        if not old_audio.exists():
            raise FileNotFoundError(old_audio)

        new_audio = old_audio.with_name(f"{new_stem}{old_audio.suffix}")
        old_cache = self.cache_path_for(old_audio)
        new_cache = self.cache_path_for(new_audio)

        if new_audio != old_audio and new_audio.exists():
            raise FileExistsError(new_audio)
        if new_cache != old_cache and new_cache.exists():
            raise FileExistsError(new_cache)

        if new_audio != old_audio:
            old_audio.rename(new_audio)
            self.audio_filepath = new_audio
        if old_cache.exists() and new_cache != old_cache:
            old_cache.rename(new_cache)
        return self.audio_filepath

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
        """Persist analysis/cache metadata next to the audio file.

        The cache is intentionally derived-state only: audio stays in the audio
        file, score content stays in the score file, and this JSON stores the
        per-recording score state plus pitch/note/alignment results.
        """
        path = self.cache_path()
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._to_cache_payload(score_filepath, recording_name)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp.replace(path)
            return True
        except Exception as e:
            print(f"Could not save recording cache '{path}': {e}")
            return False

    def load_cache(
        self,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> bool:
        path = self.cache_path()
        if path is None or not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != self.CACHE_VERSION:
                return False
            if not self._cache_matches_audio(payload):
                return False
            if not self._cache_matches_score(payload, score_filepath):
                return False
            self._load_cache_payload(payload, score_filepath=score_filepath)
            self.unsaved_changes = False
            self.loaded_from_cache = True
            print(f"Loaded recording cache: {path}")
            return True
        except Exception as e:
            print(f"Could not load recording cache '{path}': {e}")
            return False

    def _cache_matches_audio(self, payload: dict) -> bool:
        if self.audio_filepath is None or not self.audio_filepath.exists():
            return False
        meta = payload.get("recording") or {}
        cached_name = meta.get("audio_file")
        if cached_name and cached_name != self.audio_filepath.name:
            return False
        cached_size = meta.get("audio_size")
        cached_mtime = meta.get("audio_mtime_ns")
        stat = self.audio_filepath.stat()
        if cached_size is not None and int(cached_size) != stat.st_size:
            return False
        if cached_mtime is not None and int(cached_mtime) != stat.st_mtime_ns:
            return False
        return True

    @staticmethod
    def _cache_matches_score(payload: dict, score_filepath: str | Path | None) -> bool:
        if score_filepath is None:
            return True
        cached_file = (payload.get("score") or {}).get("file")
        if not cached_file:
            return True
        return cached_file == Path(score_filepath).name

    # --- JSON CACHE SERIALIZATION ---
    def _to_cache_payload(
        self,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> dict:
        audio_path = Path(self.audio_filepath) if self.audio_filepath else None
        score_path = Path(score_filepath) if score_filepath is not None else getattr(self.score_data, "filepath", None)
        score_path = Path(score_path) if score_path is not None else None
        return {
            "version": self.CACHE_VERSION,
            "recording": {
                "name": recording_name,
                "audio_file": audio_path.name if audio_path else None,
                "audio_path": str(audio_path) if audio_path else None,
                "audio_size": audio_path.stat().st_size if audio_path and audio_path.exists() else None,
                "audio_mtime_ns": audio_path.stat().st_mtime_ns if audio_path and audio_path.exists() else None,
                "active_instrument": self.active_instrument,
                "audio_t_origin": self._pack_number(self.audio_data.t_origin),
                "audio_end_index": int(self.audio_data.end_index),
            },
            "score": self._score_to_payload(score_path),
            "config": self._config_to_payload(self.config),
            "pitch_data": self._pitch_data_to_payload(),
            "note_data": self._note_data_to_payload(self.note_data),
            "alignment": self._alignment_to_payload(),
            "overridden_mistake_indices": sorted(int(i) for i in self.overridden_mistake_indices),
        }

    def _load_cache_payload(
        self,
        payload: dict,
        score_filepath: str | Path | None = None,
    ) -> None:
        self._restore_score_from_payload(payload.get("score") or {}, score_filepath)

        rec_payload = payload.get("recording") or {}
        active = rec_payload.get("active_instrument")
        if active in self.score_data.note_datas:
            self.score_data.active_instrument = int(active)
            self.active_instrument = int(active)
        else:
            self.active_instrument = self.score_data.active_instrument

        self.config = self._config_from_payload(payload.get("config") or {})
        self.update_config(self.config)

        self.audio_data.t_origin = self._unpack_number(rec_payload.get("audio_t_origin"), default=0.0)
        end_index = rec_payload.get("audio_end_index")
        if isinstance(end_index, int) and end_index >= 0:
            self.audio_data.end_index = min(end_index, len(self.audio_data.data))

        self.pitch_data = self._pitch_data_from_payload(payload.get("pitch_data") or {})
        self.note_data = self._note_data_from_payload(payload.get("note_data") or [])
        self.alignment = self._alignment_from_payload(payload.get("alignment") or {})
        self.overridden_mistake_indices = {
            int(i) for i in payload.get("overridden_mistake_indices", [])
            if isinstance(i, int)
        }
        self.alignment.reapply_overrides(self.overridden_mistake_indices)

    def _score_to_payload(self, score_path: Path | None) -> dict:
        sd = self.score_data
        return {
            "path": str(score_path) if score_path is not None else None,
            "file": score_path.name if score_path is not None else None,
            "title": sd.title,
            "bpm": self._pack_number(sd.bpm),
            "bpm_og": self._pack_number(sd.bpm_og),
            "active_instrument": sd.active_instrument,
            "transpose_semitones": sd.transpose_semitones,
            "transpose_offset": self._pack_number(sd.transpose_offset),
            "clip": list(sd.clip) if sd.clip is not None else None,
            "displayed_instruments": sorted(int(ch) for ch in sd.displayed_instruments),
            "playing_instruments": sorted(int(ch) for ch in sd.playing_instruments),
        }

    def _restore_score_from_payload(
        self,
        payload: dict,
        score_filepath: str | Path | None = None,
    ) -> None:
        score_path = score_filepath or payload.get("path") or getattr(self.score_data, "filepath", None)
        if score_path and self._score_needs_cache_reset(score_path):
            self.score_data.load(score_path)

        sd = self.score_data
        title = payload.get("title")
        if title:
            sd.set_title(title)

        active = payload.get("active_instrument")
        if active in sd.note_datas:
            sd.active_instrument = int(active)

        transpose_semitones = int(payload.get("transpose_semitones") or 0)
        if transpose_semitones and sd.first_note_midi() is not None:
            sd.transpose(transpose_semitones)

        bpm = self._unpack_number(payload.get("bpm"), default=sd.bpm)
        if bpm and round(bpm) != round(sd.bpm):
            sd.change_tempo(int(round(bpm)))

        offset = self._unpack_number(payload.get("transpose_offset"), default=0.0)
        if offset:
            sd.transpose_notes(offset)

        clip = payload.get("clip")
        if isinstance(clip, list) and len(clip) == 2:
            sd.set_clip(int(clip[0]), int(clip[1]))
        else:
            sd.clear_clip()

        displayed = self._valid_channels(payload.get("displayed_instruments"), sd)
        if displayed:
            sd.displayed_instruments = displayed
        playing = self._valid_channels(payload.get("playing_instruments"), sd)
        if playing:
            sd.playing_instruments = playing

    def _score_needs_cache_reset(self, score_path: str | Path) -> bool:
        """True when cached state should be applied to a fresh score parse."""
        if self.score_data.score is None:
            return True
        current = getattr(self.score_data, "filepath", None)
        try:
            if current is None or Path(current).expanduser().resolve() != Path(score_path).expanduser().resolve():
                return True
        except OSError:
            return True
        return (
            self.score_data.clip is not None
            or self.score_data.transpose_semitones != 0
            or abs(self.score_data.transpose_offset) > 1e-9
            or round(self.score_data.bpm) != round(self.score_data.bpm_og)
        )

    @staticmethod
    def _valid_channels(value, score_data: ScoreData) -> set[int]:
        if not isinstance(value, list):
            return set()
        channels = set()
        for ch in value:
            try:
                channel = int(ch)
            except (TypeError, ValueError):
                continue
            if channel in score_data.instruments:
                channels.add(channel)
        return channels

    def _config_to_payload(self, config: Config) -> dict:
        return {f.name: self._pack_number(getattr(config, f.name)) for f in fields(Config)}

    def _config_from_payload(self, payload: dict) -> Config:
        valid = {f.name for f in fields(Config)}
        defaults = Config()
        kwargs = {}
        for k in payload:
            if k not in valid:
                continue
            value = self._unpack_number(payload[k])
            if isinstance(getattr(defaults, k), int):
                value = int(round(value))
            kwargs[k] = value
        config = Config(**kwargs)
        if "h2" not in payload:
            config.h2 = config.w2 - 2
        if "slope_thresh" not in payload:
            config.slope_thresh = 0.75 / config.w2
        return config

    def _pitch_data_to_payload(self) -> dict:
        last = -1
        for i in range(len(self.pitch_data.data) - 1, -1, -1):
            if self.pitch_data.data[i] is not None:
                last = i
                break
        pitches = [] if last < 0 else [
            self._pitch_to_payload(p) for p in self.pitch_data.data[:last + 1]
        ]
        return {
            "t_origin": self._pack_number(self.pitch_data.t_origin),
            "pitches": pitches,
        }

    def _pitch_data_from_payload(self, payload: dict) -> PitchData:
        pd = PitchData(config=self.config)
        pd.t_origin = self._unpack_number(payload.get("t_origin"), default=0.0)
        pd.data = [self._pitch_from_payload(p) for p in payload.get("pitches", [])]
        if not pd.data:
            pd = PitchData(config=self.config)
        return pd

    def _pitch_to_payload(self, pitch: Pitch | None):
        if pitch is None:
            return None
        return [
            self._pack_number(pitch.time),
            [[self._pack_number(m), self._pack_number(prob)] for m, prob in pitch.candidates],
            self._pack_number(pitch.volume),
            self._pack_number(pitch.unvoiced_prob),
            self._pack_number(pitch.distance),
            self._pack_number(pitch.align_distance),
            pitch.is_transition,
        ]

    def _pitch_from_payload(self, payload) -> Pitch | None:
        if payload is None:
            return None
        candidates = [
            (self._unpack_number(c[0], default=0.0), self._unpack_number(c[1], default=0.0))
            for c in payload[1]
        ]
        pitch = Pitch(
            time=self._unpack_number(payload[0], default=0.0),
            candidates=candidates,
            volume=self._unpack_number(payload[2], default=0.0),
            unvoiced_prob=self._unpack_number(payload[3], default=1.0),
            distance=self._unpack_number(payload[4], default=None),
            config=self.config,
        )
        pitch.align_distance = self._unpack_number(payload[5], default=None)
        pitch.is_transition = payload[6] if len(payload) > 6 else None
        return pitch

    def _note_data_to_payload(self, note_data: NoteData) -> list:
        return [self._note_to_payload(note_data.data[t]) for t in note_data.times]

    def _note_data_from_payload(self, payload: list) -> NoteData:
        nd = NoteData()
        for item in payload:
            nd.write_note(self._note_from_payload(item))
        return nd

    def _note_to_payload(self, note: Note) -> list:
        return [
            int(note.id),
            self._pack_number(note.start_time),
            self._pack_number(note.end_time),
            [self._pack_number(m) for m in note.midi_num],
            note.velocity,
            note.instrument,
            self._pack_number(note.base_start_time),
            self._pack_number(note.base_end_time),
        ]

    def _note_from_payload(self, payload: list) -> Note:
        note = Note(
            i=int(payload[0]),
            start_time=self._unpack_number(payload[1], default=0.0),
            end_time=self._unpack_number(payload[2], default=0.0),
            midi_num=[self._unpack_number(m, default=-1) for m in payload[3]],
            velocity=payload[4],
            instrument=payload[5],
        )
        if len(payload) > 6:
            note.base_start_time = self._unpack_number(payload[6], default=note.start_time)
        if len(payload) > 7:
            note.base_end_time = self._unpack_number(payload[7], default=note.end_time)
        return note

    def _alignment_to_payload(self) -> dict:
        user_notes = self.note_data.read(i=0, j=len(self.note_data.times))
        score_nd = self.score_data.note_datas.get(self.active_instrument)
        score_notes = score_nd.read(i=0, j=len(score_nd.times)) if score_nd else []
        user_index = self._note_index_maps(user_notes)
        score_index = self._note_index_maps(score_notes)

        def uidx(note):
            return self._lookup_note_index(note, user_index)

        def sidx(note):
            return self._lookup_note_index(note, score_index)

        return {
            "pairs": [[uidx(u), sidx(s)] for u, s in self.alignment.pairs],
            "mistakes": [
                [
                    m.type,
                    uidx(m.user_note),
                    sidx(m.midi_note),
                    int(m.get_pair_index()) if m.get_pair_index() is not None else None,
                    bool(m.is_overridden()),
                ]
                for m in self.alignment.mistakes
            ],
        }

    def _alignment_from_payload(self, payload: dict) -> Alignment:
        user_notes = self.note_data.read(i=0, j=len(self.note_data.times))
        score_nd = self.score_data.note_datas.get(self.active_instrument)
        score_notes = score_nd.read(i=0, j=len(score_nd.times)) if score_nd else []

        def note_at(notes, idx):
            if idx is None:
                return None
            return notes[idx] if isinstance(idx, int) and 0 <= idx < len(notes) else None

        pairs = [
            (note_at(user_notes, uidx), note_at(score_notes, sidx))
            for uidx, sidx in payload.get("pairs", [])
        ]
        mistakes = []
        for item in payload.get("mistakes", []):
            if len(item) < 5:
                continue
            mistake = Mistake(
                type=item[0],
                user_note=note_at(user_notes, item[1]),
                midi_note=note_at(score_notes, item[2]),
            )
            if item[3] is not None:
                mistake.set_pair_index(int(item[3]))
            mistake.set_override(bool(item[4]))
            mistakes.append(mistake)

        alignment = Alignment(config=self.config)
        alignment.load_alignment(pairs, mistakes)
        return alignment

    @staticmethod
    def _note_index_maps(notes: list[Note]) -> dict:
        return {
            "object": {id(note): i for i, note in enumerate(notes)},
            "note_id": {note.id: i for i, note in enumerate(notes)},
        }

    @staticmethod
    def _lookup_note_index(note: Note | None, maps: dict) -> int | None:
        if note is None:
            return None
        by_object = maps["object"].get(id(note))
        if by_object is not None:
            return by_object
        return maps["note_id"].get(getattr(note, "id", None))

    @staticmethod
    def _pack_number(value):
        if value is None:
            return None
        value = float(value)
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value

    @staticmethod
    def _unpack_number(value, default=None):
        if value is None:
            return default
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        if value == "nan":
            return float("nan")
        return float(value)

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
        self.pitch_data.data = self.pitch_detector.detect_pitches(self.audio_data.data)
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
        """run note detection on the current pitch data"""
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
    
    def trim_end(self):
        """Remove trailing silence after the last voiced pitch."""
        last_voiced_idx = None
        for i in range(0, len(self.pitch_data.data)):
            ##index moving backwards
            rev_index = len(self.pitch_data.data)-1-i
            pitch = self.pitch_data.data[rev_index]
            if pitch is not None and pitch.unvoiced_prob < self.pitch_data.UNVOICED_THRESHOLD:
                last_voiced_idx = rev_index
                break

        if last_voiced_idx is None:
            return

        # 200ms buffer just in case
        trim_time = self.pitch_data.data[last_voiced_idx].time + 0.2
        #check against original time
        maximum_time = len(self.audio_data.data) / self.audio_data.sr
        trim_time = min(trim_time, maximum_time)

        with self.pitch_data.lock:
            self.pitch_data.data = self.pitch_data.data[:last_voiced_idx + 1]
        self.audio_data.end_index = int(trim_time * self.audio_data.sr)
