from __future__ import annotations

from dataclasses import fields
import base64
import gzip
import json
import lzma
import math
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from algorithms.Config import Config
from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import Note, NoteData
from app_logic.midi.ScoreData import ScoreData
from app_logic.user.ds.PitchData import Pitch, PitchData
from app_logic.user.ds.TimbreData import TimbreData

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording


class JsonHandler:
    """Read/write Attune recording JSON sidecars.

    The sidecar stores derived analysis and per-recording score state; audio
    remains in its original file and score content remains in the score file.
    """

    CACHE_VERSION = 1
    # v2: spectral onsets moved to score-guided deletion correction.
    # v3: unvoiced pitch-run splitting follows min_gap_factor * min_note_length.
    # v4: explicit adaptive spectral-flux candidates plus onset-corroborated
    # marginal PELT boundaries.
    # v5: adjacent KernelCPD segments are no longer merged after segmentation.
    # v6: spectral onsets removed; any unvoiced frame initially split a run.
    # v7: pitch runs instead require a majority-unvoiced three-frame gap.
    # v8: mistake detection/correction consistently uses duration-scaled,
    # score-time-aware string editing.
    # v9: matched-onset refits are re-aligned/corrected to convergence; deletion
    # recovery can jointly repartition both neighbors across unvoiced dropouts.
    # v10: time-aware string editing uses the primary user pitch plus weighted
    # absolute onset/duration errors; gap operations add unmatched durations.
    # v11: score-time stabilization uses a robust matched-onset fit, so a stray
    # release/vibrato fragment cannot dictate the tempo from one endpoint;
    # score-aware correction also folds contiguous same-pitch fragments into
    # their aligned neighbor and leaves excess length as a timing error.
    # v12: insertion/deletion timing cost uses the full unmatched duration;
    # alpha_duration applies only when onset and duration errors can be blended.
    # v13: production alignment weights follow the runner-v2 holdout result.
    # v14: deletion recovery may preserve a near-pitch substitution and jointly
    # repartition following repeated notes instead of cascading the alignment.
    # v15: KernelCPD uses explicit score-relative minimum segment length and a
    # millisecond-valued strict-majority silence window.
    # v16: production selects pitch_thresh=0.75 and removes the obsolete PELT
    # stride from Config.
    NOTE_ANALYSIS_VERSION = 17
    CACHE_SUFFIX = ".json.xz"
    GZIP_CACHE_SUFFIX = ".json.gz"
    LEGACY_CACHE_SUFFIX = ".json"

    def __init__(self, recording: Recording | None = None):
        self.recording = recording

    @classmethod
    def cache_path_for(cls, audio_filepath: str | Path) -> Path:
        """Preferred hidden compressed JSON sidecar path for an audio file.

        `take.wav` -> `.take.json.xz`, while keeping the audio extension out of
        the cache name. Gzip `.take.json.gz` and plain `.take.json` files are
        still accepted as legacy fallbacks.
        """
        path = Path(audio_filepath)
        return path.with_name(f".{path.stem}{cls.CACHE_SUFFIX}")

    @classmethod
    def legacy_cache_path_for(cls, audio_filepath: str | Path) -> Path:
        path = Path(audio_filepath)
        return path.with_name(f".{path.stem}{cls.LEGACY_CACHE_SUFFIX}")

    @classmethod
    def gzip_cache_path_for(cls, audio_filepath: str | Path) -> Path:
        path = Path(audio_filepath)
        return path.with_name(f".{path.stem}{cls.GZIP_CACHE_SUFFIX}")

    @classmethod
    def cache_paths_for(cls, audio_filepath: str | Path) -> tuple[Path, Path, Path]:
        return (
            cls.cache_path_for(audio_filepath),
            cls.gzip_cache_path_for(audio_filepath),
            cls.legacy_cache_path_for(audio_filepath),
        )

    @classmethod
    def existing_cache_path_for(cls, audio_filepath: str | Path) -> Path | None:
        for path in cls.cache_paths_for(audio_filepath):
            if path.exists() and path.is_file():
                return path
        return None

    @classmethod
    def delete_cache_for(cls, audio_filepath: str | Path) -> None:
        for path in cls.cache_paths_for(audio_filepath):
            cls._unlink_cache_file(path)

    @classmethod
    def rename_recording_files(cls, recording: Recording, new_stem: str) -> Path | None:
        """Rename the recording audio file and hidden sidecar immediately."""
        if recording.audio_filepath is None:
            return None
        old_audio = Path(recording.audio_filepath)
        if not old_audio.exists():
            raise FileNotFoundError(old_audio)

        new_audio = old_audio.with_name(f"{new_stem}{old_audio.suffix}")
        old_cache = cls.existing_cache_path_for(old_audio)
        new_cache = cls.cache_path_for(new_audio)
        cache_existed = old_cache is not None

        if new_audio != old_audio and new_audio.exists():
            raise FileExistsError(new_audio)
        for candidate in cls.cache_paths_for(new_audio):
            if old_cache is not None and candidate == old_cache:
                continue
            if candidate.exists():
                raise FileExistsError(candidate)

        if new_audio != old_audio:
            old_audio.rename(new_audio)
            recording.audio_filepath = new_audio
        if cache_existed and old_cache is not None:
            payload = cls._read_json_payload(old_cache)
            cls._update_payload_audio_metadata(payload, recording, new_stem)
            cls._write_cache_payload(new_cache, payload)
            for path in (*cls.cache_paths_for(old_audio), *cls.cache_paths_for(new_audio)):
                if path != new_cache:
                    cls._unlink_cache_file(path)
        return recording.audio_filepath

    @classmethod
    def _update_cache_audio_metadata(
        cls,
        cache_path: Path,
        recording: Recording,
        recording_name: str,
    ) -> None:
        if recording.audio_filepath is None or not cache_path.exists():
            return
        try:
            payload = cls._read_json_payload(cache_path)
            cls._update_payload_audio_metadata(payload, recording, recording_name)
            cls._write_cache_payload(cache_path, payload)
        except Exception as e:
            print(f"Could not update recording cache metadata '{cache_path}': {e}")

    def save_cache(
        self,
        recording: Recording | None = None,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> bool:
        rec = self._recording(recording)
        path = self.cache_path_for_recording(rec)
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.to_cache_payload(rec, score_filepath, recording_name)
            self._write_cache_payload(path, payload)
            if rec.audio_filepath is not None:
                self._unlink_cache_file(self.gzip_cache_path_for(rec.audio_filepath))
                self._unlink_cache_file(self.legacy_cache_path_for(rec.audio_filepath))
            return True
        except Exception as e:
            print(f"Could not save recording cache '{path}': {e}")
            return False

    def load_cache(
        self,
        recording: Recording | None = None,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> bool:
        rec = self._recording(recording)
        path = self.existing_cache_path_for_recording(rec)
        if path is None:
            return False
        try:
            payload = self._read_json_payload(path)
            if payload.get("version") != self.CACHE_VERSION:
                return False
            if not self._cache_matches_audio(rec, payload):
                return False
            if not self._cache_matches_score(payload, score_filepath):
                return False
            self.load_cache_payload(rec, payload, score_filepath=score_filepath)
            rec.unsaved_changes = False
            rec.loaded_from_cache = True
            print(f"Loaded recording cache: {path}")
            return True
        except Exception as e:
            print(f"Could not load recording cache '{path}': {e}")
            return False

    @classmethod
    def cache_path_for_recording(cls, recording: Recording) -> Path | None:
        return cls.cache_path_for(recording.audio_filepath) if recording.audio_filepath else None

    @classmethod
    def existing_cache_path_for_recording(cls, recording: Recording) -> Path | None:
        return cls.existing_cache_path_for(recording.audio_filepath) if recording.audio_filepath else None

    def to_cache_payload(
        self,
        recording: Recording | None = None,
        score_filepath: str | Path | None = None,
        recording_name: str | None = None,
    ) -> dict:
        rec = self._recording(recording)
        audio_path = Path(rec.audio_filepath) if rec.audio_filepath else None
        score_path = Path(score_filepath) if score_filepath is not None else getattr(rec.score_data, "filepath", None)
        score_path = Path(score_path) if score_path is not None else None
        return {
            "version": self.CACHE_VERSION,
            "recording": {
                "name": recording_name,
                "audio_file": audio_path.name if audio_path else None,
                "audio_path": str(audio_path) if audio_path else None,
                "audio_size": audio_path.stat().st_size if audio_path and audio_path.exists() else None,
                "audio_mtime_ns": audio_path.stat().st_mtime_ns if audio_path and audio_path.exists() else None,
                "active_instrument": rec.active_instrument,
                "audio_t_origin": self._pack_number(rec.audio_data.t_origin),
                "audio_end_index": int(rec.audio_data.end_index),
            },
            "score": self._score_to_payload(rec, score_path),
            "config": self._config_to_payload(rec.config),
            "note_analysis": {
                "version": self.NOTE_ANALYSIS_VERSION,
                "segmentation_config": rec.config.note_segmentation_config(),
            },
            "pitch_data": self._pitch_data_to_payload(rec),
            "timbre": self._timbre_to_payload(rec.timbre_data),
            "note_data": self._note_data_to_payload(rec.note_data),
            "alignment": self._alignment_to_payload(rec),
            "overridden_mistake_indices": sorted(int(i) for i in rec.overridden_mistake_indices),
        }

    def load_cache_payload(
        self,
        recording: Recording,
        payload: dict,
        score_filepath: str | Path | None = None,
    ) -> None:
        self._restore_score_from_payload(recording, payload.get("score") or {}, score_filepath)

        rec_payload = payload.get("recording") or {}
        active = rec_payload.get("active_instrument")
        if active in recording.score_data.note_datas:
            recording.score_data.active_instrument = int(active)
            recording.active_instrument = int(active)
        else:
            recording.active_instrument = recording.score_data.active_instrument

        # Per-take/user-facing settings still come from the sidecar, but note
        # segmentation and alignment-model settings are code-owned defaults.
        # Otherwise changing production costs in Config appears to do nothing
        # for every cached take.
        runtime_config = recording.config
        cached_config = self._config_from_payload(payload.get("config") or {})
        changed_analysis_config = []
        code_owned_fields = (
            *Config.NOTE_SEGMENTATION_FIELDS,
            *Config.ALIGNMENT_FIELDS,
        )
        for name in code_owned_fields:
            effective = getattr(runtime_config, name)
            cached = getattr(cached_config, name)
            if cached != effective:
                changed_analysis_config.append((name, cached, effective))
                setattr(cached_config, name, effective)
        recording.update_config(cached_config)

        recording.audio_data.t_origin = self._unpack_number(rec_payload.get("audio_t_origin"), default=0.0)
        end_index = rec_payload.get("audio_end_index")
        if isinstance(end_index, int) and end_index >= 0:
            recording.audio_data.end_index = min(end_index, len(recording.audio_data.data))

        recording.pitch_data = self._pitch_data_from_payload(recording, payload.get("pitch_data") or {})
        recording.timbre_data = self._timbre_from_payload(
            recording, payload.get("timbre") or {})
        note_meta = payload.get("note_analysis") or {}
        cached_analysis_version = int(
            # Sidecars predating note-analysis metadata contain boundaries made
            # by the old fixed 100 ms gap rule, so they must be rebuilt too.
            note_meta.get("version", 0)
        )
        invalidate_notes = (
            bool(changed_analysis_config)
            or cached_analysis_version != self.NOTE_ANALYSIS_VERSION
        )
        effective = recording.config.note_segmentation_config()
        print(f"Effective note segmentation config: {effective}")
        if invalidate_notes:
            recording.reset_analysis()
            details = ", ".join(
                f"{name} {old:g} -> {new:g}"
                for name, old, new in changed_analysis_config
            )
            if cached_analysis_version != self.NOTE_ANALYSIS_VERSION:
                details = (details + ", " if details else "") + "analysis version changed"
            recording.analysis_notice = (
                f"Cached note analysis is stale ({details}); click Analyze to recompute."
            )
            print(recording.analysis_notice)
        else:
            recording.note_data = self._note_data_from_payload(payload.get("note_data") or [])
            recording.alignment = self._alignment_from_payload(recording, payload.get("alignment") or {})
            recording.overridden_mistake_indices = {
                int(i) for i in payload.get("overridden_mistake_indices", [])
                if isinstance(i, int)
            }
            recording.alignment.reapply_overrides(recording.overridden_mistake_indices)

    def _recording(self, recording: Recording | None) -> Recording:
        rec = recording or self.recording
        if rec is None:
            raise ValueError("JsonHandler requires a Recording.")
        return rec

    @staticmethod
    def _json_bytes(payload: dict) -> bytes:
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @classmethod
    def _read_json_payload(cls, path: Path) -> dict:
        if path.name.endswith(cls.CACHE_SUFFIX):
            with lzma.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        if path.name.endswith(cls.GZIP_CACHE_SUFFIX):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def _write_cache_payload(cls, path: Path, payload: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = cls._json_bytes(payload)
        if path.name.endswith(cls.CACHE_SUFFIX):
            tmp.write_bytes(lzma.compress(data, preset=6))
        elif path.name.endswith(cls.GZIP_CACHE_SUFFIX):
            tmp.write_bytes(gzip.compress(data, compresslevel=6))
        else:
            tmp.write_bytes(data + b"\n")
        tmp.replace(path)

    @staticmethod
    def _unlink_cache_file(path: Path) -> None:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError as e:
            print(f"Could not delete recording cache '{path}': {e}")

    @staticmethod
    def _update_payload_audio_metadata(
        payload: dict,
        recording: Recording,
        recording_name: str,
    ) -> None:
        audio_path = Path(recording.audio_filepath)
        meta = payload.setdefault("recording", {})
        meta["name"] = recording_name
        meta["audio_file"] = audio_path.name
        meta["audio_path"] = str(audio_path)
        if audio_path.exists():
            stat = audio_path.stat()
            meta["audio_size"] = stat.st_size
            meta["audio_mtime_ns"] = stat.st_mtime_ns

    @staticmethod
    def _cache_matches_audio(recording: Recording, payload: dict) -> bool:
        if recording.audio_filepath is None or not recording.audio_filepath.exists():
            return False
        meta = payload.get("recording") or {}
        cached_name = meta.get("audio_file")
        if cached_name and cached_name != recording.audio_filepath.name:
            return False
        cached_size = meta.get("audio_size")
        cached_mtime = meta.get("audio_mtime_ns")
        stat = recording.audio_filepath.stat()
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

    def _score_to_payload(self, recording: Recording, score_path: Path | None) -> dict:
        sd = recording.score_data
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
        recording: Recording,
        payload: dict,
        score_filepath: str | Path | None = None,
    ) -> None:
        score_path = score_filepath or payload.get("path") or getattr(recording.score_data, "filepath", None)
        if score_path and self._score_needs_cache_reset(recording, score_path):
            recording.score_data.load(score_path)

        sd = recording.score_data
        title = payload.get("title")
        if title:
            sd.set_title(title)

        active = payload.get("active_instrument")
        if active in sd.note_datas:
            sd.active_instrument = int(active)

        transpose_semitones = int(payload.get("transpose_semitones") or 0)
        if transpose_semitones and sd.first_note_midi() is not None:
            sd.transpose(dy=transpose_semitones)

        bpm = self._unpack_number(payload.get("bpm"), default=sd.bpm)
        if bpm and abs(float(bpm) - float(sd.bpm)) > 1e-9:
            sd.change_tempo(float(bpm))

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

    @staticmethod
    def _score_needs_cache_reset(recording: Recording, score_path: str | Path) -> bool:
        if recording.score_data.score is None:
            return True
        current = getattr(recording.score_data, "filepath", None)
        try:
            if current is None or Path(current).expanduser().resolve() != Path(score_path).expanduser().resolve():
                return True
        except OSError:
            return True
        return (
            recording.score_data.clip is not None
            or recording.score_data.transpose_semitones != 0
            or abs(recording.score_data.transpose_offset) > 1e-9
            or abs(recording.score_data.bpm - recording.score_data.bpm_og) > 1e-9
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
        payload = dict(payload)
        if "pitch_thresh" not in payload and "pitch_step_semitones" in payload:
            payload["pitch_thresh"] = payload["pitch_step_semitones"]
        if "pitch_tolerance" not in payload and "tolerance" in payload:
            payload["pitch_tolerance"] = payload["tolerance"]
        if "timing_tolerance" not in payload:
            if "timing_onset_tol" in payload:
                payload["timing_tolerance"] = payload["timing_onset_tol"]
            elif "timing_dur_tol" in payload:
                payload["timing_tolerance"] = payload["timing_dur_tol"]
        valid = {f.name for f in fields(Config)}
        defaults = Config()
        kwargs = {}
        for k in payload:
            if k not in valid:
                continue
            value = self._unpack_number(payload[k])
            default_value = getattr(defaults, k)
            if isinstance(default_value, bool):
                value = bool(value)
            elif isinstance(default_value, int):
                value = int(round(value))
            kwargs[k] = value
        # Removed legacy fields such as h2 and note_detection_pelt_jump fall
        # through the valid-field filter. The short-lived explicit pitch-step
        # spelling is migrated to pitch_thresh above.
        return Config(**kwargs)

    def _pitch_data_to_payload(self, recording: Recording) -> dict:
        last = -1
        for i in range(len(recording.pitch_data.data) - 1, -1, -1):
            if recording.pitch_data.data[i] is not None:
                last = i
                break
        pitches = [] if last < 0 else [
            self._pitch_to_payload(p) for p in recording.pitch_data.data[:last + 1]
        ]
        return {
            "t_origin": self._pack_number(recording.pitch_data.t_origin),
            "pitches": pitches,
        }

    def _pitch_data_from_payload(self, recording: Recording, payload: dict) -> PitchData:
        pd = PitchData(config=recording.config)
        pd.t_origin = self._unpack_number(payload.get("t_origin"), default=0.0)
        pd.data = [self._pitch_from_payload(recording, p) for p in payload.get("pitches", [])]
        if not pd.data:
            pd = PitchData(config=recording.config)
        return pd

    def _pitch_to_payload(self, pitch: Pitch | None):
        if pitch is None:
            return None
        return [
            self._pack_number(pitch.time),
            [[self._pack_number(m), self._pack_number(prob)] for m, prob in pitch.candidate_pitches],
            self._pack_number(pitch.volume),
            self._pack_number(pitch.unvoiced_prob),
            self._pack_number(pitch.live_distance),
            self._pack_number(pitch.aligned_distance),
            pitch.is_transition,
            self._pack_number(pitch.value),
        ]

    @staticmethod
    def _timbre_to_payload(data: TimbreData):
        with data.lock:
            n_cols = int(data.computed_until)
            if n_cols <= 0 or not data.written[:n_cols].any():
                return None
            vals = data.values[:, :n_cols].copy()
            missing = ~data.written[:n_cols]
        if missing.any():
            vals[:, missing] = data.floor_db
        # One byte per bin/column. Offset by the -120 dB floor so 0..240
        # represents -120..0 dB in 0.5 dB steps.
        quantized = np.rint((np.clip(vals, data.floor_db, 0.0) - data.floor_db) * 2.0)
        quantized = quantized.astype(np.uint8)
        return {
            "stride": int(data.stride),
            "t_origin": JsonHandler._pack_number(data.t_origin),
            "midi_min": int(data.midi_min),
            "midi_max": int(data.midi_max),
            "n_cols": n_cols,
            "floor_db": data.floor_db,
            "step_db": 0.5,
            "blob": base64.b64encode(quantized.tobytes(order="C")).decode("ascii"),
        }

    def _timbre_from_payload(self, recording: Recording, payload: dict) -> TimbreData:
        td = TimbreData(config=recording.config)
        if not payload or not payload.get("blob"):
            return td
        if (int(payload.get("stride", td.stride)) != td.stride
                or int(payload.get("midi_min", td.midi_min)) != td.midi_min
                or int(payload.get("midi_max", td.midi_max)) != td.midi_max
                or float(payload.get("step_db", 0.5)) != 0.5):
            return td
        td.t_origin = self._unpack_number(payload.get("t_origin"), default=0.0)
        raw = base64.b64decode(payload["blob"], validate=True)
        td.load_quantized(np.frombuffer(raw, dtype=np.uint8), int(payload.get("n_cols", 0)))
        return td

    def _pitch_from_payload(self, recording: Recording, payload) -> Pitch | None:
        if payload is None:
            return None
        if isinstance(payload, Pitch):
            return payload.ensure_compatible(recording.config)
        if isinstance(payload, dict):
            raw_candidates = (
                payload.get("candidate_pitches")
                or payload.get("candidates")
                or []
            )
            candidates = [
                (self._unpack_number(c[0], default=0.0), self._unpack_number(c[1], default=0.0))
                for c in raw_candidates
            ]
            pitch = Pitch(
                time=self._unpack_number(payload.get("time"), default=0.0),
                candidates=candidates,
                value=self._unpack_number(payload.get("value"), default=-1),
                volume=self._unpack_number(payload.get("volume"), default=0.0),
                unvoiced_prob=self._unpack_number(payload.get("unvoiced_prob"), default=1.0),
                live_distance=self._unpack_number(
                    payload.get("live_distance", payload.get("distance")),
                    default=None,
                ),
                config=recording.config,
            )
            pitch.aligned_distance = self._unpack_number(
                payload.get("aligned_distance", payload.get("align_distance")),
                default=None,
            )
            pitch.is_transition = payload.get("is_transition")
            return pitch.ensure_compatible(recording.config)

        candidates = [
            (self._unpack_number(c[0], default=0.0), self._unpack_number(c[1], default=0.0))
            for c in (payload[1] if len(payload) > 1 else [])
        ]
        pitch = Pitch(
            time=self._unpack_number(payload[0], default=0.0),
            candidates=candidates,
            volume=self._unpack_number(payload[2], default=0.0),
            unvoiced_prob=self._unpack_number(payload[3], default=1.0),
            live_distance=self._unpack_number(payload[4] if len(payload) > 4 else None, default=None),
            value=self._unpack_number(payload[7] if len(payload) > 7 else None, default=-1),
            config=recording.config,
        )
        pitch.aligned_distance = self._unpack_number(payload[5] if len(payload) > 5 else None, default=None)
        pitch.is_transition = payload[6] if len(payload) > 6 else None
        return pitch.ensure_compatible(recording.config)

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

    def _compatible_note_end_time(
        self,
        payload: dict,
        start_time: float,
        *,
        end_key: str = "end_time",
        duration_key: str = "duration",
    ) -> float:
        """Read an end time, accepting legacy duration-only note payloads."""
        end_time = self._unpack_number(payload.get(end_key), default=None)
        if end_time is not None:
            return end_time
        duration = self._unpack_number(payload.get(duration_key), default=0.0)
        return start_time + max(0.0, duration)

    def _note_from_payload(self, payload: list | dict) -> Note:
        if isinstance(payload, dict):
            start_time = self._unpack_number(
                payload.get("start_time"),
                default=0.0,
            )
            end_time = self._compatible_note_end_time(payload, start_time)
            note = Note(
                i=int(payload.get("id", 0)),
                start_time=start_time,
                end_time=end_time,
                midi_num=[
                    self._unpack_number(m, default=-1)
                    for m in payload.get("midi_num", [])
                ],
                velocity=payload.get("velocity"),
                instrument=payload.get("instrument"),
            )
            note.base_start_time = self._unpack_number(
                payload.get("base_start_time"),
                default=note.start_time,
            )
            note.base_end_time = self._compatible_note_end_time(
                payload,
                note.base_start_time,
                end_key="base_end_time",
                duration_key="base_duration",
            )
            if (
                payload.get("base_end_time") is None
                and payload.get("base_duration") is None
            ):
                note.base_end_time = note.end_time
            return note

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

    def _alignment_to_payload(self, recording: Recording) -> dict:
        user_notes = recording.note_data.read(i=0, j=len(recording.note_data.times))
        score_nd = recording.score_data.note_datas.get(recording.active_instrument)
        score_notes = score_nd.read(i=0, j=len(score_nd.times)) if score_nd else []
        user_index = self._note_index_maps(user_notes)
        score_index = self._note_index_maps(score_notes)

        def uidx(note):
            return self._lookup_note_index(note, user_index)

        def sidx(note):
            return self._lookup_note_index(note, score_index)

        return {
            "pairs": [[uidx(u), sidx(s)] for u, s in recording.alignment.pairs],
            "pitch_mistakes": [
                self._mistake_to_payload(m, uidx, sidx)
                for m in recording.alignment.pitch_mistakes
            ],
            "timing_mistakes": [
                self._mistake_to_payload(m, uidx, sidx)
                for m in recording.alignment.timing_mistakes
            ],
        }

    def _alignment_from_payload(self, recording: Recording, payload: dict) -> Alignment:
        user_notes = recording.note_data.read(i=0, j=len(recording.note_data.times))
        score_nd = recording.score_data.note_datas.get(recording.active_instrument)
        score_notes = score_nd.read(i=0, j=len(score_nd.times)) if score_nd else []

        def note_at(notes, idx):
            if idx is None:
                return None
            return notes[idx] if isinstance(idx, int) and 0 <= idx < len(notes) else None

        pairs = [
            (note_at(user_notes, uidx), note_at(score_notes, sidx))
            for uidx, sidx in payload.get("pairs", [])
        ]
        def mistake_from_payload(item):
            if len(item) < 5:
                return None
            mistake = Mistake(
                type=item[0],
                user_note=note_at(user_notes, item[1]),
                midi_note=note_at(score_notes, item[2]),
            )
            if item[3] is not None:
                mistake.set_pair_index(int(item[3]))
            mistake.set_override(bool(item[4]))
            if len(item) > 5:
                mistake.info = str(item[5] or "")
            return mistake

        pitch_payload = payload.get("pitch_mistakes", payload.get("mistakes", []))
        pitch_mistakes = [
            mistake
            for item in pitch_payload
            if (mistake := mistake_from_payload(item)) is not None
        ]
        timing_mistakes = [
            mistake
            for item in payload.get("timing_mistakes", [])
            if (mistake := mistake_from_payload(item)) is not None
        ]

        alignment = Alignment(config=recording.config)
        alignment.load_alignment(
            pairs,
            pitch_mistakes=pitch_mistakes,
            timing_mistakes=timing_mistakes,
        )
        return alignment

    @staticmethod
    def _mistake_to_payload(mistake: Mistake, user_index, score_index) -> list:
        return [
            mistake.type,
            user_index(mistake.user_note),
            score_index(mistake.midi_note),
            int(mistake.get_pair_index()) if mistake.get_pair_index() is not None else None,
            bool(mistake.is_overridden()),
            mistake.info,
        ]

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
        if isinstance(value, bool):
            return value
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
