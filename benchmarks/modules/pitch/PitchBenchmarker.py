from __future__ import annotations

import contextlib
import lzma
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

import mir_eval
import numpy as np
import numpy.typing as npt
import pretty_midi

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback, not expected on macOS
    fcntl = None

def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


_BOOTSTRAP_ROOT = _bootstrap_repo_root()
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from benchmarks.paths import REPO_ROOT, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()
ROOT = REPO_ROOT

from algorithms.Config import (  # noqa: E402
    Config,
    PYIN_DEFAULT_MIN_VOLUME,
    PYIN_DEFAULT_MAX_VOLUME,
    PYIN_DEFAULT_UNV_THRESH,
    PYIN_PRAAT_MIRROR_UNV_THRESH,
)
from app_logic.midi.ScoreData import ScoreData  # noqa: E402
from app_logic.NoteData import NoteData  # noqa: E402
from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from app_logic.user.ds.PitchData import Pitch, PitchData  # noqa: E402
from app_logic.user.ds.Recording import Recording  # noqa: E402

PathLike: TypeAlias = str | Path
PitchMetricRow: TypeAlias = dict[str, float | bool | str]

SF_PATH: Path = ROOT / "resources" / "MuseScore_General.sf3"
SR: int = 44100
PITCH_CACHE_VERSION: int = 5
PITCH_RAW_STAGE = "raw"
PITCH_SMOOTHED_STAGE = "smoothed"
PITCH_COMPUTE_COL = "Compute Time (s)"
PITCH_REALTIME_COL = "Audio(s)/Compute(s)"
PITCH_AUDIO_SECONDS_COL = "Audio Length (s)"
PITCH_RESULT_COLUMNS: list[str] = [
    "Track ID",
    "Voicing Recall",
    "Voicing False Alarm",
    "Raw Pitch Accuracy",
    "Raw Chroma Accuracy",
    "Overall Accuracy",
    PITCH_REALTIME_COL,
    PITCH_AUDIO_SECONDS_COL,
    PITCH_COMPUTE_COL,
    "Split",
    "Track",
    "Ensemble",
    "Instrument",
    "Voice",
    "F0_voice",
    "From Cache",
    "Fmin",
    "Fmax",
]


class PitchBenchmarker:
    PITCH_DATASETS: ClassVar[list[str]] = [
        "mdb-stem-synth",
        "mdb-melody-synth",
        "bach10-mf0-synth"
    ]
    PITCH_METRICS: ClassVar[list[str]] = [
        "Raw Pitch Accuracy",
        "Raw Chroma Accuracy",
        "Overall Accuracy",
        "Voicing Recall",
        "Voicing False Alarm",
    ]

    def __init__(self) -> None:
        self.ROOT = ROOT
        self.DATASETS = ROOT / "benchmarks" / "datasets"
        self.RESULTS = ROOT / "benchmarks" / "results"
        self.MISTAKE_DIR = self.DATASETS / "mistake-db"
        self.SOUNDFONT_PATH = SF_PATH
        self.RESULTS.mkdir(parents=True, exist_ok=True)
        self.algorithm_verbose = False
        self.use_cache = True
        self.config_overrides: dict[str, Any] = {}

        self.DEFAULT_CONFIG = Config(
            sr=44100,
            w1=1024*4,
            h1=128,
            fmin=196.0,
            fmax=3000.0,
            tuning=440.0,
            unv_thresh=PYIN_DEFAULT_UNV_THRESH,
            min_volume=PYIN_DEFAULT_MIN_VOLUME,
            max_volume=PYIN_DEFAULT_MAX_VOLUME,
        )

    def config_for(self, fmin: float, fmax: float, **overrides: Any) -> Config:
        all_overrides = {
            **self.config_overrides,
            **overrides,
            "fmin": float(fmin),
            "fmax": float(fmax),
        }
        return replace(
            self.DEFAULT_CONFIG,
            **all_overrides,
        )

    @staticmethod
    def range_from_freqs(freqs: npt.ArrayLike) -> tuple[float, float]:
        voiced = np.asarray(freqs, dtype=float)
        voiced = voiced[voiced > 0]
        if voiced.size == 0:
            return 196.0, 3000.0
        return float(voiced.min()), float(voiced.max())

    @staticmethod
    def range_from_midi(
        midi_numbers: Iterable[float],
        tuning: float = 440.0,
    ) -> tuple[float, float]:
        arr = np.asarray(list(midi_numbers), dtype=float)
        if arr.size == 0:
            return 196.0, 3000.0
        to_hz = lambda m: tuning * 2 ** ((m - 69) / 12.0)
        return to_hz(arr.min()), to_hz(arr.max())

    @staticmethod
    def notedata_to_pm(
        note_data: NoteData,
        program: int = 40,
        velocity: int = 90,
    ) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program)
        for t in note_data.times:
            n = note_data.data[t]
            if not n.midi_num or n.midi_num[0] == -1 or n.end_time <= n.start_time:
                continue
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(n.velocity or velocity),
                    pitch=int(n.midi_num[0]),
                    start=float(n.start_time),
                    end=float(n.end_time),
                )
            )
        pm.instruments.append(inst)
        return pm

    @staticmethod
    def corpus_dir_for_midi(midi_path: PathLike) -> Path:
        midi_path = Path(midi_path)
        if midi_path.parent.name == "midi":
            return midi_path.parent.parent
        return midi_path.parent

    @staticmethod
    def dataset_name_for_midi(midi_path: PathLike) -> str:
        return PitchBenchmarker.corpus_dir_for_midi(midi_path).name

    @staticmethod
    def synth_audio_dir_for_midi(midi_path: PathLike) -> Path:
        return PitchBenchmarker.corpus_dir_for_midi(midi_path) / "synth_audio"

    def result_csv_path(self, algorithm: str, dataset: str) -> Path:
        safe_dataset = dataset.replace("/", "_")
        return self.RESULTS / algorithm / f"{safe_dataset}.csv"

    def pitch_result_csv_path(self, method: str, dataset: str) -> Path:
        safe_dataset = dataset.replace("/", "_")
        return self.RESULTS / "pitch" / "raw_outputs" / method / f"{safe_dataset}.csv"

    @property
    def pitch_summary_csv_path(self) -> Path:
        return self.RESULTS / "pitch" / "pitch_benchmarks.csv"

    @staticmethod
    def display_pitch_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.index.name in (None, "track_id"):
            out.index.name = "Track ID"

        rename_candidates: dict[str, str] = {
            "track_id": "Track ID",
            "realtime_factor": PITCH_REALTIME_COL,
            "audio_seconds": PITCH_AUDIO_SECONDS_COL,
            "pitch_compute_time": PITCH_COMPUTE_COL,
            "Compute Time": PITCH_COMPUTE_COL,
            "split": "Split",
            "track": "Track",
            "ensemble": "Ensemble",
            "instrument": "Instrument",
            "voice": "Voice",
            "f0_voice": "F0_voice",
            "from_cache": "From Cache",
            "fmin": "Fmin",
            "fmax": "Fmax",
        }
        rename = {
            src: dst
            for src, dst in rename_candidates.items()
            if src in out.columns and (dst not in out.columns or src == dst)
        }
        if rename:
            out = out.rename(columns=rename)

        out = out.drop(
            columns=[
                c
                for c in ("pitch_detector_compute_time", "pitch_smoother_compute_time")
                if c in out.columns
            ],
        )
        ordered = [c for c in PITCH_RESULT_COLUMNS if c in out.columns]
        remaining = [c for c in out.columns if c not in ordered]
        return out[ordered + remaining]

    def write_pitch_result(
        self,
        df: pd.DataFrame,
        method: str,
        dataset: str,
        index: bool = True,
    ) -> Path:
        out_path = self.pitch_result_csv_path(method, dataset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = self.display_pitch_columns(df).drop(
            columns=[c for c in ("model", "dataset") if c in df.columns],
            errors="ignore",
        )
        out.to_csv(out_path, index=index)
        return out_path

    def write_dataset_result(
        self,
        df: pd.DataFrame,
        algorithm: str,
        dataset: str,
        index: bool = True,
    ) -> Path:
        out_path = self.result_csv_path(algorithm, dataset)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=index)
        return out_path

    def synth_midi(
        self,
        midi_path: PathLike,
        out_dir: PathLike | None = None,
        gain: float = 1.0,
        force: bool = False,
    ) -> Path:
        midi_path = Path(midi_path)
        out_dir = (
            Path(out_dir)
            if out_dir is not None
            else self.synth_audio_dir_for_midi(midi_path)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (midi_path.stem + ".wav")
        if (
            not force
            and out.exists()
            and out.stat().st_mtime >= midi_path.stat().st_mtime
        ):
            return out
        if shutil.which("fluidsynth") is None:
            raise RuntimeError("fluidsynth CLI not found (brew install fluid-synth).")
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                # Disable reverb (-R) and chorus (-C): on a sustained violin patch
                # their tails ring the FIRST/LAST notes out past the MIDI note-off,
                # extending the voiced region with no following note to mask it.
                # That asymmetric lengthening skews the two-point onset-span fit in
                # resize_score (last onset reads late -> score compressed), so the
                # cleaner the decay, the truer the detected onsets.
                "-R", "0",
                "-C", "0",
                "-g",
                str(gain),
                "-F",
                str(out),
                "-r",
                str(SR),
                str(SF_PATH),
                str(midi_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out

    def recording_for(
        self,
        config: Config,
        score_notes: NoteData | None = None,
        score_data: ScoreData | None = None,
    ) -> Recording:
        from algorithms.PitchDetector import PitchDetector
        from algorithms.PitchSmoother import PitchSmoother

        if score_data is not None:
            rec = Recording(score_data=score_data, config=config)
            rec.pitch_detector = PitchDetector(recording=rec)
            rec.pitch_smoother = PitchSmoother(recording=rec)
            return rec

        rec = Recording(config=config)
        rec.pitch_detector = PitchDetector(recording=rec)
        rec.pitch_smoother = PitchSmoother(recording=rec)
        if score_notes is not None:
            rec.score_data.note_datas = {0: score_notes}
            rec.score_data.active_instrument = 0
            rec.score_data.clip = None
            rec.active_instrument = 0
        return rec

    def pitch_cache_path(self, corpus_dir: PathLike, track_id: str) -> Path:
        safe_track_id = track_id.replace("/", "_")
        return Path(corpus_dir) / "pitch_data" / f"{safe_track_id}.pitch.pkl.xz"

    def cache_path_for_wav(self, wav_path: PathLike) -> Path:
        """Cache path for a track's wav. Single seam for cache-path derivation so
        subclasses with a different on-disk layout (e.g. CocoChorales) can override
        it without touching ``bench_pitch_track`` or the overnight runner."""
        wav_path = Path(wav_path)
        return self.pitch_cache_path(wav_path.parents[1], wav_path.stem)

    @staticmethod
    def _pitch_to_payload(pitch: Pitch | None) -> dict[str, Any] | None:
        if pitch is None:
            return None
        pitch.ensure_compatible()
        return {
            "time": float(pitch.time),
            "value": float(pitch.value),
            "candidates": [
                (float(midi), float(prob)) for midi, prob in pitch.candidate_pitches
            ],
            "volume": float(pitch.volume),
            "unvoiced_prob": float(pitch.unvoiced_prob),
            "distance": float(getattr(pitch, "live_distance", 0.0) or 0.0),
            "align_distance": getattr(pitch, "aligned_distance", None),
            "is_transition": getattr(pitch, "is_transition", None),
        }

    @staticmethod
    def _pitch_from_payload(payload: dict[str, Any] | None, config: Config) -> Pitch | None:
        if payload is None:
            return None
        if isinstance(payload, Pitch):
            return payload.ensure_compatible(config)
        if not isinstance(payload, dict):
            raw_candidates = payload[1] if len(payload) > 1 else []
            pitch = Pitch(
                time=float(payload[0]) if len(payload) > 0 else 0.0,
                candidates=[(float(midi), float(prob)) for midi, prob in raw_candidates],
                value=float(payload[7]) if len(payload) > 7 and payload[7] is not None else -1,
                volume=float(payload[2]) if len(payload) > 2 else 0.0,
                unvoiced_prob=float(payload[3]) if len(payload) > 3 else 1.0,
                live_distance=(None if len(payload) <= 4 or payload[4] is None else float(payload[4])),
                config=config,
            )
            pitch.aligned_distance = None if len(payload) <= 5 else payload[5]
            pitch.is_transition = None if len(payload) <= 6 else payload[6]
            return pitch.ensure_compatible(config)

        raw_candidates = (
            payload.get("candidate_pitches")
            or payload.get("candidates")
            or []
        )
        raw_value = payload.get("value", -1.0)
        pitch = Pitch(
            time=float(payload["time"]),
            candidates=[(float(midi), float(prob)) for midi, prob in raw_candidates],
            value=-1 if raw_value is None else float(raw_value),
            volume=float(payload["volume"]),
            unvoiced_prob=float(payload["unvoiced_prob"]),
            live_distance=payload.get("live_distance", payload.get("distance")),
            config=config,
        )
        pitch.aligned_distance = payload.get(
            "aligned_distance",
            payload.get("align_distance"),
        )
        pitch.is_transition = payload.get("is_transition")
        return pitch.ensure_compatible(config)

    @staticmethod
    def _pitch_stage(smooth: bool) -> str:
        return PITCH_SMOOTHED_STAGE if smooth else PITCH_RAW_STAGE

    @staticmethod
    def _timing_defaults(
        timing: dict[str, Any],
        smooth: bool,
        raw_timing: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        detector_time = float(timing.get("pitch_detector_compute_time", 0.0) or 0.0)
        if smooth and raw_timing:
            raw_detector_time = float(
                raw_timing.get("pitch_detector_compute_time", 0.0) or 0.0
            )
            if raw_detector_time > 0:
                detector_time = raw_detector_time

        smoother_time = float(timing.get("pitch_smoother_compute_time", 0.0) or 0.0)
        component_compute = detector_time + smoother_time if smooth else detector_time
        stored_compute = float(timing.get("pitch_compute_time", 0.0) or 0.0)
        compute_time = component_compute if component_compute > 0 else stored_compute
        return {
            "pitch_detector_compute_time": detector_time,
            "pitch_smoother_compute_time": smoother_time if smooth else 0.0,
            "pitch_compute_time": compute_time,
        }

    @contextlib.contextmanager
    def _pitch_cache_lock(self, cache_path: Path):
        lock_path = cache_path.with_name(f"{cache_path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a", encoding="utf-8") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def _pitch_data_to_stage_payload(self, pitch_data: PitchData) -> dict[str, Any]:
        return {
            "t_origin": float(pitch_data.t_origin),
            "pitches": [self._pitch_to_payload(p) for p in pitch_data.data],
        }

    def _stage_payload_to_pitch_data(
        self,
        payload: dict[str, Any],
        config: Config,
    ) -> PitchData:
        pitch_data = PitchData(config=config)
        pitch_data.t_origin = float(payload.get("t_origin", 0.0))
        pitch_data.data = [
            self._pitch_from_payload(p, config) for p in payload.get("pitches", [])
        ]
        return pitch_data

    @staticmethod
    def _load_pitch_cache_payload(cache_path: PathLike) -> dict[str, Any]:
        with lzma.open(cache_path, "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def _normalise_pitch_cache_payload(
        payload: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        version = payload.get("version")
        if version == 1:
            return (
                {
                    PITCH_SMOOTHED_STAGE: {
                        "t_origin": float(payload.get("t_origin", 0.0)),
                        "pitches": payload.get("pitches", []),
                    },
                },
                {PITCH_SMOOTHED_STAGE: dict(payload.get("metadata") or {})},
            )
        if version != PITCH_CACHE_VERSION:
            raise ValueError(f"Unsupported pitch cache version: {version}")

        raw_stages = payload.get("stages") or {}
        raw_metadata = payload.get("metadata") or {}
        stages = {
            str(stage): dict(stage_payload or {})
            for stage, stage_payload in raw_stages.items()
        }
        metadata: dict[str, dict[str, Any]] = {}
        for stage in stages:
            stage_metadata = raw_metadata.get(stage, {}) if isinstance(raw_metadata, dict) else {}
            metadata[stage] = dict(stage_metadata or {})
        return stages, metadata

    def has_pitch_cache(self, cache_path: PathLike, smooth: bool = True) -> bool:
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False
        try:
            payload = self._load_pitch_cache_payload(cache_path)
            stages, _ = self._normalise_pitch_cache_payload(payload)
        except Exception:  # noqa: BLE001 -- corrupt caches should be recomputed
            return False
        return self._pitch_stage(smooth) in stages

    def save_pitch_data(
        self,
        pitch_data: PitchData,
        cache_path: PathLike,
        metadata: dict[str, Any]=None,
        stage: str = PITCH_SMOOTHED_STAGE,
    ) -> Path:
        return self.save_pitch_cache(
            cache_path,
            stages={stage: pitch_data},
            metadata={stage: metadata or {}},
        )

    def save_pitch_cache(
        self,
        cache_path: PathLike,
        stages: dict[str, PitchData],
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = metadata or {}

        with self._pitch_cache_lock(cache_path):
            merged_stages: dict[str, dict[str, Any]] = {}
            merged_metadata: dict[str, dict[str, Any]] = {}
            if cache_path.exists():
                try:
                    payload = self._load_pitch_cache_payload(cache_path)
                    merged_stages, merged_metadata = self._normalise_pitch_cache_payload(payload)
                except Exception:  # noqa: BLE001 -- replace corrupt/incompatible caches
                    merged_stages = {}
                    merged_metadata = {}

            for stage, pitch_data in stages.items():
                merged_stages[stage] = self._pitch_data_to_stage_payload(pitch_data)
                merged_metadata[stage] = dict(metadata.get(stage, {}))

            payload = {
                "version": PITCH_CACHE_VERSION,
                "metadata": merged_metadata,
                "stages": merged_stages,
            }
            tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            try:
                with lzma.open(tmp_path, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path.replace(cache_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        return cache_path

    def load_pitch_data(
        self,
        cache_path: PathLike,
        config: Config,
        smooth: bool = True,
    ) -> tuple[PitchData, dict[str, Any]]:
        payload = self._load_pitch_cache_payload(cache_path)
        stages, metadata = self._normalise_pitch_cache_payload(payload)
        stage = self._pitch_stage(smooth)
        if stage not in stages:
            raise ValueError(f"Pitch cache does not contain {stage!r} data: {cache_path}")
        pitch_data = self._stage_payload_to_pitch_data(stages[stage], config)
        return pitch_data, self._timing_defaults(
            metadata.get(stage, {}),
            smooth=smooth,
            raw_timing=metadata.get(PITCH_RAW_STAGE),
        )

    def _apply_pitch_origin(self, pitch_data: PitchData, origin: float) -> PitchData:
        pitch_data.t_origin = origin
        if origin:
            for p in pitch_data.data:
                if p is not None:
                    p.time += origin
        return pitch_data

    def detect_pitch_stages(
        self,
        rec: Recording,
        make_smoothed: bool = True,
        verbose: bool = False,
    ) -> tuple[dict[str, PitchData], dict[str, dict[str, float]]]:
        audio = rec.audio_data.read_all()

        detector_start = time.perf_counter()
        raw_pitches = rec.pitch_detector.detect_pitches(
            audio,
            show_progress=verbose,
            verbose=verbose,
        )
        detector_compute_time = time.perf_counter() - detector_start
        gate_stats = dict(getattr(rec.pitch_detector, "_last_volume_gate_stats", {}) or {})

        origin = rec.audio_data.t_origin
        smoothed_pitches = None
        smoother_compute_time = 0.0
        if make_smoothed:
            smoother_start = time.perf_counter()
            smoothed_pitches = rec.pitch_smoother.smooth(raw_pitches, verbose=verbose)
            smoother_compute_time = time.perf_counter() - smoother_start

        raw_data = PitchData(config=rec.config)
        raw_data.data = raw_pitches
        self._apply_pitch_origin(raw_data, origin)

        stages = {PITCH_RAW_STAGE: raw_data}
        timing = {
            PITCH_RAW_STAGE: {
                "pitch_detector_compute_time": detector_compute_time,
                "pitch_smoother_compute_time": 0.0,
                "pitch_compute_time": detector_compute_time,
                **gate_stats,
            },
        }

        if smoothed_pitches is not None:
            smoothed_data = PitchData(config=rec.config)
            smoothed_data.data = smoothed_pitches
            self._apply_pitch_origin(smoothed_data, origin)
            stages[PITCH_SMOOTHED_STAGE] = smoothed_data
            timing[PITCH_SMOOTHED_STAGE] = {
                "pitch_detector_compute_time": detector_compute_time,
                "pitch_smoother_compute_time": smoother_compute_time,
                "pitch_compute_time": detector_compute_time + smoother_compute_time,
                **gate_stats,
            }

        return stages, timing

    def detect_pitches(
        self,
        rec: Recording,
        smooth: bool = True,
        verbose: bool = False,
    ) -> dict[str, float]:
        stages, timing = self.detect_pitch_stages(
            rec,
            make_smoothed=smooth,
            verbose=verbose,
        )
        stage = self._pitch_stage(smooth)
        rec.pitch_data = stages[stage]
        return timing[stage]

    def load_or_detect_pitches(
        self,
        rec: Recording,
        cache_path: PathLike,
        smooth: bool = True,
        use_cache: bool = True,
        write_cache: bool = True,
        verbose: bool = False,
    ) -> dict[str, float]:
        cache_path = Path(cache_path)
        if use_cache and cache_path.exists():
            try:
                rec.pitch_data, metadata = self.load_pitch_data(
                    cache_path,
                    rec.config,
                    smooth=smooth,
                )
                return metadata
            except Exception:  # noqa: BLE001 -- missing/corrupt stage gets recomputed
                pass

        stages, timing = self.detect_pitch_stages(
            rec,
            make_smoothed=smooth,
            verbose=verbose,
        )
        stage = self._pitch_stage(smooth)
        rec.pitch_data = stages[stage]
        if write_cache:
            self.save_pitch_cache(cache_path, stages=stages, metadata=timing)
        return timing[stage]

    @staticmethod
    def pitchdata_to_melody(
        pitch_data: PitchData,
        config: Config,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        times, freqs = [], []
        for p in pitch_data.data:
            if p is None:
                continue
            times.append(p.time)
            voiced = p.value != -1 and p.unvoiced_prob < config.unv_thresh
            freqs.append(config.midi_to_freq(p.value) if voiced else 0.0)
        return np.asarray(times, dtype=float), np.asarray(freqs, dtype=float)

    def iter_tracks(self, dataset: str) -> Iterator[tuple[str, Path, Path]]:
        audio_dir = self.DATASETS / dataset / "audio"
        annot_dir = self.DATASETS / dataset / "annot"
        for wav_path in sorted(audio_dir.glob("*.wav")):
            annot_path = annot_dir / (wav_path.stem + ".csv")
            if annot_path.exists():
                yield wav_path.stem, wav_path, annot_path

    @staticmethod
    def parse_annot(
        annot_path: PathLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        import pandas as pd

        df = pd.read_csv(annot_path, sep=r"\s+|,", header=None, engine="python")
        return df[0].to_numpy(float), df[1].to_numpy(float)

    def _limit(self, items: list[Any], max_tracks: int | None) -> list[Any]:
        if max_tracks is None:
            return items
        return items[:max_tracks]

    def bench_pitch_track(
        self,
        wav_path: PathLike,
        annot_path: PathLike,
        tighten_pitch_range: bool = True,
        smooth: bool = True,
        fmin: float = 196.0,
        fmax: float = 3000.0,
    ) -> PitchMetricRow:
        ref_times, ref_freqs = self.parse_annot(annot_path)
        if tighten_pitch_range:
            fmin, fmax = self.range_from_freqs(ref_freqs)
        cfg = self.config_for(fmin, fmax)
        rec = self.recording_for(cfg)
        rec.audio_data = AudioData(
            audio_filepath=str(wav_path),
            config=rec.config,
        )
        cache_path = self.cache_path_for_wav(wav_path)
        timing = self.load_or_detect_pitches(
            rec,
            cache_path=cache_path,
            smooth=smooth,
            use_cache=getattr(self, "use_cache", True),
            write_cache=True,
            verbose=self.algorithm_verbose,
        )
        est_times, est_freqs = self.pitchdata_to_melody(rec.pitch_data, cfg)
        metrics = {
            k: float(v)
            for k, v in mir_eval.melody.evaluate(
                ref_times, ref_freqs, est_times, est_freqs
            ).items()
        }
        return {
            **metrics,
            **timing,
            "fmin": float(fmin),
            "fmax": float(fmax),
        }

    def bench_pitch_dataset(
        self,
        dataset: str,
        max_tracks: int | None = None,
        tighten_pitch_range: bool = True,
        smooth: bool = True,
        verbose: bool = True,
        write: bool = False,
    ) -> pd.DataFrame:
        import pandas as pd

        tracks = self._limit(list(self.iter_tracks(dataset)), max_tracks)
        rows = []
        for i, (title, wav_path, annot_path) in enumerate(tracks):
            row = self.bench_pitch_track(
                wav_path,
                annot_path,
                tighten_pitch_range=tighten_pitch_range,
                smooth=smooth,
            )
            row["track_id"] = title
            rows.append(row)
            if verbose:
                print(
                    f"[{dataset}] {i+1:>3}/{len(tracks)} {title[:36]:36s} "
                    f"RPA={row['Raw Pitch Accuracy']:.3f} "
                    f"OA={row['Overall Accuracy']:.3f}"
                )
        df = pd.DataFrame(rows).set_index("track_id")
        if write:
            method = "pyin_smoothed" if smooth else "pyin"
            self.write_pitch_result(df, method, dataset)
        return df

    @staticmethod
    def summarize(
        df: pd.DataFrame,
        cols: Sequence[str] | None = None,
        name: str = "",
    ) -> pd.Series:
        import pandas as pd

        num = df.select_dtypes(include=[np.number])
        if cols:
            num = num[[c for c in cols if c in num.columns]]
        means = num.mean(numeric_only=True)
        if name:
            print(f"\n=== {name}: mean over {len(df)} rows ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(means.to_string())
        return means
