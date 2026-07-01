from __future__ import annotations

import lzma
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

from algorithms.Config import Config  # noqa: E402
from app_logic.midi.ScoreData import ScoreData  # noqa: E402
from app_logic.NoteData import NoteData  # noqa: E402
from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from app_logic.user.ds.PitchData import Pitch, PitchData  # noqa: E402
from app_logic.user.ds.Recording import Recording  # noqa: E402

PathLike: TypeAlias = str | Path
PitchMetricRow: TypeAlias = dict[str, float | bool | str]

SF_PATH: Path = ROOT / "resources" / "MuseScore_General.sf3"
SR: int = 44100
PITCH_CACHE_VERSION: int = 1
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

        self.DEFAULT_CONFIG = Config(
            sr=44100,
            w1=1024*4,
            h1=128,
            fmin=196.0,
            fmax=3000.0,
            tuning=440.0,
            unv_thresh=0.9,
            pitch_thresh=0.5,
            ins_cost=5,
            del_cost=5,
            pitch_tolerance=1,
            timing_tolerance=0.05,
        )

    def config_for(self, fmin: float, fmax: float, **overrides: Any) -> Config:
        return replace(
            self.DEFAULT_CONFIG,
            fmin=float(fmin),
            fmax=float(fmax),
            **overrides,
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
        if score_data is not None:
            rec = Recording(score_data=score_data, config=config)
            rec.sync_min_note_length_from_score()
            return rec

        rec = Recording(config=config)
        if score_notes is not None:
            rec.score_data.note_datas = {0: score_notes}
            rec.score_data.active_instrument = 0
            rec.score_data.clip = None
            rec.active_instrument = 0
            rec.sync_min_note_length_from_score()
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
        return {
            "time": float(pitch.time),
            "candidates": [
                (float(midi), float(prob)) for midi, prob in pitch.candidates
            ],
            "volume": float(pitch.volume),
            "unvoiced_prob": float(pitch.unvoiced_prob),
            "distance": float(getattr(pitch, "distance", 0.0) or 0.0),
            "align_distance": getattr(pitch, "align_distance", None),
            "is_transition": getattr(pitch, "is_transition", None),
        }

    @staticmethod
    def _pitch_from_payload(payload: dict[str, Any] | None, config: Config) -> Pitch | None:
        if payload is None:
            return None
        pitch = Pitch(
            time=float(payload["time"]),
            candidates=[(float(midi), float(prob)) for midi, prob in payload["candidates"]],
            volume=float(payload["volume"]),
            unvoiced_prob=float(payload["unvoiced_prob"]),
            distance=float(payload.get("distance", 0.0) or 0.0),
            config=config,
        )
        pitch.align_distance = payload.get("align_distance")
        pitch.is_transition = payload.get("is_transition")
        return pitch

    def save_pitch_data(
        self, pitch_data: PitchData, cache_path: PathLike, metadata: dict[str, Any]=None,
    ) -> Path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": PITCH_CACHE_VERSION,
            "metadata": metadata or {},
            "t_origin": float(pitch_data.t_origin),
            "pitches": [self._pitch_to_payload(p) for p in pitch_data.data],
        }
        with lzma.open(cache_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_path

    def load_pitch_data(
        self,
        cache_path: PathLike,
        config: Config,
    ) -> tuple[PitchData, dict[str, Any]]:
        with lzma.open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != PITCH_CACHE_VERSION:
            raise ValueError(f"Unsupported pitch cache version: {payload.get('version')}")
        pitch_data = PitchData(config=config)
        pitch_data.t_origin = float(payload.get("t_origin", 0.0))
        pitch_data.data = [
            self._pitch_from_payload(p, config) for p in payload.get("pitches", [])
        ]
        return pitch_data, dict(payload.get("metadata") or {})

    def detect_pitches(
        self,
        rec: Recording,
        smooth: bool = True,
    ) -> dict[str, float]:
        audio = rec.audio_data.read_all()
        rec.pitch_data = PitchData(config=rec.config)

        detector_start = time.perf_counter()
        rec.pitch_data.data = rec.pitch_detector.detect_pitches(audio)
        detector_compute_time = time.perf_counter() - detector_start

        smoother_compute_time = 0.0
        if smooth:
            smoother_start = time.perf_counter()
            rec.pitch_data.data = rec.pitch_smoother.smooth(rec.pitch_data.data)
            smoother_compute_time = time.perf_counter() - smoother_start

        origin = rec.audio_data.t_origin
        rec.pitch_data.t_origin = origin
        if origin:
            for p in rec.pitch_data.data:
                if p is not None:
                    p.time += origin

        return {
            "pitch_detector_compute_time": detector_compute_time,
            "pitch_smoother_compute_time": smoother_compute_time,
            "pitch_compute_time": detector_compute_time + smoother_compute_time,
        }

    def load_or_detect_pitches(
        self,
        rec: Recording,
        cache_path: PathLike,
        smooth: bool = True,
        write_cache: bool = True,
    ) -> dict[str, float]:
        cache_path = Path(cache_path)
        if smooth and cache_path.exists():
            rec.pitch_data, metadata = self.load_pitch_data(cache_path, rec.config)
            return {
                "pitch_detector_compute_time": float(
                    metadata.get("pitch_detector_compute_time", 0.0)
                ),
                "pitch_smoother_compute_time": float(
                    metadata.get("pitch_smoother_compute_time", 0.0)
                ),
                "pitch_compute_time": float(metadata.get("pitch_compute_time", 0.0)),
            }

        timing = self.detect_pitches(rec, smooth=smooth)
        if smooth and write_cache:
            self.save_pitch_data(rec.pitch_data, cache_path, metadata=dict(timing))
        return timing

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
            voiced = p.candidates and p.unvoiced_prob < config.unv_thresh
            freqs.append(config.midi_to_freq(p.candidates[0][0]) if voiced else 0.0)
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
            write_cache=smooth,
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
