from __future__ import annotations

"""CocoChorales benchmark support.

The useful unit for Attune is one stem:

    main_dataset/<split>/<shard>.tar.bz2:
      <track>/stems_audio/<stem>.wav
      <track>/stems_midi/<stem>.mid
      <track>/metadata.yaml

    f0/<split>/<track>.pickle:
      {0: voice0_f0, 1: voice1_f0, 2: voice2_f0, 3: voice3_f0}

The benchmarker therefore builds a lightweight manifest from the compressed
``main_dataset`` shards and materializes only selected stems. That keeps the
large ``mix.wav`` files and unrelated stems out of the working tree while still
leaving a repeatable path to pitch, note, and mistake benchmarks.
"""

import csv
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import mir_eval
import numpy as np
import numpy.typing as npt

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

from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from benchmarks.modules.pitch.PitchBenchmarker import PathLike, PitchBenchmarker, PitchMetricRow  # noqa: E402

F0_FPS_DEFAULT: float = float(os.environ.get("COCO_F0_FPS", "250"))

_TRACK_RE = re.compile(r"^(?P<ensemble>string|brass|woodwind|random)_track\d+$")
_SPLITS = ("train", "valid", "test")
_MANIFEST_FIELDS = (
    "split",
    "shard",
    "track",
    "ensemble",
    "stem",
    "stem_voice",
    "f0_voice",
    "instrument",
    "wav_member",
    "midi_member",
    "metadata_member",
    "mix_midi_member",
    "f0_path",
)


@dataclass(frozen=True)
class CocoStemRecord:
    split: str
    shard: str
    track: str
    ensemble: str
    stem: str
    stem_voice: int
    f0_voice: int
    instrument: str
    wav_member: str
    midi_member: str
    metadata_member: str
    mix_midi_member: str
    f0_path: str

    @property
    def track_id(self) -> str:
        return f"{self.split}__{self.track}__{self.stem}"

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "CocoStemRecord":
        data = dict(row)
        data["stem_voice"] = int(data["stem_voice"])
        data["f0_voice"] = int(data["f0_voice"])
        return cls(**data)


class CocoChoralesBenchmarker(PitchBenchmarker):
    """Per-stem pitch benchmark over a pruned CocoChorales tiny download."""

    COCO_SPLITS: ClassVar[list[str]] = ["test", "valid", "train"]
    DATASET_LABEL: ClassVar[str] = "cocochorales_tiny"
    DEFAULT_ROOT_NAME: ClassVar[str] = "cocochorales_tiny"
    F0_EXTS: ClassVar[tuple[str, ...]] = (".npy", ".npz", ".csv", ".pkl", ".pickle")

    def __init__(self, root: PathLike | None = None, f0_fps: float = F0_FPS_DEFAULT) -> None:
        super().__init__()
        self.COCO_ROOT = Path(root) if root is not None else self.DATASETS / self.DEFAULT_ROOT_NAME
        self.F0_FPS = float(f0_fps)

    # ------------------------------------------------------------------ paths
    @property
    def manifest_dir(self) -> Path:
        return self.COCO_ROOT / "manifest"

    @property
    def materialized_dir(self) -> Path:
        return self.COCO_ROOT / "materialized"

    def manifest_path(self, split: str) -> Path:
        return self.manifest_dir / f"{split}_stems.csv"

    def shard_path(self, record: CocoStemRecord) -> Path:
        return self.COCO_ROOT / record.shard

    def f0_path_for_track(self, split: str, track_name: str) -> Path | None:
        for ext in (".pickle", ".pkl", ".npz", ".npy", ".csv"):
            candidate = self.COCO_ROOT / "f0" / split / f"{track_name}{ext}"
            if candidate.exists():
                return candidate
        return None

    def f0_path_for_record(self, record: CocoStemRecord) -> Path:
        return self.COCO_ROOT / record.f0_path

    def materialized_track_dir(self, record: CocoStemRecord) -> Path:
        return self.materialized_dir / record.split / record.track

    def materialized_wav_path(self, record: CocoStemRecord) -> Path:
        return self.materialized_track_dir(record) / "stems_audio" / f"{record.stem}.wav"

    def materialized_midi_path(self, record: CocoStemRecord) -> Path:
        return self.materialized_track_dir(record) / "stems_midi" / f"{record.stem}.mid"

    def local_wav_path(self, record: CocoStemRecord) -> Path | None:
        candidates = [
            self.materialized_wav_path(record),
            self.COCO_ROOT / "main_dataset" / record.split / record.track / "stems_audio" / f"{record.stem}.wav",
        ]
        return next((p for p in candidates if p.exists()), None)

    def cache_path_for_track(self, track_id: str) -> Path:
        safe = track_id.replace("/", "_")
        return self.COCO_ROOT / "pitch_data" / f"{safe}.pitch.pkl.xz"

    def cache_path_for_wav(self, wav_path: PathLike) -> Path:
        return self.cache_path_for_track(self.track_id_for_wav(wav_path))

    # --------------------------------------------------------------- naming/meta
    @staticmethod
    def _split_of(path: PathLike) -> str:
        parts = set(Path(path).parts)
        for split in _SPLITS:
            if split in parts:
                return split
        return "unknown"

    @staticmethod
    def _norm_member(name: str) -> str:
        clean = str(PurePosixPath(name))
        while clean.startswith("./"):
            clean = clean[2:]
        if clean in ("", ".") or clean.startswith("../") or "/../" in clean:
            raise ValueError(f"unsafe tar member path: {name!r}")
        return clean

    @staticmethod
    def _voice_idx(stem_path: PathLike) -> int:
        head = Path(stem_path).stem.split("_", 1)[0]
        return int(head) if head.isdigit() else 0

    @staticmethod
    def _instrument(stem_path: PathLike) -> str:
        parts = Path(stem_path).stem.split("_", 1)
        return parts[1] if len(parts) == 2 else "unknown"

    @staticmethod
    def _ensemble_from_track(track_name: str) -> str:
        match = _TRACK_RE.match(track_name)
        return match.group("ensemble") if match else track_name.split("_track")[0]

    @classmethod
    def _ensemble(cls, wav_path: PathLike) -> str:
        return cls._ensemble_from_track(Path(wav_path).parent.parent.name)

    @staticmethod
    def _f0_voice_for_stem(stem_voice: int) -> int:
        # main_dataset stems are 1..4 in the tiny shards, while f0 pickles are
        # keyed 0..3. If a future extraction uses 0..3, leave those unchanged.
        return stem_voice - 1 if 1 <= stem_voice <= 4 else stem_voice

    def track_id_for_wav(self, wav_path: PathLike) -> str:
        wav_path = Path(wav_path)
        track_dir = wav_path.parent.parent
        return f"{self._split_of(track_dir)}__{track_dir.name}__{wav_path.stem}"

    def meta_for_wav(self, wav_path: PathLike) -> dict[str, Any]:
        stem_voice = self._voice_idx(wav_path)
        track_dir = Path(wav_path).parent.parent
        return {
            "split": self._split_of(track_dir),
            "track": track_dir.name,
            "ensemble": self._ensemble(wav_path),
            "instrument": self._instrument(wav_path),
            "voice": stem_voice,
            "f0_voice": self._f0_voice_for_stem(stem_voice),
        }

    # --------------------------------------------------------------- manifest IO
    def read_manifest(self, split: str = "test") -> list[CocoStemRecord]:
        path = self.manifest_path(split)
        if not path.exists():
            return []
        with path.open(newline="") as fh:
            return [CocoStemRecord.from_row(row) for row in csv.DictReader(fh)]

    def write_manifest(self, records: Sequence[CocoStemRecord], split: str) -> Path:
        path = self.manifest_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        return path

    def load_or_build_manifest(self, split: str = "test", rebuild: bool = False) -> list[CocoStemRecord]:
        if split == "all":
            records: list[CocoStemRecord] = []
            for part in _SPLITS:
                records.extend(self.load_or_build_manifest(part, rebuild=rebuild))
            return records

        existing = self.read_manifest(split)
        if existing and not rebuild:
            return existing
        return self.build_manifest(split=split, write=True)

    def build_manifest(self, split: str = "test", write: bool = True) -> list[CocoStemRecord]:
        """Scan retained main_dataset shards and write one row per usable stem."""
        records: dict[str, CocoStemRecord] = {}

        for record in self._scan_extracted_tracks(split):
            records[record.track_id] = record

        for record in self._scan_main_dataset_tars(split):
            records[record.track_id] = record

        out = sorted(records.values(), key=lambda r: (r.split, r.track, r.stem))
        if write:
            self.write_manifest(out, split)
        return out

    def _scan_extracted_tracks(self, split: str) -> Iterator[CocoStemRecord]:
        base = self.COCO_ROOT / "main_dataset"
        if not base.is_dir():
            return
        for stems_audio in sorted(base.glob("**/stems_audio")):
            track_dir = stems_audio.parent
            track_split = self._split_of(track_dir)
            if split not in ("all", track_split):
                continue
            if not _TRACK_RE.match(track_dir.name):
                continue
            f0_src = self.f0_path_for_track(track_split, track_dir.name)
            if f0_src is None:
                continue
            for wav in sorted(stems_audio.glob("*.wav")):
                stem_voice = self._voice_idx(wav)
                stem = wav.stem
                midi = track_dir / "stems_midi" / f"{stem}.mid"
                if not midi.exists():
                    midi = track_dir / "stems_MIDI" / f"{stem}.mid"
                yield CocoStemRecord(
                    split=track_split,
                    shard="",
                    track=track_dir.name,
                    ensemble=self._ensemble_from_track(track_dir.name),
                    stem=stem,
                    stem_voice=stem_voice,
                    f0_voice=self._f0_voice_for_stem(stem_voice),
                    instrument=self._instrument(wav),
                    wav_member=str(wav.relative_to(self.COCO_ROOT)),
                    midi_member=str(midi.relative_to(self.COCO_ROOT)) if midi.exists() else "",
                    metadata_member=str((track_dir / "metadata.yaml").relative_to(self.COCO_ROOT)),
                    mix_midi_member=str((track_dir / "mix.mid").relative_to(self.COCO_ROOT)),
                    f0_path=str(f0_src.relative_to(self.COCO_ROOT)),
                )

    def _scan_main_dataset_tars(self, split: str) -> Iterator[CocoStemRecord]:
        base = self.COCO_ROOT / "main_dataset"
        if not base.is_dir():
            return
        tars = sorted(base.glob("**/*.tar.bz2"))
        for shard in tars:
            shard_split = self._split_of(shard)
            if split not in ("all", shard_split):
                continue
            shard_rel = str(shard.relative_to(self.COCO_ROOT))
            print(f"scanning {shard_rel}", flush=True)
            tar_cmd = shutil.which("bsdtar") or "tar"
            proc = subprocess.Popen(
                [tar_cmd, "-tjf", str(shard)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env={**os.environ, "LC_ALL": "C"},
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                raw_name = line.strip()
                if raw_name in ("", ".", "./"):
                    continue
                name = self._norm_member(raw_name)
                parts = PurePosixPath(name).parts
                if len(parts) != 3 or parts[1] != "stems_audio":
                    continue
                if not parts[2].lower().endswith(".wav"):
                    continue
                track, _, wav_name = parts
                if not _TRACK_RE.match(track):
                    continue
                f0_src = self.f0_path_for_track(shard_split, track)
                if f0_src is None:
                    continue
                stem = Path(wav_name).stem
                stem_voice = self._voice_idx(stem)
                yield CocoStemRecord(
                    split=shard_split,
                    shard=shard_rel,
                    track=track,
                    ensemble=self._ensemble_from_track(track),
                    stem=stem,
                    stem_voice=stem_voice,
                    f0_voice=self._f0_voice_for_stem(stem_voice),
                    instrument=self._instrument(stem),
                    wav_member=name,
                    midi_member=f"{track}/stems_midi/{stem}.mid",
                    metadata_member=f"{track}/metadata.yaml",
                    mix_midi_member=f"{track}/mix.mid",
                    f0_path=str(f0_src.relative_to(self.COCO_ROOT)),
                )
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"tar listing failed for {shard}")

    # ------------------------------------------------------------- selection/run
    def select_records(
        self,
        split: str = "test",
        per_stratum: int | None = None,
        seed: int = 0,
        max_tracks: int | None = None,
        ensembles: Iterable[str] | None = None,
        instruments: Iterable[str] | None = None,
        shards: Iterable[str] | None = None,
        rebuild_manifest: bool = False,
    ) -> list[CocoStemRecord]:
        records = self.load_or_build_manifest(split, rebuild=rebuild_manifest)
        if shards:
            records = [r for r in records if self._record_matches_shards(r, shards)]
        if ensembles:
            want = {x.lower() for x in ensembles}
            records = [r for r in records if r.ensemble.lower() in want]
        if instruments:
            want = {x.lower() for x in instruments}
            records = [r for r in records if r.instrument.lower() in want]
        if per_stratum:
            records = self.sample_records(records, per_stratum=per_stratum, seed=seed)
        if max_tracks is not None:
            records = records[:max_tracks]
        return records

    @staticmethod
    def _record_matches_shards(record: CocoStemRecord, shards: Iterable[str]) -> bool:
        record_shard = record.shard
        record_name = Path(record_shard).name
        for shard in shards:
            want = shard.strip()
            if not want:
                continue
            if want == record_shard or want == record_name or record_shard.endswith(want):
                return True
        return False

    @staticmethod
    def sample_records(
        records: Sequence[CocoStemRecord],
        per_stratum: int,
        seed: int = 0,
    ) -> list[CocoStemRecord]:
        rng = random.Random(seed)
        groups: dict[tuple[str, str], list[CocoStemRecord]] = defaultdict(list)
        for record in records:
            groups[(record.ensemble, record.instrument)].append(record)

        out: list[CocoStemRecord] = []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda r: r.track_id)
            rng.shuffle(group)
            out.extend(group[:per_stratum])
        return sorted(out, key=lambda r: r.track_id)

    def records_to_tracks(self, records: Sequence[CocoStemRecord]) -> list[tuple[str, Path, Path]]:
        tracks: list[tuple[str, Path, Path]] = []
        for record in records:
            wav = self.local_wav_path(record)
            if wav is None:
                continue
            tracks.append((record.track_id, wav, self.f0_path_for_record(record)))
        return tracks

    def iter_tracks(self, dataset: str = "test") -> Iterator[tuple[str, Path, Path]]:
        """Yield materialized/extracted stems for the overnight runner."""
        yield from self.records_to_tracks(self.load_or_build_manifest(dataset))

    def materialize_records(
        self,
        records: Sequence[CocoStemRecord],
        include_mix_midi: bool = False,
        force: bool = False,
    ) -> list[Path]:
        """Extract only selected stem files from retained shards."""
        by_shard: dict[str, list[CocoStemRecord]] = defaultdict(list)
        for record in records:
            if record.shard:
                by_shard[record.shard].append(record)

        written: list[Path] = []
        for shard_rel, shard_records in sorted(by_shard.items()):
            needed: dict[str, Path] = {}
            for record in shard_records:
                member_names = [record.wav_member, record.midi_member, record.metadata_member]
                if include_mix_midi:
                    member_names.append(record.mix_midi_member)
                for member_name in member_names:
                    if not member_name:
                        continue
                    dest = self.materialized_dir / record.split / self._norm_member(member_name)
                    if force or not dest.exists():
                        needed[self._norm_member(member_name)] = dest
            if not needed:
                continue

            shard = self.COCO_ROOT / shard_rel
            print(f"materializing {len(needed)} file(s) from {shard_rel}", flush=True)
            with tarfile.open(shard, "r:bz2") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    name = self._norm_member(member.name)
                    dest = needed.pop(name, None)
                    if dest is None:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    written.append(dest)
                    if not needed:
                        break
        return written

    # --------------------------------------------------------------- f0 loading
    @staticmethod
    def _squeeze_f0(val: Any) -> npt.NDArray[np.float64]:
        arr = np.asarray(val, dtype=float)
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)[:, 0]
        return arr.reshape(-1)

    def _f0_from_obj(self, obj: Any, voice_idx: int) -> npt.NDArray[np.float64]:
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
            obj = obj.item()
        if isinstance(obj, dict):
            candidates = [self._f0_voice_for_stem(voice_idx), voice_idx]
            for key in candidates:
                for actual in (key, str(key)):
                    if actual in obj:
                        val = obj[actual]
                        if isinstance(val, dict):
                            val = val.get("f0_hz", val.get("f0", next(iter(val.values()))))
                        return self._squeeze_f0(val)
            for key in ("f0_hz", "f0"):
                if key in obj:
                    return self._squeeze_f0(obj[key])
            return self._squeeze_f0(next(iter(obj.values())))
        if isinstance(obj, np.ndarray) and obj.dtype != object:
            f0_voice = self._f0_voice_for_stem(voice_idx)
            if obj.ndim == 2 and obj.shape[1] == 4 and obj.shape[0] > 4:
                return obj[:, f0_voice].astype(float).reshape(-1)
        return self._squeeze_f0(obj)

    def load_f0(
        self,
        f0_src: PathLike,
        voice_idx: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        p = Path(f0_src)
        suffix = p.suffix.lower()
        if suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            key = next(
                (k for k in (str(self._f0_voice_for_stem(voice_idx)), str(voice_idx), "f0_hz", "f0") if k in z.files),
                z.files[0],
            )
            freqs = self._f0_from_obj(z[key], voice_idx)
        elif suffix == ".csv":
            data = np.loadtxt(p, delimiter=",", ndmin=2)
            if data.shape[1] >= 2:
                t, f = data[:, 0].astype(float), data[:, 1].astype(float)
                return t, np.where(np.isfinite(f) & (f > 0), f, 0.0)
            freqs = self._squeeze_f0(data)
        else:
            if suffix == ".npy":
                obj = np.load(p, allow_pickle=True)
            else:
                with p.open("rb") as fh:
                    obj = pickle.load(fh)
            freqs = self._f0_from_obj(obj, voice_idx)

        freqs = np.where(np.isfinite(freqs) & (freqs > 0), freqs, 0.0)
        times = np.arange(freqs.size, dtype=float) / self.F0_FPS
        return times, freqs

    @staticmethod
    def robust_range_from_freqs(
        freqs: npt.ArrayLike,
        low_percentile: float = 1.0,
        high_percentile: float = 99.5,
        pad_semitones: float = 2.0,
        floor_hz: float = 30.0,
        ceiling_hz: float = 3000.0,
    ) -> tuple[float, float]:
        voiced = np.asarray(freqs, dtype=float)
        voiced = voiced[np.isfinite(voiced) & (voiced > floor_hz)]
        if voiced.size == 0:
            return 196.0, 3000.0
        lo, hi = np.percentile(voiced, [low_percentile, high_percentile])
        pad = 2 ** (pad_semitones / 12.0)
        return float(max(floor_hz, lo / pad)), float(min(ceiling_hz, hi * pad))

    # ----------------------------------------------------------------- audio I/O
    def load_resampled_audio(self, wav_path: PathLike, target_sr: int) -> AudioData:
        import librosa

        y, _ = librosa.load(str(wav_path), sr=int(target_sr), mono=True)
        y = np.ascontiguousarray(y, dtype=np.float32)
        ad = AudioData(config=self.DEFAULT_CONFIG)
        ad.data = y
        ad.sr = int(target_sr)
        ad.capacity = y.size
        ad.end_index = y.size
        ad.t_origin = 0.0
        return ad

    # -------------------------------------------------------------- the benchmark
    def bench_pitch_track(
        self,
        wav_path: PathLike,
        annot_path: PathLike,
        tighten_pitch_range: bool = True,
        smooth: bool = True,
        fmin: float = 196.0,
        fmax: float = 3000.0,
    ) -> PitchMetricRow:
        stem_voice = self._voice_idx(wav_path)
        ref_times, ref_freqs = self.load_f0(annot_path, stem_voice)
        if tighten_pitch_range:
            fmin, fmax = self.robust_range_from_freqs(ref_freqs)

        cfg = self.config_for(fmin, fmax)
        rec = self.recording_for(cfg)
        rec.audio_data = self.load_resampled_audio(wav_path, cfg.sr)

        timing = self.load_or_detect_pitches(
            rec,
            cache_path=self.cache_path_for_wav(wav_path),
            smooth=smooth,
            write_cache=smooth,
        )
        est_times, est_freqs = self.pitchdata_to_melody(rec.pitch_data, cfg)
        metrics = {
            k: float(v)
            for k, v in mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs).items()
        }
        return {
            **metrics,
            **timing,
            "fmin": float(fmin),
            "fmax": float(fmax),
            **self.meta_for_wav(wav_path),
        }

    def bench_records(
        self,
        records: Sequence[CocoStemRecord],
        smooth: bool = True,
        write: bool = False,
        verbose: bool = True,
    ):
        import pandas as pd

        tracks = self.records_to_tracks(records)
        rows: list[dict[str, Any]] = []
        for i, (track_id, wav, f0_src) in enumerate(tracks):
            row = self.bench_pitch_track(wav, f0_src, smooth=smooth)
            row["track_id"] = track_id
            rows.append(row)
            if verbose:
                print(
                    f"[pitch] {i + 1:>4}/{len(tracks)} "
                    f"{row['ensemble'][:6]:6s}/{row['instrument'][:12]:12s} "
                    f"RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f}",
                    flush=True,
                )
        if not rows:
            print("no materialized stems found; rerun with --materialize or extract a subset first.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("track_id")
        if write:
            for instrument, sub in df.groupby("instrument"):
                out = self.write_pitch_result(sub, "pyin_smoothed" if smooth else "pyin", f"coco_{instrument}")
                print(f"\nwrote {len(sub)} rows -> {out}")
        self.summarize(df, cols=self.PITCH_METRICS, name=self.DATASET_LABEL)
        self.summarize_by_stratum(df)
        return df

    def bench_split(
        self,
        split: str = "test",
        per_stratum: int | None = None,
        seed: int = 0,
        max_tracks: int | None = None,
        smooth: bool = True,
        write: bool = False,
        verbose: bool = True,
        materialize: bool = False,
        ensembles: Iterable[str] | None = None,
        instruments: Iterable[str] | None = None,
        shards: Iterable[str] | None = None,
        rebuild_manifest: bool = False,
    ):
        records = self.select_records(
            split=split,
            per_stratum=per_stratum,
            seed=seed,
            max_tracks=max_tracks,
            ensembles=ensembles,
            instruments=instruments,
            shards=shards,
            rebuild_manifest=rebuild_manifest,
        )
        if materialize:
            self.materialize_records(records)
        return self.bench_records(records, smooth=smooth, write=write, verbose=verbose)

    def summarize_by_stratum(self, df) -> None:
        if df.empty or "ensemble" not in df.columns:
            return
        cols = [c for c in self.PITCH_METRICS if c in df.columns]
        grp = df.groupby(["ensemble", "instrument"])
        summary = grp[cols].mean()
        summary.insert(0, "n", grp.size())
        print("\n=== per (ensemble, instrument) ===")
        try:
            import pandas as pd

            with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.max_rows", None):
                print(summary.to_string())
        except Exception:
            print(summary)

    # -------------------------------------------------------------- diagnostics
    def list_plan(
        self,
        split: str = "test",
        per_stratum: int | None = None,
        seed: int = 0,
        max_tracks: int | None = None,
        ensembles: Iterable[str] | None = None,
        instruments: Iterable[str] | None = None,
        shards: Iterable[str] | None = None,
        rebuild_manifest: bool = False,
    ) -> None:
        all_records = self.load_or_build_manifest(split, rebuild=rebuild_manifest)
        selected = self.select_records(
            split=split,
            per_stratum=per_stratum,
            seed=seed,
            max_tracks=max_tracks,
            ensembles=ensembles,
            instruments=instruments,
            shards=shards,
        )
        materialized = sum(1 for record in selected if self.local_wav_path(record) is not None)
        cached = sum(1 for record in selected if self.cache_path_for_track(record.track_id).exists())
        counts: dict[tuple[str, str], int] = defaultdict(int)
        touched_shards = sorted({r.shard for r in selected if r.shard})
        for record in selected:
            counts[(record.ensemble, record.instrument)] += 1

        print(f"root:         {self.COCO_ROOT}")
        print(f"manifest:     {self.manifest_path(split)}")
        print(f"split:        {split}  |  f0 fps: {self.F0_FPS:g}")
        print(f"stems:        {len(all_records)} in manifest | {len(selected)} selected")
        print(f"materialized: {materialized}/{len(selected)} selected stems")
        print(f"cached:       {cached}/{len(selected)} selected stems")
        if touched_shards:
            print(f"shards:       {len(touched_shards)} touched")
            for shard in touched_shards:
                print(f"  - {shard}")
        if per_stratum:
            print(f"sample:       <= {per_stratum} per (ensemble,instrument), seed={seed}")
        print("\nstrata (ensemble, instrument -> stems):")
        for key in sorted(counts):
            print(f"  {key[0]:9s} {key[1]:16s} {counts[key]:>5}")

    def probe(self, run_detect: bool = False) -> None:
        root = self.COCO_ROOT
        print(f"root: {root}  (exists={root.is_dir()})")
        for comp in ("main_dataset", "f0", "manifest", "materialized", "pitch_data"):
            path = root / comp
            tars = len(list(path.glob("**/*.tar.bz2"))) if path.is_dir() else 0
            files = len(list(path.glob("**/*"))) if path.is_dir() else 0
            print(f"  {comp:16s} present={path.is_dir()!s:5s} tar.bz2={tars:<4} entries={files}")

        f0_files = sorted((root / "f0").glob("**/*.pickle")) if (root / "f0").is_dir() else []
        if f0_files:
            fp = f0_files[0]
            print(f"\nf0 sample: {fp.relative_to(root)}")
            try:
                _, f = self.load_f0(fp, voice_idx=1)
                voiced = float(np.mean(f > 0)) if f.size else 0.0
                pos = f[f > 0]
                print(
                    f"  frames={f.size}  dur~{f.size / self.F0_FPS:.2f}s @ {self.F0_FPS:g}fps  "
                    f"voiced={voiced:.1%}  Hz[min..max]={pos.min() if pos.size else 0:.1f}.."
                    f"{pos.max() if pos.size else 0:.1f}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  could not parse f0: {exc!r}")

        wavs = list((root / "materialized").glob("**/stems_audio/*.wav"))
        if wavs:
            import soundfile as sf

            info = sf.info(str(wavs[0]))
            print(f"\naudio sample: {wavs[0].relative_to(root)}  sr={info.samplerate}Hz  dur={info.duration:.2f}s")

        if run_detect and wavs:
            print("\nrunning ONE end-to-end detection...")
            track_id, wav, f0_src = next(self.iter_tracks("test"))
            row = self.bench_pitch_track(wav, f0_src)
            print(f"  {track_id}\n  RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f}")

    def prune_f0_to_manifest(self, split: str = "test", dry_run: bool = False) -> int:
        records = self.load_or_build_manifest(split)
        keep = {self.f0_path_for_record(record).resolve() for record in records}
        f0_dir = self.COCO_ROOT / "f0" / split
        if not f0_dir.is_dir():
            return 0
        stale = [p for p in f0_dir.glob("*.pickle") if p.resolve() not in keep]
        for path in stale:
            if not dry_run:
                path.unlink()
        return len(stale)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help=f"dataset root (default: benchmarks/datasets/{CocoChoralesBenchmarker.DEFAULT_ROOT_NAME})")
    parser.add_argument("--split", default="test", choices=[*_SPLITS, "all"])
    parser.add_argument("--per-stratum", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--ensemble", action="append", dest="ensembles")
    parser.add_argument("--instrument", action="append", dest="instruments")
    parser.add_argument("--shard", action="append", dest="shards")
    parser.add_argument("--f0-fps", type=float, default=F0_FPS_DEFAULT)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-detect", action="store_true")
    parser.add_argument("--prune-f0", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    pb = CocoChoralesBenchmarker(root=args.root, f0_fps=args.f0_fps)

    if args.probe or args.probe_detect:
        pb.probe(run_detect=args.probe_detect)
        return 0
    if args.rebuild_manifest:
        records = pb.load_or_build_manifest(args.split, rebuild=True)
        print(f"wrote {len(records)} rows -> {pb.manifest_path(args.split)}")
    if args.prune_f0:
        n = pb.prune_f0_to_manifest(args.split, dry_run=args.dry_run)
        action = "would delete" if args.dry_run else "deleted"
        print(f"{action} {n} f0 pickle(s) not referenced by the manifest")
    if args.list or args.dry_run:
        pb.list_plan(
            split=args.split,
            per_stratum=args.per_stratum,
            seed=args.seed,
            max_tracks=args.max_tracks,
            ensembles=args.ensembles,
            instruments=args.instruments,
            shards=args.shards,
        )
        return 0

    pb.bench_split(
        split=args.split,
        per_stratum=args.per_stratum,
        seed=args.seed,
        max_tracks=args.max_tracks,
        smooth=not args.no_smooth,
        write=args.write,
        materialize=args.materialize,
        ensembles=args.ensembles,
        instruments=args.instruments,
        shards=args.shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
