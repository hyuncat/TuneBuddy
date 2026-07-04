from __future__ import annotations

"""Competitor pitch-detection benchmark, in the same format as our own.

This orchestrates third-party monophonic pitch trackers over the SAME corpora
and the SAME ``mir_eval.melody`` scoring that ``PitchBenchmarker`` /
``CocoChoralesBenchmarker`` use for Attune's pYIN, so every model lands as one
more row in the comparison and one more ``results/pitch/raw_outputs/<model>/<dataset>.csv``.

Models (the prominent contenders from lars76/pitch-benchmark):

    swiftf0     CNN, lars76's own model    (pip install swift-f0)
    crepe       CNN, TensorFlow            (pip install crepe tensorflow)
    torchcrepe  CNN, PyTorch CREPE         (pip install torchcrepe torch)
    penn        FCNF0++ (PyTorch)          (pip install penn torch)
    spice       self-supervised, TF-Hub    (pip install tensorflow tensorflow_hub)
    praat       autocorrelation (DSP)      (pip install praat-parselmouth)
    rmvpe       deep U-Net (PyTorch)       (vendor rmvpe.py + rmvpe.pt; see RmvpeTracker)

Plus our own pYIN as two baselines, so we can see the HMM smoother's effect:

    pyin            stage-1 only  (PitchDetector, NO HMM smoothing)
    pyin_smoothed   stage-1 + stage-2 PitchSmoother (the app default)

These two route through the EXISTING Attune pipeline (``super().bench_pitch_track``
with ``smooth=False`` / ``smooth=True``), so they are byte-for-byte the same
detector the app ships -- not a re-implementation.

The contract every model shares:  audio -> (times, freqs) in mir_eval melody
form (0 Hz = unvoiced) -> ``mir_eval.melody.evaluate(ref, est)`` -> the five
``PitchBenchmarker.PITCH_METRICS``.  Only the audio->melody step differs per
model; the dataset (ref loading, range, audio, cache) is inherited from the two
existing benchmarkers.

Usage (sarah runs these -- they are slow; this module only sets them up):

    # one or all models over bach10 (default: violin + the other 3 stems):
    python benchmarks/CompetitorPitchBenchmarker.py --dataset bach10-mf0-synth --write
    python benchmarks/CompetitorPitchBenchmarker.py --dataset bach10-mf0-synth \
        --models pyin pyin_smoothed swiftf0 praat --instrument violin --write

    # CocoChorales test split (string ensemble = violin-relevant), 3 stems/stratum:
    python benchmarks/CompetitorPitchBenchmarker.py --dataset coco --split test \
        --per-stratum 3 --ensemble string --write

    # just print the plan / which model deps are installed:
    python benchmarks/CompetitorPitchBenchmarker.py --dataset coco --list

Each model self-caches its estimated melody at
``<corpus>/competitor_pitch/<model>__<track>.npz`` (separate from the pYIN pickle
cache), so a crashed run resumes from cache.  Reported ``Compute Time`` /
``Audio(s)/Compute(s)`` are the model's wall-clock inference time measured on
the FIRST (uncached) pass; pass ``--no-cache`` to re-time every track.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, ClassVar

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

from benchmarks.modules.CocoChoralesBenchmarker import CocoChoralesBenchmarker  # noqa: E402
from benchmarks.modules.pitch.PitchBenchmarker import (  # noqa: E402
    PITCH_REALTIME_COL,
    PathLike,
    PitchBenchmarker,
    PitchMetricRow,
)

Melody = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]

# pYIN baselines are not trackers -- they reuse the app pipeline. name -> smooth.
PYIN_MODELS: dict[str, bool] = {"pyin": False, "pyin_smoothed": True}

INSTALL_HINTS: dict[str, str] = {
    "swiftf0": "pip install swift-f0",
    "crepe": "pip install crepe tensorflow   (tensorflow-macos on Apple Silicon)",
    "torchcrepe": "pip install torchcrepe torch",
    "penn": "pip install penn torch",
    "spice": "pip install tensorflow tensorflow_hub   (tensorflow-macos on Apple Silicon)",
    "praat": "pip install praat-parselmouth",
    "rmvpe": (
        "vendor an RMVPE module exposing class RMVPE + the rmvpe.pt checkpoint, then pass "
        "--rmvpe-checkpoint PATH --rmvpe-module DOTTED.PATH (or set RMVPE_CHECKPOINT / "
        "RMVPE_MODULE). e.g. RVC's infer/lib/rmvpe.py"
    ),
}


class MissingDependency(RuntimeError):
    """A model's libraries/checkpoint aren't available -- skip the model, don't crash."""

    def __init__(self, model: str, detail: str = "") -> None:
        hint = INSTALL_HINTS.get(model, "")
        msg = f"{model}: dependency unavailable"
        if detail:
            msg += f" ({detail})"
        if hint:
            msg += f"\n    -> {hint}"
        super().__init__(msg)
        self.model = model


def _quiet_tensorflow() -> None:
    """Silence TensorFlow's logging flood (SPICE / crepe SavedModel-restore spam,
    e.g. 'Unable to create a python object for variable ... reference variable').

    That spam comes through Python ``logging`` + ``absl`` on the ``tensorflow``
    logger, NOT the ``warnings`` module -- so the ``warnings.filterwarnings`` set in
    ``benchmarks/__init__.py`` can't reach it. ``TF_CPP_MIN_LOG_LEVEL`` (set there,
    before TF import) handles the C++ side; this raises the Python log level once TF
    is importable. Idempotent + best-effort, so a missing/odd TF never blocks a model.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import logging

        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)
    except Exception:  # noqa: BLE001 -- log setup must never crash a benchmark run
        pass
    try:
        from absl import logging as _absl_logging

        _absl_logging.set_verbosity(_absl_logging.ERROR)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
#  Pitch trackers: each wraps one library behind predict(audio, sr) -> melody. #
# --------------------------------------------------------------------------- #
class PitchTracker:
    """One competitor model.  ``predict`` returns (times, freqs) with 0 = unvoiced.

    ``input_sr`` is the sample rate the audio is fed in at (``None`` = keep the
    file's native rate).  Models with a fixed internal pitch range ignore the
    per-track ``fmin/fmax`` and rely on their own voicing; the few that accept a
    range (torchcrepe, penn, praat) get the tightened range, clamped to what the
    model can represent.
    """

    name: ClassVar[str] = "base"
    input_sr: ClassVar[int | None] = 16000
    model_fmin: ClassVar[float | None] = None  # representable range; None = unbounded
    model_fmax: ClassVar[float | None] = None

    def __init__(self, confidence: float | None = None, step_seconds: float = 0.01) -> None:
        self.confidence = confidence
        self.step_seconds = float(step_seconds)
        self._model: Any = None

    def ensure_available(self) -> None:
        """Import the deps (cheap) so a missing model is skipped before the loop."""
        raise NotImplementedError

    def predict(self, audio: npt.NDArray[np.float32], sr: int, fmin: float, fmax: float) -> Melody:
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------- #
    def _clamp_range(self, fmin: float, fmax: float) -> tuple[float, float]:
        lo = self.model_fmin if self.model_fmin is not None else fmin
        hi = self.model_fmax if self.model_fmax is not None else fmax
        fmin = float(min(max(fmin, lo), hi))
        fmax = float(max(min(fmax, hi), lo))
        if fmax <= fmin:
            fmin, fmax = lo, hi
        return fmin, fmax

    @staticmethod
    def _voiced_freqs(
        freqs: npt.ArrayLike, voiced: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        v = np.asarray(voiced).reshape(-1).astype(bool)
        out = np.where(v & np.isfinite(f) & (f > 0), f, 0.0)
        return out


class SwiftF0Tracker(PitchTracker):
    name = "swiftf0"
    input_sr = 16000
    # SwiftF0's fixed model range, G1..C7 -- the violin's top notes exceed fmax.
    model_fmin = 46.875
    model_fmax = 2093.75

    def ensure_available(self) -> None:
        try:
            import swift_f0  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc

    def _detector(self):
        if self._model is None:
            from swift_f0 import SwiftF0

            self._model = SwiftF0(
                confidence_threshold=0.9 if self.confidence is None else self.confidence,
                fmin=self.model_fmin,
                fmax=self.model_fmax,
            )
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        result = self._detector().detect_from_array(audio, sr)
        freqs = self._voiced_freqs(result.pitch_hz, result.voicing)
        return np.asarray(result.timestamps, dtype=float), freqs


class CrepeTracker(PitchTracker):
    name = "crepe"
    input_sr = 16000  # crepe resamples to 16k internally; feeding 16k avoids that

    def __init__(self, model_capacity: str = "full", viterbi: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.model_capacity = model_capacity
        self.viterbi = viterbi

    def ensure_available(self) -> None:
        try:
            _quiet_tensorflow()  # crepe imports TensorFlow; quiet it before that happens
            import crepe  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import crepe

        thr = 0.5 if self.confidence is None else self.confidence
        times, freq, conf, _ = crepe.predict(
            audio,
            sr,
            model_capacity=self.model_capacity,
            viterbi=self.viterbi,
            step_size=int(round(self.step_seconds * 1000)),
            verbose=0,
        )
        return np.asarray(times, float), self._voiced_freqs(freq, np.asarray(conf) >= thr)


class TorchCrepeTracker(PitchTracker):
    name = "torchcrepe"
    input_sr = 16000
    model_fmin = 32.70
    model_fmax = 1975.5  # CREPE's 360-bin ceiling (~B6)

    def __init__(self, model_capacity: str = "full", **kw: Any) -> None:
        super().__init__(**kw)
        self.model_capacity = model_capacity

    def ensure_available(self) -> None:
        try:
            import torch  # noqa: F401
            import torchcrepe  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import torch
        import torchcrepe

        fmin, fmax = self._clamp_range(fmin, fmax)
        thr = 0.21 if self.confidence is None else self.confidence
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hop = int(round(self.step_seconds * sr))
        audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        pitch, periodicity = torchcrepe.predict(
            audio_t,
            sr,
            hop_length=hop,
            fmin=fmin,
            fmax=fmax,
            model=self.model_capacity,
            return_periodicity=True,
            batch_size=512,
            device=device,
            pad=True,
        )
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()
        times = np.arange(pitch.size, dtype=float) * hop / sr
        return times, self._voiced_freqs(pitch, periodicity >= thr)


class PennTracker(PitchTracker):
    name = "penn"
    input_sr = 16000  # penn resamples internally to its model rate

    def ensure_available(self) -> None:
        try:
            import torch  # noqa: F401

            self._disable_mps_backend(torch)
            import penn  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc

    @staticmethod
    def _disable_mps_backend(torch: Any) -> None:
        """Stop penn's viterbi dep (torbi) from JIT-compiling its MPS Metal backend
        """
        try:
            torch.backends.mps.is_available = lambda: False
        except Exception:  # noqa: BLE001 -- never block penn on this best-effort guard
            pass

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import penn
        import torch

        lo = float(getattr(penn, "FMIN", 30.0))
        hi = float(getattr(penn, "FMAX", 1984.0))
        fmin = float(min(max(fmin, lo), hi))
        fmax = float(max(min(fmax, hi), lo))
        if fmax <= fmin:
            fmin, fmax = lo, hi
        thr = 0.065 if self.confidence is None else self.confidence
        gpu = 0 if torch.cuda.is_available() else None
        audio_t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        # batch_size caps peak memory: without it penn pushes the WHOLE track through
        # FCNF0++ in one forward pass (core.py: batch_size=None -> one giant batch),
        # which OOM-kills a memory-pressured run. Tiling is loss-free -- preprocess()
        # yields every frame incl. the tail, so the per-frame melody is identical.
        pitch, periodicity = penn.from_audio(
            audio_t,
            sample_rate=sr,
            hopsize=self.step_seconds,
            fmin=fmin,
            fmax=fmax,
            batch_size=512,
            gpu=gpu,
        )
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()
        times = np.arange(pitch.size, dtype=float) * self.step_seconds
        return times, self._voiced_freqs(pitch, periodicity >= thr)


class SpiceTracker(PitchTracker):
    name = "spice"
    input_sr = 16000  # SPICE REQUIRES 16k
    HUB_URL = "https://tfhub.dev/google/spice/2"
    HOP = 512  # CQT hop at 16k -> 32 ms/frame

    def ensure_available(self) -> None:
        try:
            import tensorflow  # noqa: F401
            import tensorflow_hub  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc
        _quiet_tensorflow()

    @staticmethod
    def _output2hz(pitch_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # constants from https://tfhub.dev/google/spice/2
        PT_OFFSET, PT_SLOPE, FMIN, BINS_PER_OCTAVE = 25.58, 63.07, 10.0, 12.0
        cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
        return FMIN * 2.0 ** (cqt_bin / BINS_PER_OCTAVE)

    def _load(self):
        if self._model is None:
            import tensorflow_hub as hub

            _quiet_tensorflow()  # SavedModel restore in hub.load is where the flood fires
            self._model = hub.load(self.HUB_URL)
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import tensorflow as tf

        model = self._load()
        out = model.signatures["serving_default"](tf.constant(audio, tf.float32))
        pitch = np.asarray(out["pitch"]).reshape(-1)
        conf = 1.0 - np.asarray(out["uncertainty"]).reshape(-1)
        thr = 0.9 if self.confidence is None else self.confidence
        freqs = self._output2hz(pitch)
        times = np.arange(pitch.size, dtype=float) * self.HOP / sr
        return times, self._voiced_freqs(freqs, conf >= thr)


class PraatTracker(PitchTracker):
    name = "praat"
    input_sr = None  # native rate; autocorrelation handles any sr

    def ensure_available(self) -> None:
        try:
            import parselmouth  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        import parselmouth

        floor = max(float(fmin), 40.0)
        ceil = max(float(fmax), floor + 1.0)
        snd = parselmouth.Sound(np.ascontiguousarray(audio, dtype=np.float64), sr)
        pitch = snd.to_pitch_ac(
            time_step=self.step_seconds,
            pitch_floor=floor,
            pitch_ceiling=ceil,
        )
        freqs = np.asarray(pitch.selected_array["frequency"], dtype=float)  # 0 = unvoiced
        times = np.asarray(pitch.xs(), dtype=float)
        return times, np.where(np.isfinite(freqs) & (freqs > 0), freqs, 0.0)


# Where this session vendored RMVPE (benchmarks/datasets/rmvpe/{rmvpe.py,rmvpe.pt}),
# so the model is auto-found without --rmvpe-checkpoint/--rmvpe-module or PYTHONPATH.
_VENDORED_RMVPE_DIR = ROOT / "benchmarks" / "datasets" / "rmvpe"

class RmvpeTracker(PitchTracker):
    """RMVPE has no canonical pip package -- it's vendored across RVC forks.

    We therefore load it pluggably: a checkpoint path (``--rmvpe-checkpoint`` or
    ``$RMVPE_CHECKPOINT``) plus a dotted module that exposes class ``RMVPE``
    (``--rmvpe-module`` / ``$RMVPE_MODULE``, else a few common paths are tried).
    The call follows the RVC-standard ``infer_from_audio(audio16k, thred=...)``
    returning per-frame Hz at a 10 ms hop with 0 for unvoiced.
    """

    name = "rmvpe"
    input_sr = 16000  # RVC's RMVPE expects 16k

    def __init__(
        self, checkpoint: str | None = None, module: str | None = None, thred: float = 0.03, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.checkpoint = checkpoint or os.environ.get("RMVPE_CHECKPOINT")
        if not self.checkpoint and (_VENDORED_RMVPE_DIR / "rmvpe.pt").exists():
            self.checkpoint = str(_VENDORED_RMVPE_DIR / "rmvpe.pt")  # auto-find the vendored checkpoint
        self.module = module or os.environ.get("RMVPE_MODULE")
        self.thred = float(thred)
        self.hop_seconds = 0.01
        self._module_name: str | None = None

    def _candidate_modules(self) -> list[str]:
        mods = [self.module] if self.module else []
        mods += ["rmvpe", "infer.lib.rmvpe", "lib.rmvpe", "rvc.rmvpe"]
        return [m for m in mods if m]

    def ensure_available(self) -> None:
        try:
            import importlib

            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise MissingDependency(self.name, repr(exc)) from exc
        # let the vendored module import as `rmvpe` without the caller setting PYTHONPATH
        if _VENDORED_RMVPE_DIR.is_dir() and str(_VENDORED_RMVPE_DIR) not in sys.path:
            sys.path.insert(0, str(_VENDORED_RMVPE_DIR))
        if not self.checkpoint or not Path(self.checkpoint).exists():
            raise MissingDependency(self.name, f"checkpoint not found: {self.checkpoint!r}")
        for mod in self._candidate_modules():
            try:
                if hasattr(importlib.import_module(mod), "RMVPE"):
                    self._module_name = mod
                    return
            except Exception:  # noqa: BLE001
                continue
        raise MissingDependency(self.name, f"no module with class RMVPE in {self._candidate_modules()}")

    def _load(self):
        if self._model is None:
            import importlib

            import torch

            if self._module_name is None:
                self.ensure_available()
            RMVPE = getattr(importlib.import_module(self._module_name), "RMVPE")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = RMVPE(self.checkpoint, is_half=False, device=device)
        return self._model

    def predict(self, audio, sr, fmin, fmax) -> Melody:
        model = self._load()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        try:
            f0 = model.infer_from_audio(audio, thred=self.thred)
        except TypeError:
            f0 = model.infer_from_audio(audio, self.thred)
        f0 = np.asarray(f0, dtype=float).reshape(-1)
        times = np.arange(f0.size, dtype=float) * self.hop_seconds
        return times, np.where(np.isfinite(f0) & (f0 > 0), f0, 0.0)


TRACKERS: dict[str, Callable[..., PitchTracker]] = {
    "swiftf0": SwiftF0Tracker,
    "crepe": CrepeTracker,
    "torchcrepe": TorchCrepeTracker,
    "penn": PennTracker,
    "spice": SpiceTracker,
    "praat": PraatTracker,
    "rmvpe": RmvpeTracker,
}

ALL_MODELS: list[str] = [
    "pyin",
    "pyin_smoothed",
    "praat",
    "swiftf0",
    "crepe",
    "torchcrepe",
    "spice",
    "penn",
    "rmvpe",
]

# pYIN first (the baseline we're trying to beat), then the practical contenders.
# torchcrepe and penn stay available via --models, but are omitted by default
# because they are heavy optional PyTorch competitors.
DEFAULT_MODELS: list[str] = [m for m in ALL_MODELS if m not in {"torchcrepe", "penn"}]

def make_tracker(model: str, **opts: Any) -> PitchTracker | None:
    """Build a tracker for ``model`` (or ``None`` for the pYIN baselines)."""
    if model in PYIN_MODELS:
        return None
    if model not in TRACKERS:
        raise KeyError(f"unknown model {model!r}; choices: {ALL_MODELS}")
    factory = TRACKERS[model]
    kwargs: dict[str, Any] = {}
    if opts.get("confidence") is not None:
        kwargs["confidence"] = opts["confidence"]
    if opts.get("step_seconds") is not None:
        kwargs["step_seconds"] = opts["step_seconds"]
    if model in ("crepe", "torchcrepe") and opts.get("crepe_capacity"):
        kwargs["model_capacity"] = opts["crepe_capacity"]
    if model == "rmvpe":
        kwargs["checkpoint"] = opts.get("rmvpe_checkpoint")
        kwargs["module"] = opts.get("rmvpe_module")
    return factory(**kwargs)


# --------------------------------------------------------------------------- #
#  benchmarker bindings: swap detector, keep each dataset's loader/scoring    #
# --------------------------------------------------------------------------- #
class _CompetitorEvalMixin:
    """Override ``bench_pitch_track`` to score a tracker (or a pYIN baseline).

    The two concrete classes below supply the dataset-specific seams
    (``_load_ref`` / ``_competitor_cache_path`` / ``_row_meta``); everything
    else -- audio load, timing, caching, mir_eval -- lives here so both datasets
    share it.
    """

    model_name: str = "pyin"
    tracker: PitchTracker | None = None
    use_cache: bool = True

    # -- dataset seams (implemented by the two subclasses) ----------------- #
    def _load_ref(
        self, wav_path: PathLike, annot_path: PathLike, tighten: bool, fmin: float, fmax: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        raise NotImplementedError

    def _competitor_cache_path(self, wav_path: PathLike) -> Path:
        raise NotImplementedError

    def _row_meta(self, wav_path: PathLike) -> dict[str, Any]:
        return {}

    # -- shared machinery -------------------------------------------------- #
    def bench_pitch_track(
        self,
        wav_path: PathLike,
        annot_path: PathLike,
        tighten_pitch_range: bool = True,
        smooth: bool = True,
        fmin: float = 196.0,
        fmax: float = 3000.0,
    ) -> PitchMetricRow:
        if self.model_name in PYIN_MODELS:
            from_cache = self.has_pitch_cache(  # type: ignore[attr-defined]
                self.cache_path_for_wav(wav_path),  # type: ignore[attr-defined]
                smooth=PYIN_MODELS[self.model_name],
            )
            # Reuse the EXISTING Attune pipeline unchanged: stage-1 only vs +HMM.
            row = super().bench_pitch_track(  # type: ignore[misc]
                wav_path,
                annot_path,
                tighten_pitch_range=tighten_pitch_range,
                smooth=PYIN_MODELS[self.model_name],
                fmin=fmin,
                fmax=fmax,
            )
            row["from_cache"] = from_cache
        else:
            ref_times, ref_freqs, fmin, fmax = self._load_ref(
                wav_path, annot_path, tighten_pitch_range, fmin, fmax
            )
            row = self._competitor_row(wav_path, ref_times, ref_freqs, fmin, fmax)
        row["model"] = self.model_name
        secs = self._audio_seconds(wav_path)
        row["audio_seconds"] = secs
        compute = float(row.get("pitch_compute_time", 0.0) or 0.0)
        row["realtime_factor"] = (secs / compute) if (compute > 0 and secs) else float("nan")
        row.update(self._row_meta(wav_path))
        return row

    def _competitor_row(
        self,
        wav_path: PathLike,
        ref_times: npt.NDArray[np.float64],
        ref_freqs: npt.NDArray[np.float64],
        fmin: float,
        fmax: float,
    ) -> PitchMetricRow:
        assert self.tracker is not None
        cache_path = self._competitor_cache_path(wav_path)
        cached = self._load_est_cache(cache_path) if self.use_cache else None
        if cached is not None:
            est_times, est_freqs, compute_time = cached
            from_cache = True
        else:
            want_sr = self.tracker.input_sr
            audio, sr = self._load_audio(wav_path, want_sr)
            t0 = time.perf_counter()
            est_times, est_freqs = self.tracker.predict(audio, sr, fmin, fmax)
            compute_time = time.perf_counter() - t0
            est_times = np.asarray(est_times, dtype=float).reshape(-1)
            est_freqs = np.asarray(est_freqs, dtype=float).reshape(-1)
            if est_times.size == 0:  # a silent / failed track still needs a frame
                est_times = np.array([0.0])
                est_freqs = np.array([0.0])
            if self.use_cache:
                self._save_est_cache(cache_path, est_times, est_freqs, compute_time)
            from_cache = False

        # est is a uniform arange*hop grid; rebuild it cleanly so float-quantized
        # timestamps (e.g. an older float32 cache) can't trip mir_eval's
        # non-uniform-timescale warning + silence-unaware interpolation fallback.
        est_times = self._uniform_grid(est_times)
        metrics = {
            k: float(v)
            for k, v in mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs).items()
        }
        return {
            **metrics,
            "pitch_compute_time": float(compute_time),
            "from_cache": bool(from_cache),
            "fmin": float(fmin),
            "fmax": float(fmax),
        }

    @staticmethod
    def _load_audio(wav_path: PathLike, want_sr: int | None) -> tuple[npt.NDArray[np.float32], int]:
        import librosa

        y, sr = librosa.load(str(wav_path), sr=want_sr, mono=True)
        return np.ascontiguousarray(y, dtype=np.float32), int(sr)

    @staticmethod
    def _audio_seconds(wav_path: PathLike) -> float:
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            return float(info.frames) / float(info.samplerate)
        except Exception:  # noqa: BLE001
            return float("nan")

    @staticmethod
    def _load_est_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path) as z:
                return (
                    z["times"].astype(float),
                    z["freqs"].astype(float),
                    float(z["compute_time"]),
                )
        except Exception:  # noqa: BLE001 -- a corrupt cache should just re-detect
            return None

    @staticmethod
    def _save_est_cache(
        cache_path: Path, times: np.ndarray, freqs: np.ndarray, compute_time: float
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            # times stay float64: a uniform arange*hop grid quantized to float32
            # gains ~1us jitter near the track end, enough to fail mir_eval's
            # uniform-timescale test on reload. freqs are fine at float32 (silence
            # is an exact 0.0 and pitch error is measured in cents).
            times=np.asarray(times, dtype=np.float64),
            freqs=np.asarray(freqs, dtype=np.float32),
            compute_time=np.asarray(compute_time, dtype=np.float64),
        )

    @staticmethod
    def _uniform_grid(times: np.ndarray) -> npt.NDArray[np.float64]:
        """Snap near-uniform frame times back onto an exact ``t0 + k*step`` grid.

        On lossy float32 conversions, this helps avoid mir_eval's "Non-uniform
        timescale" warning which triggers a fallback interpolation that corrupts silences
        """
        t = np.asarray(times, dtype=float).reshape(-1)
        if t.size < 2:
            return t
        step = (t[-1] - t[0]) / (t.size - 1)
        return t[0] + np.arange(t.size, dtype=float) * step


class CompetitorPitchBenchmarker(_CompetitorEvalMixin, PitchBenchmarker):
    """Competitor models over the audio/annot corpora (bach10-mf0-synth, mdb-*)."""
    def _load_ref(self, wav_path, annot_path, tighten, fmin, fmax):
        ref_times, ref_freqs = self.parse_annot(annot_path)
        if tighten:
            fmin, fmax = self.range_from_freqs(ref_freqs)
        return ref_times, ref_freqs, float(fmin), float(fmax)

    def _competitor_cache_path(self, wav_path: PathLike) -> Path:
        wav_path = Path(wav_path)
        corpus_dir = wav_path.parents[1]  # <dataset>/audio/<stem>.wav -> <dataset>
        safe = wav_path.stem.replace("/", "_")
        return corpus_dir / "competitor_pitch" / f"{self.model_name}__{safe}.npz"


class CompetitorCocoBenchmarker(_CompetitorEvalMixin, CocoChoralesBenchmarker):
    """Competitor models over the CocoChorales tiny per-stem corpus."""
    def _load_ref(self, wav_path, annot_path, tighten, fmin, fmax):
        stem_voice = self._voice_idx(wav_path)
        ref_times, ref_freqs = self.load_f0(annot_path, stem_voice)
        if tighten:
            fmin, fmax = self.robust_range_from_freqs(ref_freqs)
        return ref_times, ref_freqs, float(fmin), float(fmax)

    def _competitor_cache_path(self, wav_path: PathLike) -> Path:
        track_id = self.track_id_for_wav(wav_path).replace("/", "_")
        return self.COCO_ROOT / "competitor_pitch" / f"{self.model_name}__{track_id}.npz"

    def _row_meta(self, wav_path: PathLike) -> dict[str, Any]:
        return self.meta_for_wav(wav_path)


# --------------------------------------------------------------------------- #
#  Orchestration                                                              #
# --------------------------------------------------------------------------- #
def is_coco_dataset(dataset: str) -> bool:
    return dataset.lower() in ("coco", "cocochorales", "cocochorales_tiny")


def select_bach10_tracks(
    bench: CompetitorPitchBenchmarker,
    dataset: str,
    instrument: str | None,
    max_tracks: int | None,
) -> list[tuple[str, Path, Path]]:
    tracks = list(bench.iter_tracks(dataset))
    if instrument:
        want = instrument.lower()
        tracks = [t for t in tracks if want in t[0].lower()]
    return tracks[:max_tracks] if max_tracks else tracks


def select_coco_tracks(
    bench: CompetitorCocoBenchmarker,
    split: str,
    per_stratum: int | None,
    seed: int,
    ensembles: list[str] | None,
    instruments: list[str] | None,
    max_tracks: int | None,
) -> list[tuple[str, Path, Path]]:
    records = bench.select_records(
        split=split,
        per_stratum=per_stratum,
        seed=seed,
        max_tracks=max_tracks,
        ensembles=ensembles,
        instruments=instruments,
    )
    return bench.records_to_tracks(records)


def run_model(
    bench: _CompetitorEvalMixin,
    model: str,
    tracks: list[tuple[str, Path, Path]],
    dataset_label: str,
    opts: dict[str, Any],
    write: bool,
    verbose: bool = True,
):
    import pandas as pd

    bench.model_name = model
    bench.use_cache = not opts.get("no_cache", False)
    bench.tracker = make_tracker(model, **opts)

    if bench.tracker is not None:
        try:
            bench.tracker.ensure_available()
        except MissingDependency as exc:
            print(f"\n[{model}] SKIPPED -- {exc}")
            return None

    print(f"\n=== {model} on {dataset_label} ({len(tracks)} tracks) ===", flush=True)
    rows: list[dict[str, Any]] = []
    for i, (track_id, wav, annot) in enumerate(tracks):
        try:
            row = bench.bench_pitch_track(wav, annot)
        except MissingDependency as exc:  # only surfaces here for rmvpe lazy-loads
            print(f"[{model}] SKIPPED mid-run -- {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 -- isolate one bad track
            print(f"[{model}] {i + 1:>4}/{len(tracks)} {track_id[:40]:40s} ERROR: {exc!r}")
            continue
        row["track_id"] = track_id
        rows.append(row)
        if verbose:
            rtf = row.get("realtime_factor", float("nan"))
            print(
                f"[{model}] {i + 1:>4}/{len(tracks)} {track_id[:40]:40s} "
                f"RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f} "
                f"{rtf:.0f}xRT",
                flush=True,
            )

    if not rows:
        print(f"[{model}] no rows produced.")
        return None

    df = pd.DataFrame(rows).set_index("track_id")
    if write:
        if isinstance(bench, CompetitorCocoBenchmarker) and "instrument" in df.columns:
            for instrument, sub in df.groupby("instrument"):
                out = bench.write_pitch_result(sub, model, f"coco_{instrument}")
                print(f"[{model}] wrote {len(sub)} rows -> {out}")
        else:
            out = bench.write_pitch_result(df, model, dataset_label)
            print(f"[{model}] wrote {len(df)} rows -> {out}")
    display = bench.display_pitch_columns(df)
    bench.summarize(display, cols=[*PitchBenchmarker.PITCH_METRICS, PITCH_REALTIME_COL], name=model)
    return df


def write_comparison(
    bench: _CompetitorEvalMixin,
    summaries: dict[str, Any],
    dataset_label: str,
    write: bool,
) -> None:
    import pandas as pd

    cols = [*PitchBenchmarker.PITCH_METRICS, PITCH_REALTIME_COL]
    display_summaries = {
        model: bench.display_pitch_columns(df) for model, df in summaries.items()
    }
    table = pd.DataFrame(
        {
            model: df[[c for c in cols if c in df.columns]].mean(numeric_only=True)
            for model, df in display_summaries.items()
        }
    ).T
    table.index.name = "model"
    table = table.reindex([m for m in DEFAULT_MODELS if m in table.index] + [m for m in table.index if m not in DEFAULT_MODELS])

    print(f"\n{'=' * 72}\nCOMPARISON -- {dataset_label} (mean over tracks)\n{'=' * 72}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(table.to_string())

    if write:
        out = bench.pitch_summary_csv_path  # type: ignore[attr-defined]
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)
        print(f"\nwrote comparison -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="bach10-mf0-synth",
        help="audio/annot corpus name (e.g. bach10-mf0-synth, mdb-stem-synth) or 'coco'",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"models to run (default: {DEFAULT_MODELS}). choices: {ALL_MODELS}",
    )
    parser.add_argument("--instrument", default=None, help="filter stems by instrument (e.g. violin)")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="ignore + rewrite the competitor est cache")
    parser.add_argument("--confidence", type=float, default=None, help="override per-model voicing threshold")
    parser.add_argument("--step", type=float, default=0.01, dest="step_seconds", help="frame hop seconds (default 0.01)")
    parser.add_argument("--crepe-capacity", default="full", choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--rmvpe-checkpoint", default=None, help="path to rmvpe.pt")
    parser.add_argument("--rmvpe-module", default=None, help="dotted module exposing class RMVPE")
    # coco-only selection
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-stratum", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble", action="append", dest="ensembles")
    parser.add_argument("--root", default=None, help="coco dataset root override")
    parser.add_argument("--write", action="store_true", help="write per-model CSVs + comparison")
    parser.add_argument("--list", action="store_true", help="print the plan + which model deps are available; detect nothing")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        parser.error(f"unknown models: {unknown}; choices: {ALL_MODELS}")

    coco = is_coco_dataset(args.dataset)
    opts = {
        "no_cache": args.no_cache,
        "confidence": args.confidence,
        "step_seconds": args.step_seconds,
        "crepe_capacity": args.crepe_capacity,
        "rmvpe_checkpoint": args.rmvpe_checkpoint,
        "rmvpe_module": args.rmvpe_module,
    }

    if coco:
        bench: _CompetitorEvalMixin = CompetitorCocoBenchmarker(root=args.root)
        dataset_label = CocoChoralesBenchmarker.DATASET_LABEL
        instruments = [args.instrument] if args.instrument else None
        tracks = select_coco_tracks(
            bench, args.split, args.per_stratum, args.seed, args.ensembles, instruments, args.max_tracks
        )
    else:
        bench = CompetitorPitchBenchmarker()
        dataset_label = args.dataset
        tracks = select_bach10_tracks(bench, args.dataset, args.instrument, args.max_tracks)

    print(f"dataset:  {dataset_label}  ({'coco' if coco else 'audio/annot'})")
    print(f"models:   {', '.join(models)}")
    print(f"tracks:   {len(tracks)}")
    if not tracks:
        print("no tracks selected. for coco, materialize a subset first "
              "(CocoChoralesBenchmarker --materialize) or check --split/--ensemble/--instrument.")
        return 1

    if args.list:
        print("\nmodel availability:")
        for model in models:
            if model in PYIN_MODELS:
                print(f"  {model:14s} OK (built-in pYIN, smooth={PYIN_MODELS[model]})")
                continue
            tracker = make_tracker(model, **opts)
            assert tracker is not None
            try:
                tracker.ensure_available()
                print(f"  {model:14s} OK")
            except MissingDependency as exc:
                print(f"  {model:14s} MISSING -- {str(exc).splitlines()[-1].strip()}")
        print(f"\nfirst {min(5, len(tracks))} track(s):")
        for track_id, wav, _ in tracks[:5]:
            print(f"  {track_id}  <-  {wav}")
        return 0

    summaries: dict[str, Any] = {}
    for model in models:
        df = run_model(bench, model, tracks, dataset_label, opts, write=args.write)
        if df is not None:
            summaries[model] = df

    if summaries:
        write_comparison(bench, summaries, dataset_label, write=args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
