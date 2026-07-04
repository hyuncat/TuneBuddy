from __future__ import annotations

"""SNR sweep: how pYIN vs Praat degrade as additive noise is mixed in,
in BOTH whole-file and per-buffer (streaming) execution.

Motivation
----------
On clean CocoChorales, Praat won the pitch leaderboard (best RPA/OA, lowest
voicing false-alarm). In the actual app -- a real mic in a real room -- Praat
detects far MORE noise than pYIN. Two things the clean leaderboard never tested
are responsible, and this script exposes both as columns:

  * NOISE. The corpus's "unvoiced" frames are effectively digital silence, so
    the headline voicing-false-alarm number never tests the hard case (call
    HVAC + bow noise + mic hiss between notes "unvoiced"). We mix noise in at a
    range of SNRs and re-score against the CLEAN reference f0.

  * EXECUTION. The leaderboard ran every model WHOLE-FILE. The app runs the live
    path PER-BUFFER: it pops a w1 window every h1 hop and detects one pitch from
    it (PitchDetector.detect_pitch / PraatPitchDetector.detect_pitch). Praat's
    whole-file win comes from its global Viterbi context + a silence gate keyed
    to the global peak; per-buffer, both are gone (each rest-buffer normalizes to
    its own noise floor), which is what makes Praat light up on noise live.

Only Praat actually differs by execution mode, so only Praat is run twice:

  * pyin  -- its detector is frame-independent (no cross-frame state), so the
    per-buffer live path is BYTE-FOR-BYTE the whole-file result. We compute it
    once and copy it into the streaming row (recomputing would only inject a
    float32/float64 plumbing difference into a row that is, by construction,
    identical).
  * pyin_smoothed -- the HMM smoother is a global whole-track stage with no live
    form, so it is reported in whole-file mode only.
  * praat -- whole-file (PraatTracker) vs per-buffer (PraatPitchDetector), the
    two genuinely different execution paths.

Noise model
-----------
Additive noise is scaled to a target SNR relative to the signal power over the
VOICED (note) frames of the reference f0, then applied across the WHOLE clip
including rests (so the between-note regions carry noise at a fixed level vs the
notes). Default noise is white Gaussian; pass --noise-file for real recorded
room/mic noise (colored, more representative). Everything is seeded.

    SNR_dB = 10 * log10( mean(x[voiced]^2) / mean(noise^2) )   # per track

Timing
------
pyin/pyin_smoothed compute times are co-measured in one detector pass
(pyin = detector, pyin_smoothed = detector + HMM smoother), and the realtime
factor is aggregated as SUM(audio)/SUM(compute) -- a true throughput, so
pyin_smoothed is guaranteed <= pyin (fixes the old mean-of-ratios inversion).
Streaming Praat is per-buffer parselmouth and is MUCH slower -- that cost is
itself part of the finding.

Usage (sarah runs these -- slow; this script only sets them up)
---------------------------------------------------------------
    # both execution modes, default SNR ladder, string ensemble subset:
    python benchmarks/scripts/noisy_pitch_detection.py \
        --ensemble string --per-stratum 25 --snrs inf,20,15,10,5 \
        --workers 8 --write --plot

    # only the per-buffer path (reproduces the live symptom), real room noise:
    python benchmarks/scripts/noisy_pitch_detection.py --ensemble string \
        --per-stratum 25 --streaming stream --noise-file ~/room_tone.wav --write

    # materialize the stems first if they are not extracted yet:
    python benchmarks/scripts/noisy_pitch_detection.py --ensemble string \
        --per-stratum 25 --materialize --list
"""

import argparse
import math
import os
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# --- cap per-worker math-library threads BEFORE numpy is imported anywhere ---
# With spawn (the macOS default) each worker re-imports this module top-to-
# bottom before running a task, so setting these here covers the children too.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import numpy.typing as npt


def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


_ROOT = _bootstrap_repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mir_eval  # noqa: E402

from algorithms.competitors.PraatPitchDetector import PraatPitchDetector  # noqa: E402
from app_logic.user.ds.AudioData import AudioData  # noqa: E402
from app_logic.user.ds.PitchData import PitchData  # noqa: E402
from benchmarks.modules.CocoChoralesBenchmarker import CocoChoralesBenchmarker  # noqa: E402
from benchmarks.modules.pitch.CompetitorPitchBenchmarker import (  # noqa: E402
    MissingDependency,
    PraatTracker,
)
from benchmarks.modules.pitch.PitchBenchmarker import (  # noqa: E402
    PYIN_DEFAULT_MIN_VOLUME,
    PYIN_DEFAULT_MAX_VOLUME,
    PYIN_DEFAULT_UNV_THRESH,
    PYIN_PRAAT_MIRROR_UNV_THRESH,
)

Melody = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
TrackSpec = tuple[str, Path, Path]
RowKey = tuple[str, str, str, str]

METRIC_KEYS = [
    "Raw Pitch Accuracy",
    "Raw Chroma Accuracy",
    "Overall Accuracy",
    "Voicing Recall",
    "Voicing False Alarm",
]
# The metrics we surface as grids (the story lives in OA + false-alarm).
GRID_METRICS = ["Overall Accuracy", "Raw Pitch Accuracy", "Voicing False Alarm"]
PYIN_MODELS = ("pyin", "pyin_smoothed")
ALL_MODELS = ("pyin", "pyin_smoothed", "praat")
ROW_KEY_COLUMNS = ("track_id", "model", "mode", "snr_label")
RUN_PARAM_COLUMNS = (
    "run_seed",
    "noise_file",
    "praat_step",
    "pyin_unv_thresh",
    "pyin_volume_gate_ratio",
    "pyin_volume_gate_percentile",
)

WHOLE, STREAM = "whole", "stream"  # execution-mode labels


def mode_label(streaming: bool) -> str:
    return STREAM if streaming else WHOLE


# --------------------------------------------------------------------------- #
#  Noise                                                                       #
# --------------------------------------------------------------------------- #
def voiced_sample_mask(
    n_samples: int, sr: int, ref_times: np.ndarray, ref_freqs: np.ndarray
) -> np.ndarray:
    """Per-sample boolean: is the reference f0 voiced at this sample's time?

    ref f0 is on a uniform grid (times[k] = k / fps); map each audio sample to
    its nearest ref frame and read voicing (freq > 0).
    """
    if ref_times.size < 2 or n_samples == 0:
        return np.ones(n_samples, dtype=bool)
    fps = 1.0 / float(np.median(np.diff(ref_times)))
    idx = np.clip((np.arange(n_samples) / sr * fps).astype(np.int64), 0, ref_freqs.size - 1)
    return ref_freqs[idx] > 0


def load_noise_pool(noise_file: str | None, sr: int) -> np.ndarray | None:
    if not noise_file:
        return None
    import librosa

    y, _ = librosa.load(str(noise_file), sr=int(sr), mono=True)
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    if not np.any(y):
        raise ValueError(f"noise file is silent: {noise_file}")
    return y


def add_noise(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    voiced_mask: np.ndarray,
    noise_pool: np.ndarray | None,
) -> np.ndarray:
    """Add noise to the whole clip at ``snr_db`` relative to voiced-frame power."""
    if not math.isfinite(snr_db):  # clean
        return np.asarray(x, dtype=np.float32)

    x = np.asarray(x, dtype=np.float64)
    signal = x[voiced_mask] if voiced_mask.any() else x
    p_sig = float(np.mean(signal ** 2))
    if p_sig <= 0:
        return np.asarray(x, dtype=np.float32)

    if noise_pool is not None:
        reps = int(np.ceil(x.size / noise_pool.size))
        start = int(rng.integers(0, noise_pool.size))
        noise = np.roll(np.tile(noise_pool, reps), -start)[: x.size]
    else:
        noise = rng.standard_normal(x.size)
    noise = noise - np.mean(noise)
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 0:
        return np.asarray(x, dtype=np.float32)

    scale = math.sqrt(p_sig / (10.0 ** (snr_db / 10.0)) / p_noise)
    return np.asarray(x + scale * noise, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Scoring / melody helpers                                                    #
# --------------------------------------------------------------------------- #
def score_melody(
    ref_times: np.ndarray, ref_freqs: np.ndarray, est: Melody
) -> dict[str, float]:
    est_times, est_freqs = est
    est_times = np.asarray(est_times, dtype=float).reshape(-1)
    est_freqs = np.asarray(est_freqs, dtype=float).reshape(-1)
    if est_times.size == 0:  # a silent / failed track still needs one frame
        est_times = np.array([0.0])
        est_freqs = np.array([0.0])
    return {
        k: float(v)
        for k, v in mir_eval.melody.evaluate(
            ref_times, ref_freqs, est_times, est_freqs
        ).items()
    }


def inject_audio(cfg, samples: np.ndarray) -> AudioData:
    """Wrap a raw sample array as an AudioData (mirrors load_resampled_audio)."""
    ad = AudioData(config=cfg)
    ad.data = np.ascontiguousarray(samples, dtype=np.float32)
    ad.sr = int(cfg.sr)
    ad.capacity = ad.data.size
    ad.end_index = ad.data.size
    ad.t_origin = 0.0
    return ad


def _slide_frames(audio: np.ndarray, w1: int, h1: int) -> np.ndarray:
    """The app's live framing: a w1 window every h1 samples (== offline stride)."""
    audio = np.ascontiguousarray(audio, dtype=np.float64)
    if audio.size < w1:
        return np.empty((0, w1), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(audio, w1)[::h1]


def _pitches_to_melody(bench, cfg, pitches: list) -> Melody:
    pd = PitchData(config=cfg)
    pd.data = pitches
    return bench.pitchdata_to_melody(pd, cfg)


# --------------------------------------------------------------------------- #
#  Whole-file detection                                                        #
# --------------------------------------------------------------------------- #
def whole_pyin_melodies(bench, rec, cfg, noisy, want_smoothed) -> dict[str, tuple[Melody, float]]:
    """One detector pass -> {model: (melody, compute_seconds)} for the pYIN family.

    raw compute      = detector time
    smoothed compute = detector time + smoother time   (co-measured, same pass)
    """
    rec.audio_data = inject_audio(cfg, noisy)
    stages, timing = bench.detect_pitch_stages(rec, make_smoothed=want_smoothed)
    out = {"pyin": (bench.pitchdata_to_melody(stages["raw"], cfg),
                    float(timing["raw"]["pitch_compute_time"]))}
    if want_smoothed and "smoothed" in stages:
        out["pyin_smoothed"] = (bench.pitchdata_to_melody(stages["smoothed"], cfg),
                                float(timing["smoothed"]["pitch_compute_time"]))
    return out


def whole_praat_melody(tracker: PraatTracker, noisy, sr, fmin, fmax) -> tuple[Melody, float]:
    t0 = time.perf_counter()
    est = tracker.predict(np.asarray(noisy, dtype=np.float32), sr, fmin, fmax)
    return est, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
#  Per-buffer (streaming) detection -- mirrors the app's live _run loop        #
# --------------------------------------------------------------------------- #
def stream_praat_melody(bench, praat_det: PraatPitchDetector, cfg, noisy) -> tuple[Melody, float]:
    """Run PraatPitchDetector.detect_pitch per w1 buffer -- the exact live path
    that loses Praat's global context (this is where the false alarms come from).
    """
    frames = _slide_frames(noisy, cfg.w1, cfg.h1)
    t0 = time.perf_counter()
    pitches = [praat_det.detect_pitch(frames[i], (i * cfg.h1) / cfg.sr) for i in range(frames.shape[0])]
    return _pitches_to_melody(bench, cfg, pitches), time.perf_counter() - t0


# --------------------------------------------------------------------------- #
#  Sweep                                                                       #
# --------------------------------------------------------------------------- #
def parse_snrs(text: str) -> list[float]:
    out: list[float] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(math.inf if tok in ("inf", "clean", "none") else float(tok))
    return out


def snr_label(snr: float) -> str:
    return "clean" if not math.isfinite(snr) else f"{snr:g}dB"


def expected_row_keys_for_track(
    track_id: str,
    models: list[str],
    snrs: list[float],
    streaming_modes: list[bool],
) -> set[RowKey]:
    keys: set[RowKey] = set()
    mode_labels = [mode_label(streaming) for streaming in streaming_modes]
    for snr in snrs:
        label = snr_label(snr)
        for model in models:
            if model == "pyin_smoothed":
                if WHOLE in mode_labels:
                    keys.add((track_id, model, WHOLE, label))
                continue
            for mode in mode_labels:
                keys.add((track_id, model, mode, label))
    return keys


def expected_row_keys(
    tracks: list[TrackSpec],
    models: list[str],
    snrs: list[float],
    streaming_modes: list[bool],
) -> set[RowKey]:
    keys: set[RowKey] = set()
    for track_id, _, _ in tracks:
        keys.update(expected_row_keys_for_track(track_id, models, snrs, streaming_modes))
    return keys


def row_key(row: Any) -> RowKey:
    return (
        str(row["track_id"]),
        str(row["model"]),
        str(row["mode"]),
        str(row["snr_label"]),
    )


def run_sweep(
    bench: CocoChoralesBenchmarker,
    tracks: list[TrackSpec],
    models: list[str],
    snrs: list[float],
    streaming_modes: list[bool],
    seed: int,
    noise_pool: np.ndarray | None,
    praat_step: float,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    want_smoothed = "pyin_smoothed" in models
    want_pyin_family = any(m in models for m in PYIN_MODELS)
    want_praat = "praat" in models
    if want_praat:
        try:
            PraatTracker(step_seconds=praat_step).ensure_available()
        except MissingDependency as exc:
            if verbose:
                print(f"[praat] SKIPPED -- {exc}")
            want_praat = False
            models = [m for m in models if m != "praat"]
    tracker = PraatTracker(step_seconds=praat_step) if (want_praat and False in streaming_modes) else None
    if verbose and want_praat and True in streaming_modes:
        print("[note] streaming Praat runs parselmouth per w1 buffer -- expect it to be slow.")
    if verbose and want_smoothed and True in streaming_modes:
        print("[note] pyin_smoothed is an offline (global-HMM) stage -- reported in whole-file mode only.")

    rows: list[dict[str, Any]] = []
    for i, (track_id, wav, f0_src) in enumerate(tracks):
        voice_idx = bench._voice_idx(wav)
        ref_times, ref_freqs = bench.load_f0(f0_src, voice_idx)
        fmin, fmax = bench.robust_range_from_freqs(ref_freqs)
        cfg = bench.config_for(fmin, fmax)
        rec = bench.recording_for(cfg) if want_pyin_family else None
        praat_det = (
            PraatPitchDetector(config=cfg) if (want_praat and True in streaming_modes) else None
        )

        clean = bench.load_resampled_audio(wav, cfg.sr).read_all().astype(np.float64)
        audio_s = clean.size / cfg.sr
        vmask = voiced_sample_mask(clean.size, cfg.sr, ref_times, ref_freqs)
        meta = bench.meta_for_wav(wav)

        for snr in snrs:
            # one noise draw per (track, snr), shared across models AND modes for
            # fairness. crc32 (not hash()) so the draw is reproducible across runs.
            draw_seed = (zlib.crc32(f"{track_id}|{snr_label(snr)}".encode()) ^ (seed & 0xFFFFFFFF)) & 0xFFFFFFFF
            noisy = add_noise(clean, snr, np.random.default_rng(draw_seed), vmask, noise_pool)

            # pYIN is frame-independent, so per-buffer == whole-file exactly:
            # compute the family ONCE and copy it into every mode. Only Praat is
            # recomputed per mode.
            pyin_results = (
                whole_pyin_melodies(bench, rec, cfg, noisy, want_smoothed) if want_pyin_family else {}
            )

            for streaming in streaming_modes:
                results: dict[str, tuple[Melody, float]] = {}
                if "pyin" in pyin_results:
                    results["pyin"] = pyin_results["pyin"]  # identical in every mode
                if not streaming and "pyin_smoothed" in pyin_results:
                    results["pyin_smoothed"] = pyin_results["pyin_smoothed"]  # HMM is offline-only
                if want_praat:
                    if streaming and praat_det is not None:
                        results["praat"] = stream_praat_melody(bench, praat_det, cfg, noisy)
                    elif not streaming and tracker is not None:
                        results["praat"] = whole_praat_melody(tracker, noisy, cfg.sr, fmin, fmax)

                for model in models:
                    if model not in results:
                        continue
                    melody, compute_s = results[model]
                    metrics = score_melody(ref_times, ref_freqs, melody)
                    rows.append(
                        {
                            "track_id": track_id,
                            "model": model,
                            "streaming": bool(streaming),
                            "mode": mode_label(streaming),
                            "snr": float(snr) if math.isfinite(snr) else math.inf,
                            "snr_label": snr_label(snr),
                            **{k: metrics[k] for k in METRIC_KEYS},
                            "audio_s": float(audio_s),
                            "compute_s": float(compute_s),
                            "fmin": float(fmin),
                            "fmax": float(fmax),
                            **meta,
                        }
                    )

        if verbose:
            print(f"[sweep] {i + 1:>4}/{len(tracks)}  {track_id}", flush=True)

    return rows


def _make_bench(root: str | None, config_overrides: dict[str, Any]) -> CocoChoralesBenchmarker:
    bench = CocoChoralesBenchmarker(root=root)
    bench.config_overrides.update(config_overrides)
    return bench


def _chunks(items: list[TrackSpec], size: int) -> list[list[TrackSpec]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fmt_dur(seconds: float) -> str:
    seconds_i = int(seconds)
    minutes, sec = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m{sec:02d}s" if hours else f"{minutes:d}m{sec:02d}s"


def _run_sweep_chunk(payload: dict[str, Any]) -> tuple[int, int, str, list[dict[str, Any]]]:
    bench = _make_bench(payload["root"], payload["config_overrides"])
    noise_pool = load_noise_pool(payload["noise_file"], bench.DEFAULT_CONFIG.sr)
    tracks = payload["tracks"]
    rows = run_sweep(
        bench,
        tracks,
        list(payload["models"]),
        list(payload["snrs"]),
        list(payload["streaming_modes"]),
        int(payload["seed"]),
        noise_pool,
        float(payload["praat_step"]),
        verbose=False,
    )
    first_track = tracks[0][0] if tracks else "empty"
    last_track = tracks[-1][0] if tracks else "empty"
    label = first_track if len(tracks) == 1 else f"{first_track} .. {last_track}"
    return int(payload["chunk_index"]), len(tracks), label, rows


def run_sweep_parallel(
    root: str | None,
    config_overrides: dict[str, Any],
    tracks: list[TrackSpec],
    models: list[str],
    snrs: list[float],
    streaming_modes: list[bool],
    seed: int,
    noise_file: str | None,
    praat_step: float,
    workers: int,
    tracks_per_task: int,
) -> list[dict[str, Any]]:
    if workers <= 1 or len(tracks) <= 1:
        bench = _make_bench(root, config_overrides)
        noise_pool = load_noise_pool(noise_file, bench.DEFAULT_CONFIG.sr)
        return run_sweep(
            bench,
            tracks,
            list(models),
            snrs,
            streaming_modes,
            seed,
            noise_pool,
            praat_step,
        )

    if "praat" in models and True in streaming_modes:
        print("[note] streaming Praat runs parselmouth per w1 buffer -- expect it to be slow.")
    if "pyin_smoothed" in models and True in streaming_modes:
        print("[note] pyin_smoothed is an offline (global-HMM) stage -- reported in whole-file mode only.")

    workers = max(1, workers)
    if tracks_per_task <= 0:
        tracks_per_task = max(1, math.ceil(len(tracks) / workers))
    chunks = _chunks(tracks, tracks_per_task)
    workers = min(max(1, workers), len(chunks))
    payloads = [
        {
            "chunk_index": i,
            "root": root,
            "config_overrides": dict(config_overrides),
            "tracks": chunk,
            "models": list(models),
            "snrs": list(snrs),
            "streaming_modes": list(streaming_modes),
            "seed": int(seed),
            "noise_file": noise_file,
            "praat_step": float(praat_step),
        }
        for i, chunk in enumerate(chunks)
    ]

    print(
        f"[parallel] {workers} worker(s), {len(chunks)} task(s), "
        f"{tracks_per_task} track(s)/task",
        flush=True,
    )
    started = time.perf_counter()
    completed = 0
    rows_by_chunk: dict[int, list[dict[str, Any]]] = {}
    total = len(tracks)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_sweep_chunk, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                chunk_index, n_tracks, label, rows = future.result()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[error] task {payload['chunk_index']} failed:\n"
                    f"{''.join(traceback.format_exception(exc))}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

            rows_by_chunk[chunk_index] = rows
            completed += n_tracks
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (total - completed) / rate if rate else 0.0
            print(
                f"[sweep] {completed:>4}/{total}  {label} "
                f"| elapsed {_fmt_dur(elapsed)} eta {_fmt_dur(eta)}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for chunk_index in sorted(rows_by_chunk):
        rows.extend(rows_by_chunk[chunk_index])
    return rows


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #
def build_run_metadata(
    seed: int,
    noise_file: str | None,
    praat_step: float,
    pyin_unv_thresh: float,
    pyin_volume_gate_ratio: float,
    pyin_volume_gate_percentile: float,
) -> dict[str, Any]:
    return {
        "run_seed": int(seed),
        "noise_file": str(noise_file or "white_gaussian"),
        "praat_step": float(praat_step),
        "pyin_unv_thresh": float(pyin_unv_thresh),
        "pyin_volume_gate_ratio": float(pyin_volume_gate_ratio),
        "pyin_volume_gate_percentile": float(pyin_volume_gate_percentile),
    }


def add_run_metadata(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    for row in rows:
        row.update(metadata)


def _snr_label_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return ""
        if text == "clean" or text.endswith("dB"):
            return text
    try:
        return snr_label(float(value))
    except (TypeError, ValueError):
        return str(value)


def _series_matches_value(series: Any, expected: Any):
    import pandas as pd

    if isinstance(expected, (int, float)):
        expected_f = float(expected)
        values = pd.to_numeric(series, errors="coerce")
        if math.isfinite(expected_f):
            return (values - expected_f).abs() <= 1e-12
        return values.map(lambda value: bool(math.isinf(value) and value == expected_f))

    expected_s = str(expected or "")
    return series.fillna("").astype(str) == expected_s


def load_matching_existing_rows(
    path: Path,
    metadata: dict[str, Any],
    expected_keys: set[RowKey],
):
    import pandas as pd

    if not path.exists():
        print(f"[skip] no existing per-track CSV at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        print(f"[skip] existing per-track CSV is empty: {path}")
        return df

    if "snr_label" not in df.columns and "snr" in df.columns:
        df["snr_label"] = df["snr"].map(_snr_label_from_value)
    if "mode" not in df.columns and "streaming" in df.columns:
        df["mode"] = df["streaming"].map(lambda value: STREAM if bool(value) else WHOLE)

    missing_key_cols = [col for col in ROW_KEY_COLUMNS if col not in df.columns]
    if missing_key_cols:
        print(f"[skip] existing CSV is missing key columns {missing_key_cols}; not skipping")
        return pd.DataFrame()

    original_cols = set(df.columns)
    mask = pd.Series(True, index=df.index)
    legacy_cols: list[str] = []
    for col in RUN_PARAM_COLUMNS:
        if col in original_cols:
            mask &= _series_matches_value(df[col], metadata[col])
        else:
            df[col] = metadata[col]
            legacy_cols.append(col)

    if legacy_cols:
        print(
            "[skip] existing CSV lacks run-parameter columns; "
            "matching older rows by track/model/mode/SNR only"
        )

    current = df[mask].copy()
    if current.empty:
        print(f"[skip] no existing rows match this run's parameters in {path}")
        return current

    current["_row_key"] = current.apply(row_key, axis=1)
    current = current[current["_row_key"].isin(expected_keys)].copy()
    current = current.drop(columns=["_row_key"])
    return current


def split_tracks_by_existing(
    tracks: list[TrackSpec],
    existing_df: Any,
    models: list[str],
    snrs: list[float],
    streaming_modes: list[bool],
) -> tuple[list[TrackSpec], list[TrackSpec]]:
    if existing_df is None or existing_df.empty:
        return tracks, []

    existing_keys = set(existing_df.apply(row_key, axis=1))
    to_run: list[TrackSpec] = []
    skipped: list[TrackSpec] = []
    for track in tracks:
        track_id = track[0]
        expected = expected_row_keys_for_track(track_id, models, snrs, streaming_modes)
        if expected and expected.issubset(existing_keys):
            skipped.append(track)
        else:
            to_run.append(track)
    return to_run, skipped


def combine_existing_and_new_rows(
    existing_df: Any,
    new_rows: list[dict[str, Any]],
    tracks: list[TrackSpec],
    models: list[str],
    snrs: list[float],
    streaming_modes: list[bool],
):
    import pandas as pd

    frames = []
    if existing_df is not None and not existing_df.empty:
        frames.append(existing_df.copy())
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    expected_keys = expected_row_keys(tracks, models, snrs, streaming_modes)
    df["_row_key"] = df.apply(row_key, axis=1)
    df = df[df["_row_key"].isin(expected_keys)].copy()
    df = df.drop_duplicates(subset=list(ROW_KEY_COLUMNS), keep="last")

    track_order = {track_id: i for i, (track_id, _, _) in enumerate(tracks)}
    model_order = {model: i for i, model in enumerate(models)}
    mode_order = {WHOLE: 0, STREAM: 1}
    snr_order = {snr_label(snr): i for i, snr in enumerate(snrs)}
    df["_track_order"] = df["track_id"].map(track_order).fillna(len(track_order))
    df["_model_order"] = df["model"].map(model_order).fillna(len(model_order))
    df["_mode_order"] = df["mode"].map(mode_order).fillna(len(mode_order))
    df["_snr_order"] = df["snr_label"].map(snr_order).fillna(len(snr_order))
    df = df.sort_values(["_track_order", "_model_order", "_mode_order", "_snr_order"])
    return df.drop(
        columns=[
            "_row_key",
            "_track_order",
            "_model_order",
            "_mode_order",
            "_snr_order",
        ]
    )


def summarize(rows: list[dict[str, Any]] | Any):
    import pandas as pd

    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for (model, mode, label), g in df.groupby(["model", "mode", "snr_label"], sort=False):
        rec = {"model": model, "mode": mode, "snr": label, "n": int(len(g))}
        for k in METRIC_KEYS:
            rec[k] = float(g[k].mean())
        total_compute = float(g["compute_s"].sum())
        rec["Audio(s)/Compute(s)"] = (
            float(g["audio_s"].sum()) / total_compute if total_compute > 0 else float("nan")
        )
        summary.append(rec)
    return df, pd.DataFrame(summary)


def _row_index(summary_df, models: list[str]) -> list[tuple[str, str]]:
    have = set(map(tuple, summary_df[["model", "mode"]].drop_duplicates().values))
    order = [(m, mode) for m in models for mode in (WHOLE, STREAM)]
    return [key for key in order if key in have]


def print_report(summary_df, snrs: list[float], models: list[str]) -> None:
    import pandas as pd

    col_order = [snr_label(s) for s in snrs]
    idx_order = _row_index(summary_df, models)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 220):
        for metric in [*GRID_METRICS, "Voicing Recall", "Audio(s)/Compute(s)"]:
            grid = (
                summary_df.pivot(index=["model", "mode"], columns="snr", values=metric)
                .reindex(index=idx_order, columns=col_order)
            )
            if metric == "Voicing False Alarm":
                note = "(lower is better)"
            elif metric == "Audio(s)/Compute(s)":
                note = "(Sum audio / Sum compute; higher = faster)"
            else:
                note = "(higher is better)"
            print(f"\n=== {metric} vs SNR  {note} ===")
            print(grid.to_string())


def maybe_plot(summary_df, snrs: list[float], models: list[str], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] skipped ({exc!r})")
        return

    finite = [s for s in snrs if math.isfinite(s)]
    xmax = (max(finite) + 10) if finite else 40.0  # place "clean" just past the best finite SNR

    def xval(label: str) -> float:
        return xmax if label == "clean" else float(label.replace("dB", ""))

    colors = {"pyin": "#2a78d6", "pyin_smoothed": "#eda100", "praat": "#1baf7a"}
    styles = {WHOLE: "-", STREAM: "--"}
    fig, axes = plt.subplots(1, len(GRID_METRICS), figsize=(5 * len(GRID_METRICS), 4), squeeze=False)
    for ax, metric in zip(axes[0], GRID_METRICS):
        for model, mode in _row_index(summary_df, models):
            sub = summary_df[(summary_df["model"] == model) & (summary_df["mode"] == mode)]
            pts = sorted(((xval(r["snr"]), r[metric]) for _, r in sub.iterrows()), key=lambda p: p[0])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=colors.get(model), linestyle=styles[mode],
                    label=f"{model} · {mode}")
        ax.set_title(metric)
        ax.set_xlabel("SNR (dB)  |  rightmost = clean")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] wrote {out_path}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def resolve_streaming_modes(choice: str) -> list[bool]:
    return {"both": [False, True], "whole": [False], "stream": [True]}[choice]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=None, help="CocoChorales dataset root override")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--ensemble", action="append", dest="ensembles",
                        help="filter by ensemble (repeatable), e.g. --ensemble string")
    parser.add_argument("--instrument", action="append", dest="instruments",
                        help="filter by instrument (repeatable)")
    parser.add_argument("--per-stratum", type=int, default=None,
                        help="<= N stems per (ensemble,instrument) for a spread subset")
    parser.add_argument("--max-tracks", type=int, default=100,
                        help="hard cap on total stems (default 100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snrs", default="inf,20,15,10,5",
                        help="comma list of target SNRs in dB; 'inf'/'clean' = no noise")
    parser.add_argument("--streaming", default="both", choices=["both", "whole", "stream"],
                        help="execution mode(s): whole-file, per-buffer, or both (default)")
    parser.add_argument("--noise-file", default=None,
                        help="wav of real noise to add (default: white Gaussian)")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                        help=f"subset of {ALL_MODELS}")
    parser.add_argument(
        "--pyin-unv-thresh",
        type=float,
        default=None,
        help=(
            "override Attune pYIN unvoiced-probability threshold; lower is "
            f"stricter. Default is {PYIN_DEFAULT_UNV_THRESH:.2f}; Praat's "
            f"voicing_threshold=0.45 corresponds to {PYIN_PRAAT_MIRROR_UNV_THRESH:.2f} here."
        ),
    )
    parser.add_argument(
        "--mirror-praat-voicing",
        action="store_true",
        help=(
            "set --pyin-unv-thresh to 0.55, the pYIN analogue to Praat's "
            "voicing_threshold=0.45; stricter than the default 0.90"
        ),
    )
    parser.add_argument(
        "--pyin-volume-gate-ratio",
        type=float,
        default=None,
        help=(
            "relative RMS gate for pYIN before per-frame normalization; "
            "frames below ratio * percentile(frame-RMS) are forced unvoiced. "
            f"Default is {PYIN_DEFAULT_MIN_VOLUME:.2f}; "
            "use 0 to disable."
        ),
    )
    parser.add_argument(
        "--pyin-volume-gate-percentile",
        type=float,
        default=None,
        help=(
            "frame-RMS percentile used as the pYIN volume-gate reference "
            f"(default {PYIN_DEFAULT_MAX_VOLUME * 100:.1f}; use 100 for the old max-frame gate)"
        ),
    )
    parser.add_argument("--praat-step", type=float, default=0.01, help="Praat frame hop (s)")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 4),
        help=(
            "process pool size for track-level parallelism "
            "(default: all logical CPUs; use 1 for serial)"
        ),
    )
    parser.add_argument(
        "--tracks-per-task",
        type=int,
        default=0,
        help="number of stems per worker task (0 = auto-size by worker count)",
    )
    parser.add_argument("--materialize", action="store_true",
                        help="extract selected stems from shards before running")
    parser.add_argument("--out-dir", default=None,
                        help="where to write CSVs/plot (default: benchmarks/results/pitch/noisy)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "read noisy_per_track.csv from --out-dir and skip stems whose "
            "requested model/mode/SNR rows already exist for the current run parameters"
        ),
    )
    parser.add_argument("--write", action="store_true", help="write per-track + summary CSVs")
    parser.add_argument("--plot", action="store_true", help="save a metric-vs-SNR PNG")
    parser.add_argument("--list", action="store_true", help="print the selection plan and exit")
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        parser.error(f"unknown models {unknown}; choices: {list(ALL_MODELS)}")
    if args.max_tracks and args.max_tracks > 100:
        print(f"[warn] --max-tracks {args.max_tracks} > 100; this is meant to stay small.")

    models = list(args.models)
    if "praat" in models:
        try:
            PraatTracker(step_seconds=args.praat_step).ensure_available()
        except MissingDependency as exc:
            print(f"[praat] SKIPPED -- {exc}")
            models = [m for m in models if m != "praat"]

    streaming_modes = resolve_streaming_modes(args.streaming)
    bench = CocoChoralesBenchmarker(root=args.root)
    pyin_unv_thresh = (
        PYIN_PRAAT_MIRROR_UNV_THRESH
        if args.mirror_praat_voicing and args.pyin_unv_thresh is None
        else args.pyin_unv_thresh
    )
    if pyin_unv_thresh is not None:
        bench.config_overrides["unv_thresh"] = float(pyin_unv_thresh)
    pyin_volume_gate_ratio = args.pyin_volume_gate_ratio
    if pyin_volume_gate_ratio is not None:
        bench.config_overrides["min_volume"] = float(pyin_volume_gate_ratio)
    pyin_volume_gate_percentile = args.pyin_volume_gate_percentile
    if pyin_volume_gate_percentile is not None:
        # CLI percentile is 0-100; Config.max_volume stores it as a fraction.
        bench.config_overrides["max_volume"] = float(pyin_volume_gate_percentile) / 100.0
    records = bench.select_records(
        split=args.split,
        per_stratum=args.per_stratum,
        seed=args.seed,
        max_tracks=args.max_tracks,
        ensembles=args.ensembles,
        instruments=args.instruments,
    )
    if args.materialize:
        bench.materialize_records(records)
    tracks = bench.records_to_tracks(records)
    snrs = parse_snrs(args.snrs)

    print(f"root:      {bench.COCO_ROOT}")
    print(f"models:    {', '.join(models) if models else '(none)'}")
    if any(model in PYIN_MODELS for model in models):
        effective_pyin_unv = (
            pyin_unv_thresh
            if pyin_unv_thresh is not None
            else PYIN_DEFAULT_UNV_THRESH
        )
        source = "override" if pyin_unv_thresh is not None else "default"
        print(f"pYIN unv:  {effective_pyin_unv:.3f} ({source})")
        effective_gate = (
            pyin_volume_gate_ratio
            if pyin_volume_gate_ratio is not None
            else PYIN_DEFAULT_MIN_VOLUME
        )
        gate_source = "override" if pyin_volume_gate_ratio is not None else "default"
        effective_gate_percentile = (
            pyin_volume_gate_percentile
            if pyin_volume_gate_percentile is not None
            else PYIN_DEFAULT_MAX_VOLUME * 100
        )
        percentile_source = (
            "override" if pyin_volume_gate_percentile is not None else "default"
        )
        print(
            f"pYIN gate: {effective_gate:.3f} ({gate_source}) "
            f"* p{effective_gate_percentile:g} RMS ({percentile_source})"
        )
    else:
        effective_pyin_unv = PYIN_DEFAULT_UNV_THRESH
        effective_gate = PYIN_DEFAULT_MIN_VOLUME
        effective_gate_percentile = PYIN_DEFAULT_MAX_VOLUME * 100
    print(f"modes:     {', '.join(mode_label(s) for s in streaming_modes)}")
    print(f"snrs:      {', '.join(snr_label(s) for s in snrs)}")
    print(f"noise:     {'file:' + args.noise_file if args.noise_file else 'white gaussian'}")
    print(f"stems:     {len(tracks)} selected"
          + (f" (of {len(records)} records)" if len(records) != len(tracks) else ""))
    print(f"workers:   {args.workers}  (cpu_count={os.cpu_count()}, BLAS threads capped to 1/worker)")

    if args.list:
        for track_id, wav, _ in tracks[: min(10, len(tracks))]:
            print(f"  {track_id}  <-  {wav}")
        return 0
    if not tracks:
        print("no materialized stems found. rerun with --materialize, or extract a subset "
              "(CocoChoralesBenchmarker --materialize) first.")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else bench.RESULTS / "pitch" / "noisy"
    per_track_path = out_dir / "noisy_per_track.csv"
    run_metadata = build_run_metadata(
        args.seed,
        args.noise_file,
        args.praat_step,
        effective_pyin_unv,
        effective_gate,
        effective_gate_percentile,
    )

    existing_df = None
    tracks_to_run = tracks
    skipped_tracks: list[TrackSpec] = []
    if args.skip_existing:
        existing_df = load_matching_existing_rows(
            per_track_path,
            run_metadata,
            expected_row_keys(tracks, models, snrs, streaming_modes),
        )
        tracks_to_run, skipped_tracks = split_tracks_by_existing(
            tracks,
            existing_df,
            models,
            snrs,
            streaming_modes,
        )
        print(
            f"[skip] {len(skipped_tracks)} complete stem(s) already present; "
            f"{len(tracks_to_run)} stem(s) to run",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    if tracks_to_run:
        rows = run_sweep_parallel(
            args.root,
            dict(bench.config_overrides),
            tracks_to_run,
            models,
            snrs,
            streaming_modes,
            args.seed,
            args.noise_file,
            args.praat_step,
            max(1, int(args.workers)),
            int(args.tracks_per_task),
        )
        add_run_metadata(rows, run_metadata)

    combined_df = combine_existing_and_new_rows(
        existing_df,
        rows,
        tracks,
        models,
        snrs,
        streaming_modes,
    )
    if combined_df.empty:
        print("no rows produced.")
        return 1

    df, summary_df = summarize(combined_df)
    print_report(summary_df, snrs, models)

    if args.write or args.plot:
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.write:
        df.to_csv(out_dir / "noisy_per_track.csv", index=False)
        summary_df.to_csv(out_dir / "noisy_summary.csv", index=False)
        print(f"\nwrote {out_dir / 'noisy_per_track.csv'}\nwrote {out_dir / 'noisy_summary.csv'}")
    if args.plot:
        maybe_plot(summary_df, snrs, models, out_dir / "noisy_snr_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
