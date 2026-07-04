#!/usr/bin/env python3
"""Parallel, resumable pitch benchmark runner.

Runs Attune pYIN plus third-party competitors over the selected pitch corpora.
Each method writes raw per-track CSVs under
``benchmarks/results/pitch/raw_outputs/<method>/`` and the final comparison table
to ``benchmarks/results/pitch/pitch_benchmarks.csv``.

The work is partitioned into model-aware chunks sized to keep the configured
worker pool busy. Use ``--tracks-per-task`` to override that partitioning.
"""
from __future__ import annotations

# --- cap per-worker math-library threads BEFORE numpy is imported anywhere ---
# With `spawn` (the macOS default) each worker re-imports this module top-to-
# bottom before running a task, so setting these here covers the children too.
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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

from benchmarks.modules.CocoChoralesBenchmarker import F0_FPS_DEFAULT  # noqa: E402
from benchmarks.modules.pitch.CompetitorPitchBenchmarker import (  # noqa: E402
    ALL_MODELS,
    DEFAULT_MODELS,
    PYIN_MODELS,
    CompetitorCocoBenchmarker,
    CompetitorPitchBenchmarker,
    MissingDependency,
    make_tracker,
)
from benchmarks.modules.pitch.PitchBenchmarker import (  # noqa: E402
    PITCH_AUDIO_SECONDS_COL,
    PITCH_COMPUTE_COL,
    PITCH_RAW_STAGE,
    PITCH_REALTIME_COL,
    PITCH_SMOOTHED_STAGE,
    PYIN_DEFAULT_MIN_VOLUME,
    PYIN_DEFAULT_MAX_VOLUME,
    PYIN_DEFAULT_UNV_THRESH,
    PYIN_PRAAT_MIRROR_UNV_THRESH,
    PitchBenchmarker,
)
from benchmarks.modules.runner import (  # noqa: E402
    TrackItem,
    WorkChunk,
    fmt_dur,
    parse_shard,
    process_chunks,
)


def make_benchmarker(kind: str, opts: dict[str, Any] | None = None):
    opts = opts or {}
    if kind == "coco":
        bench = CompetitorCocoBenchmarker(
            root=opts.get("root"),
            f0_fps=float(opts.get("f0_fps", F0_FPS_DEFAULT)),
        )
    else:
        bench = CompetitorPitchBenchmarker()
    if opts.get("pyin_unv_thresh") is not None:
        bench.config_overrides["unv_thresh"] = float(opts["pyin_unv_thresh"])
    if opts.get("pyin_volume_gate_ratio") is not None:
        bench.config_overrides["min_volume"] = float(opts["pyin_volume_gate_ratio"])
    if opts.get("pyin_volume_gate_percentile") is not None:
        # CLI percentile is 0-100; Config.max_volume stores it as a fraction.
        bench.config_overrides["max_volume"] = float(
            opts["pyin_volume_gate_percentile"]
        ) / 100.0
    return bench


def list_work(
    datasets: list[str],
    shard: tuple[int, int] | None = None,
    kind: str = "pitch",
    opts: dict[str, Any] | None = None,
) -> list[TrackItem]:
    opts = opts or {}
    pb = make_benchmarker(kind, opts)
    work: list[TrackItem] = []
    if kind == "coco" and (
        opts.get("instruments")
        or opts.get("ensembles")
        or opts.get("max_tracks") is not None
        or opts.get("per_stratum") is not None
        or opts.get("materialize")
    ):
        for ds in datasets:
            records = pb.select_records(
                split=ds,
                per_stratum=opts.get("per_stratum"),
                seed=int(opts.get("seed", 0)),
                # Cap after records_to_tracks() below so --max-tracks means
                # runnable materialized stems, not manifest rows that may be
                # skipped because their WAVs are not local.
                max_tracks=opts.get("max_tracks") if opts.get("materialize") else None,
                ensembles=opts.get("ensembles"),
                instruments=opts.get("instruments"),
                rebuild_manifest=False,
            )
            if opts.get("materialize"):
                written = pb.materialize_records(
                    records,
                    force=bool(opts.get("force_materialize", False)),
                )
                print(f"materialized {len(written)} file(s) for {ds}", flush=True)
            for track_id, wav, annot in pb.records_to_tracks(records):
                work.append((ds, track_id, str(wav), str(annot)))
    else:
        for ds in datasets:
            for track_id, wav, annot in pb.iter_tracks(ds):
                work.append((ds, track_id, str(wav), str(annot)))
    work.sort()
    if kind != "coco" and opts.get("instruments"):
        wanted = [str(name).lower() for name in opts["instruments"]]
        work = [item for item in work if any(name in item[1].lower() for name in wanted)]
    if opts.get("max_tracks") is not None:
        work = work[: int(opts["max_tracks"])]
    if shard is not None:
        i, n = shard
        work = [w for idx, w in enumerate(work) if idx % n == i]
    return work


def is_cached_for_model(bench: Any, model: str, item: TrackItem, no_cache: bool) -> bool:
    if no_cache:
        return False
    wav = item[2]
    if model in PYIN_MODELS:
        return bench.has_pitch_cache(bench.cache_path_for_wav(wav), smooth=PYIN_MODELS[model])
    previous = getattr(bench, "model_name", "pyin")
    bench.model_name = model
    try:
        return bench._competitor_cache_path(wav).exists()
    finally:
        bench.model_name = previous


def chunk_size_for(model: str, track_count: int, workers: int, tracks_per_task: int) -> int:
    if track_count <= 0:
        return 1
    if tracks_per_task > 0:
        return tracks_per_task
    return max(1, math.ceil(track_count / max(1, workers)))


def build_chunks(
    models: list[str],
    tracks: list[TrackItem],
    workers: int,
    tracks_per_task: int,
    kind: str,
    opts: dict[str, Any],
    skip_cached: bool,
    cache_only: bool = False,
) -> tuple[list[WorkChunk], dict[str, dict[str, int]]]:
    cache_bench = make_benchmarker(kind, opts)
    chunks: list[WorkChunk] = []
    counts: dict[str, dict[str, int]] = {}
    progress_index = 1

    # In cache-only mode we never run detection, so the caches are what we can
    # actually score: select only the cached tracks (missing ones are reported
    # as skipped afterwards) and ignore --no-cache when probing.
    cache_probe_no_cache = opts["no_cache"] and not cache_only

    for model in models:
        cached = [item for item in tracks if is_cached_for_model(cache_bench, model, item, cache_probe_no_cache)]
        if cache_only:
            selected = list(cached)
        else:
            selected = [item for item in tracks if item not in cached] if skip_cached else list(tracks)
        counts[model] = {
            "total": len(tracks),
            "cached": len(cached),
            "selected": len(selected),
        }
        size = chunk_size_for(model, len(selected), workers, tracks_per_task)
        for start in range(0, len(selected), size):
            chunks.append((model, progress_index + start, tuple(selected[start : start + size])))
        progress_index += len(selected)

    chunks.sort(key=lambda c: (c[0], c[2][0][0] if c[2] else "", c[2][0][1] if c[2] else ""))
    return chunks, counts


def run_chunk(
    chunk: WorkChunk,
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
    progress_queue: Any | None = None,
    progress_total: int = 0,
) -> tuple[str, str, list[dict[str, Any]], list[tuple[str, str, str, str]], float, str | None]:
    model, first_progress_index, items = chunk
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []

    def progress(event: str, index: int, dataset: str, track_id: str, ok: bool | None = None) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put({
                "event": event,
                "pid": os.getpid(),
                "index": index,
                "total": progress_total,
                "model": model,
                "dataset": dataset,
                "track_id": track_id,
                "ok": ok,
            })
        except (BrokenPipeError, EOFError, OSError):
            return

    try:
        bench = make_benchmarker(kind, opts)
        bench.model_name = model
        bench.use_cache = not opts["no_cache"]
        bench.algorithm_verbose = bool(opts.get("algorithm_verbose", False))
        bench.tracker = make_tracker(model, **opts)
        if bench.tracker is not None:
            bench.tracker.ensure_available()
    except MissingDependency as exc:
        return ("skip", model, rows, errors, time.perf_counter() - started, str(exc))
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return ("err", model, rows, [(model, "", "", tb)], time.perf_counter() - started, None)

    for i, (dataset, track_id, wav, annot) in enumerate(items, start=1):
        progress_index = first_progress_index + i - 1
        progress("start", progress_index, dataset, track_id)
        try:
            if model == "pyin_smoothed" and opts.get("emit_pyin_from_smoothed", False):
                track_rows = bench_pyin_family_rows(bench, wav, annot)
            else:
                row = bench.bench_pitch_track(wav, annot)
                row["model"] = model
                track_rows = [row]

            for row in track_rows:
                row["track_id"] = track_id
                row["dataset"] = dataset
                rows.append(row)
            progress("done", progress_index, dataset, track_id, ok=True)
            if verbose and progress_queue is None:
                primary = next((r for r in track_rows if r.get("model") == model), track_rows[-1])
                rtf = primary.get("realtime_factor", float("nan"))
                print(
                    f"[{model}] {i:>4}/{len(items)} {dataset:18s} {track_id[:36]:36s} "
                    f"RPA={primary['Raw Pitch Accuracy']:.3f} OA={primary['Overall Accuracy']:.3f} "
                    f"{rtf:.0f}xRT"
                    + (" (+pyin)" if len(track_rows) > 1 else ""),
                    flush=True,
                )
        except MissingDependency as exc:
            progress("done", progress_index, dataset, track_id, ok=False)
            return ("skip", model, rows, errors, time.perf_counter() - started, str(exc))
        except Exception as exc:  # noqa: BLE001 -- isolate bad tracks
            progress("done", progress_index, dataset, track_id, ok=False)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            errors.append((model, dataset, track_id, tb))
            print(f"[{model}] {dataset} / {track_id} ERROR: {exc!r}", file=sys.stderr, flush=True)

    return ("ok", model, rows, errors, time.perf_counter() - started, None)


def bench_pyin_family_rows(
    bench: Any,
    wav_path: str,
    annot_path: str,
) -> list[dict[str, Any]]:
    """Run one fresh pYIN detector pass and score both raw and smoothed stages."""
    import mir_eval

    from app_logic.user.ds.AudioData import AudioData

    ref_times, ref_freqs, fmin, fmax = bench._load_ref(
        wav_path,
        annot_path,
        True,
        196.0,
        3000.0,
    )
    cfg = bench.config_for(fmin, fmax)
    rec = bench.recording_for(cfg)
    if hasattr(bench, "load_resampled_audio"):
        rec.audio_data = bench.load_resampled_audio(wav_path, cfg.sr)
    else:
        rec.audio_data = AudioData(audio_filepath=str(wav_path), config=rec.config)

    stages, timing = bench.detect_pitch_stages(
        rec,
        make_smoothed=True,
        verbose=getattr(bench, "algorithm_verbose", False),
    )
    bench.save_pitch_cache(bench.cache_path_for_wav(wav_path), stages=stages, metadata=timing)

    secs = bench._audio_seconds(wav_path)
    meta = bench._row_meta(wav_path)
    rows: list[dict[str, Any]] = []
    for row_model, stage in (
        ("pyin", PITCH_RAW_STAGE),
        ("pyin_smoothed", PITCH_SMOOTHED_STAGE),
    ):
        if stage not in stages:
            continue
        est_times, est_freqs = bench.pitchdata_to_melody(stages[stage], cfg)
        metrics = {
            k: float(v)
            for k, v in mir_eval.melody.evaluate(
                ref_times,
                ref_freqs,
                est_times,
                est_freqs,
            ).items()
        }
        compute_time = float(timing[stage]["pitch_compute_time"])
        row = {
            **metrics,
            **timing[stage],
            "from_cache": False,
            "fmin": float(fmin),
            "fmax": float(fmax),
            "model": row_model,
            "audio_seconds": secs,
            "realtime_factor": (secs / compute_time) if (compute_time > 0 and secs) else float("nan"),
        }
        row.update(meta)
        rows.append(row)
    return rows



def run_cache_chunk(
    chunk: WorkChunk,
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
    progress_queue: Any | None = None,
    progress_total: int = 0,
) -> tuple[str, str, list[dict[str, Any]], list[tuple[str, str, str, str]], float, str | None]:
    """Parallel worker for --cache-only: re-score cached estimates, no inference.

    Mirrors ``run_chunk``'s contract so it can be driven by ``process_chunks``.
    Chunks are pre-filtered to cached tracks in ``build_chunks``, but each track
    is re-checked here to stay safe against races/corruption."""
    model, first_progress_index, items = chunk
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []

    def progress(event: str, index: int, dataset: str, track_id: str, ok: bool | None = None) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put({
                "event": event,
                "pid": os.getpid(),
                "index": index,
                "total": progress_total,
                "model": model,
                "dataset": dataset,
                "track_id": track_id,
                "ok": ok,
            })
        except (BrokenPipeError, EOFError, OSError):
            return

    cache_opts = dict(opts)
    cache_opts["no_cache"] = False
    try:
        bench = make_benchmarker(kind, cache_opts)
        bench.model_name = model
        bench.use_cache = True
        bench.algorithm_verbose = bool(opts.get("algorithm_verbose", False))
        bench.tracker = make_tracker(model, **cache_opts)
        if bench.tracker is not None:
            bench.tracker.ensure_available()
    except MissingDependency as exc:
        return ("skip", model, rows, errors, time.perf_counter() - started, str(exc))
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return ("err", model, rows, [(model, "", "", tb)], time.perf_counter() - started, None)

    for i, (dataset, track_id, wav, annot) in enumerate(items, start=1):
        progress_index = first_progress_index + i - 1
        progress("start", progress_index, dataset, track_id)
        if not is_cached_for_model(bench, model, (dataset, track_id, wav, annot), no_cache=False):
            progress("done", progress_index, dataset, track_id, ok=False)
            continue
        try:
            row = (
                score_cached_pyin(bench, wav, annot, smooth=PYIN_MODELS[model])
                if model in PYIN_MODELS
                else bench.bench_pitch_track(wav, annot)
            )
            row["track_id"] = track_id
            row["dataset"] = dataset
            row["model"] = model
            rows.append(row)
            progress("done", progress_index, dataset, track_id, ok=True)
            if verbose and progress_queue is None:
                print(
                    f"[{model}] cache {i:>4}/{len(items)} {dataset:18s} {track_id[:36]:36s} "
                    f"RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 -- isolate corrupt/incompatible caches
            progress("done", progress_index, dataset, track_id, ok=False)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            errors.append((model, dataset, track_id, tb))
            print(f"[{model}] {dataset} / {track_id} CACHE ERROR: {exc!r}", file=sys.stderr, flush=True)

    return ("ok", model, rows, errors, time.perf_counter() - started, None)


def score_cached_pyin(
    bench: Any,
    wav_path: str,
    annot_path: str,
    smooth: bool,
) -> dict[str, Any]:
    """Score cached Attune PitchData without constructing Recording."""
    import mir_eval

    ref_times, ref_freqs, fmin, fmax = bench._load_ref(
        wav_path,
        annot_path,
        True,
        196.0,
        3000.0,
    )
    cfg = bench.config_for(fmin, fmax)
    cache_path = bench.cache_path_for_wav(wav_path)
    pitch_data, metadata = bench.load_pitch_data(cache_path, cfg, smooth=smooth)
    est_times, est_freqs = bench.pitchdata_to_melody(pitch_data, cfg)
    metrics = {
        k: float(v)
        for k, v in mir_eval.melody.evaluate(
            ref_times,
            ref_freqs,
            est_times,
            est_freqs,
        ).items()
    }
    detector_time = float(metadata.get("pitch_detector_compute_time", 0.0))
    smoother_time = float(metadata.get("pitch_smoother_compute_time", 0.0))
    compute_time = float(metadata.get("pitch_compute_time", detector_time + smoother_time))
    secs = bench._audio_seconds(wav_path)
    row = {
        **metrics,
        "pitch_compute_time": compute_time,
        "from_cache": True,
        "fmin": float(fmin),
        "fmax": float(fmax),
        "model": "pyin_smoothed" if smooth else "pyin",
        "audio_seconds": secs,
        "realtime_factor": (secs / compute_time) if (compute_time > 0 and secs) else float("nan"),
    }
    row.update(bench._row_meta(wav_path))
    return row


def write_raw_outputs(pb: PitchBenchmarker, rows: list[dict[str, Any]], kind: str) -> None:
    import pandas as pd

    full = pd.DataFrame(rows)
    for model, model_df in full.groupby("model", sort=False):
        if kind == "coco":
            groups = ((f"coco_{instrument}", sub) for instrument, sub in model_df.groupby("instrument", sort=True))
        else:
            groups = ((dataset, sub) for dataset, sub in model_df.groupby("dataset", sort=True))
        for dataset_label, sub in groups:
            out_df = sub.drop(columns=["model", "dataset"], errors="ignore").set_index("track_id")
            out = pb.write_pitch_result(out_df, model, dataset_label)
            print(f"wrote {len(out_df)} rows -> {out}")


def write_summary(pb: PitchBenchmarker, rows: list[dict[str, Any]], model_order: list[str]) -> None:
    import pandas as pd

    full = pb.display_pitch_columns(pd.DataFrame(rows))
    metric_cols = [c for c in PitchBenchmarker.PITCH_METRICS if c in full.columns]
    grouped = full.groupby("model", sort=False)
    table = grouped[metric_cols].mean(numeric_only=True)
    table.insert(0, "Tracks", grouped.size())
    if {PITCH_AUDIO_SECONDS_COL, PITCH_COMPUTE_COL}.issubset(full.columns):
        totals = grouped[[PITCH_AUDIO_SECONDS_COL, PITCH_COMPUTE_COL]].sum(numeric_only=True)
        table[PITCH_REALTIME_COL] = (
            totals[PITCH_AUDIO_SECONDS_COL] / totals[PITCH_COMPUTE_COL]
        ).where(totals[PITCH_COMPUTE_COL] > 0)
    elif PITCH_REALTIME_COL in full.columns:
        table[PITCH_REALTIME_COL] = grouped[PITCH_REALTIME_COL].mean(numeric_only=True)
    table = table.reindex([m for m in model_order if m in table.index] + [m for m in table.index if m not in model_order])
    table.index.name = "model"

    print(
        f"\n{'=' * 72}\n"
        "pitch_benchmarks.csv update (metric means; throughput=sum audio/sum compute)\n"
        f"{'=' * 72}"
    )
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(table.to_string())

    out = pb.pitch_summary_csv_path
    out.parent.mkdir(parents=True, exist_ok=True)
    updated_models = [str(model) for model in table.index]
    preserved_count = 0
    if out.exists():
        existing = pd.read_csv(out)
        if "model" in existing.columns:
            existing = existing.set_index("model")
        elif len(existing.columns) > 0:
            existing = existing.rename(columns={existing.columns[0]: "model"}).set_index("model")
        else:
            existing = pd.DataFrame()
        existing.index = existing.index.astype(str)
        existing = existing[~existing.index.duplicated(keep="last")]
        existing.index.name = "model"

        merged = existing.copy()
        for col in table.columns:
            if col not in merged.columns:
                merged[col] = pd.NA
        preferred_columns = list(table.columns) + [
            col for col in merged.columns if col not in table.columns
        ]
        merged = merged.reindex(columns=preferred_columns)
        for model, row in table.iterrows():
            if model not in merged.index:
                merged.loc[model, :] = pd.NA
            merged.loc[model, table.columns] = row.to_numpy()
        preserved_count = len([idx for idx in merged.index if idx not in table.index])
    else:
        merged = table

    merged.index.name = "model"
    merged.to_csv(out)
    print(
        f"\nwrote summary -> {out} "
        f"(updated rows: {', '.join(updated_models)}; preserved {preserved_count})"
    )


def summary_model_order(models: list[str], opts: dict[str, Any]) -> list[str]:
    order = list(models)
    if opts.get("emit_pyin_from_smoothed") and "pyin_smoothed" in order and "pyin" not in order:
        order.insert(order.index("pyin_smoothed"), "pyin")
    return order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 4),
        help="process pool size (default: all logical CPUs; BLAS/OpenMP threads are capped to 1 per worker)",
    )
    p.add_argument(
        "--benchmarker",
        choices=["pitch", "coco"],
        default="pitch",
        help="'pitch' runs mdb/bach10 audio+annot corpora; 'coco' runs materialized CocoChorales splits",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="datasets/splits to run. Default: all monophonic pitch corpora for 'pitch', or 'test' for 'coco'.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"models to run. Default omits torchcrepe and penn: {DEFAULT_MODELS}. Choices: {ALL_MODELS}",
    )
    p.add_argument(
        "--include-slow-models",
        action="store_true",
        help="when --models is omitted, also include torchcrepe and penn",
    )
    p.add_argument(
        "--instrument",
        action="append",
        dest="instruments",
        help="filter selected tracks by instrument; repeatable. For smoke tests, use one instrument plus --max-tracks.",
    )
    p.add_argument(
        "--ensemble",
        action="append",
        dest="ensembles",
        help="CocoChorales-only ensemble filter; repeatable",
    )
    p.add_argument("--max-tracks", type=int, default=None, help="cap selected corpus tracks after filtering")
    p.add_argument("--per-stratum", type=int, default=None, help="CocoChorales sample cap per (ensemble,instrument)")
    p.add_argument("--seed", type=int, default=0, help="CocoChorales sampling seed")
    p.add_argument(
        "--materialize",
        action="store_true",
        help="CocoChorales-only: extract selected stems before benchmarking",
    )
    p.add_argument(
        "--force-materialize",
        action="store_true",
        help="CocoChorales-only: overwrite already materialized stems",
    )
    p.add_argument("--shard", type=parse_shard, default=None, help="run only shard i/N (e.g. 0/2)")
    p.add_argument(
        "--tracks-per-task",
        type=int,
        default=0,
        help="tracks per worker task (0 = model-aware auto partitioning)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="chunks per fresh process pool (0 = workers*2)",
    )
    p.add_argument(
        "--watchdog",
        type=float,
        default=1200.0,
        help="seconds with no track/chunk progress before in-flight chunks are treated as stuck (default 1200)",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="how many times a stuck/failed chunk is retried before giving up (default 2)",
    )
    p.add_argument(
        "--skip-cached",
        action="store_true",
        help="skip model/track pairs with an existing cache; skipped pairs are left out of CSVs",
    )
    p.add_argument("--no-cache", action="store_true", help="ignore + rewrite pitch estimate caches")
    p.add_argument("--confidence", type=float, default=None, help="override per-model voicing threshold")
    p.add_argument(
        "--pyin-unv-thresh",
        type=float,
        default=None,
        help=(
            "override Attune pYIN unvoiced-probability threshold; lower is "
            f"stricter. Default is {PYIN_DEFAULT_UNV_THRESH:.2f}; Praat's "
            f"voicing_threshold=0.45 corresponds to {PYIN_PRAAT_MIRROR_UNV_THRESH:.2f} here."
        ),
    )
    p.add_argument(
        "--mirror-praat-voicing",
        action="store_true",
        help=(
            "set --pyin-unv-thresh to 0.55, the pYIN analogue to Praat's "
            "voicing_threshold=0.45; stricter than the default 0.90"
        ),
    )
    p.add_argument(
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
    p.add_argument(
        "--pyin-volume-gate-percentile",
        type=float,
        default=None,
        help=(
            "frame-RMS percentile used as the pYIN volume-gate reference "
            f"(default {PYIN_DEFAULT_MAX_VOLUME * 100:.1f}; use 100 for the old max-frame gate)"
        ),
    )
    p.add_argument("--step", type=float, default=0.01, dest="step_seconds", help="frame hop seconds (default 0.01)")
    p.add_argument("--crepe-capacity", default="full", choices=["tiny", "small", "medium", "large", "full"])
    p.add_argument("--rmvpe-checkpoint", default=None, help="path to rmvpe.pt")
    p.add_argument("--rmvpe-module", default=None, help="dotted module exposing class RMVPE")
    p.add_argument("--root", default=None, help="CocoChorales dataset root override")
    p.add_argument("--f0-fps", type=float, default=F0_FPS_DEFAULT, help="CocoChorales f0 frame rate")
    p.add_argument(
        "--quiet-tracks",
        action="store_true",
        help="suppress legacy per-track worker lines; live progress remains enabled unless --no-progress is set",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="disable live per-track progress lines",
    )
    p.add_argument(
        "--algorithm-verbose",
        action="store_true",
        help="let pitch/note/mistake algorithms print their own verbose diagnostics inside workers",
    )
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="re-score existing caches and write CSVs without running pitch algorithms",
    )
    p.add_argument("--dry-run", action="store_true", help="print the work plan and exit; detect nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.datasets is None:
        args.datasets = ["test"] if args.benchmarker == "coco" else PitchBenchmarker.PITCH_DATASETS

    models = list(args.models or DEFAULT_MODELS)
    if args.models is None and args.include_slow_models:
        models = list(ALL_MODELS)
    unknown = [model for model in models if model not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choices: {ALL_MODELS}")
    pyin_unv_thresh = (
        PYIN_PRAAT_MIRROR_UNV_THRESH
        if args.mirror_praat_voicing and args.pyin_unv_thresh is None
        else args.pyin_unv_thresh
    )
    pyin_volume_gate_ratio = args.pyin_volume_gate_ratio
    pyin_volume_gate_percentile = args.pyin_volume_gate_percentile
    emit_pyin_from_smoothed = (
        args.no_cache
        and not args.cache_only
        and "pyin_smoothed" in models
        and "pyin" not in models
    )

    opts: dict[str, Any] = {
        "no_cache": args.no_cache,
        "emit_pyin_from_smoothed": emit_pyin_from_smoothed,
        "confidence": args.confidence,
        "pyin_unv_thresh": pyin_unv_thresh,
        "pyin_volume_gate_ratio": pyin_volume_gate_ratio,
        "pyin_volume_gate_percentile": pyin_volume_gate_percentile,
        "step_seconds": args.step_seconds,
        "crepe_capacity": args.crepe_capacity,
        "rmvpe_checkpoint": args.rmvpe_checkpoint,
        "rmvpe_module": args.rmvpe_module,
        "root": args.root,
        "f0_fps": args.f0_fps,
        "instruments": args.instruments,
        "ensembles": args.ensembles,
        "max_tracks": args.max_tracks,
        "per_stratum": args.per_stratum,
        "seed": args.seed,
        "materialize": args.materialize and not args.dry_run,
        "force_materialize": args.force_materialize,
        "algorithm_verbose": args.algorithm_verbose,
    }

    tracks = list_work(args.datasets, args.shard, kind=args.benchmarker, opts=opts)
    chunks, counts = build_chunks(
        models,
        tracks,
        workers=args.workers,
        tracks_per_task=args.tracks_per_task,
        kind=args.benchmarker,
        opts=opts,
        skip_cached=args.skip_cached,
        cache_only=args.cache_only,
    )

    total_model_tracks = len(models) * len(tracks)
    total_cached = sum(c["cached"] for c in counts.values())
    total_selected = sum(c["selected"] for c in counts.values())
    batch_size = args.batch_size if args.batch_size > 0 else max(1, args.workers * 2)

    print(f"benchmarker: {args.benchmarker}")
    print(f"datasets:   {', '.join(args.datasets)}")
    print(f"models:     {', '.join(models)}")
    if emit_pyin_from_smoothed:
        print("companions: pyin rows scored from the same pyin_smoothed detector pass")
    if any(model in PYIN_MODELS for model in models):
        effective_pyin_unv = (
            pyin_unv_thresh
            if pyin_unv_thresh is not None
            else PYIN_DEFAULT_UNV_THRESH
        )
        source = "override" if pyin_unv_thresh is not None else "default"
        print(f"pYIN unv:   {effective_pyin_unv:.3f} ({source})")
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
            f"pYIN gate:  {effective_gate:.3f} ({gate_source}) "
            f"* p{effective_gate_percentile:g} RMS ({percentile_source})"
        )
    if args.instruments:
        print(f"instruments:{' ' * 3}{', '.join(args.instruments)}")
    if args.ensembles:
        print(f"ensembles:  {', '.join(args.ensembles)}")
    if args.max_tracks is not None:
        print(f"max tracks: {args.max_tracks}")
    if args.materialize:
        print(f"materialize:{' yes' if not args.dry_run else ' dry-run only'}")
    print(f"shard:      {f'{args.shard[0]}/{args.shard[1]}' if args.shard else 'all'}")
    print(f"workers:    {args.workers}  (cpu_count={os.cpu_count()}, BLAS threads capped to 1/worker)")
    print(f"tracks:     {len(tracks)} corpus tracks | {total_model_tracks} model/track pairs")
    print(f"cache:      {total_cached} cached | {total_selected} selected for this run")
    if args.cache_only:
        print("mode:       cache-only CSV rebuild (parallel); no pitch algorithms will run")
    print(f"chunks:     {len(chunks)} total | batch={batch_size} | watchdog={fmt_dur(args.watchdog)}")
    for model in models:
        c = counts[model]
        print(f"  - {model:14s} {c['cached']:>4}/{c['total']:<4} cached | {c['selected']:>4} selected")

    if args.dry_run:
        print(f"\n[dry-run] would process {total_selected} model/track pair(s); no detection performed.")
        return 0
    if not tracks:
        print("\nno tracks selected.")
        if args.benchmarker == "coco":
            pb = make_benchmarker("coco", opts)
            selected = 0
            materialized = 0
            for ds in args.datasets:
                records = pb.select_records(
                    split=ds,
                    per_stratum=args.per_stratum,
                    seed=args.seed,
                    max_tracks=args.max_tracks,
                    ensembles=args.ensembles,
                    instruments=args.instruments,
                    rebuild_manifest=False,
                )
                selected += len(records)
                materialized += len(pb.records_to_tracks(records))
            if selected and materialized == 0:
                print(
                    f"CocoChorales selected {selected} manifest row(s), but none are materialized. "
                    "Re-run this command with --materialize, or materialize them first with "
                    "benchmarks/scripts/benchmark_cocochorales.py --materialize."
                )
        return 1
    if not chunks:
        print("\nnothing to do.")
        return 0

    if args.cache_only:
        print(
            f"\nre-scoring {total_selected} cached model/track pair(s) across "
            f"{args.workers} worker(s) at {time.strftime('%Y-%m-%d %H:%M:%S')} ...\n",
            flush=True,
        )
        started = time.perf_counter()
        rows, errors, skipped = process_chunks(
            run_cache_chunk,
            chunks,
            workers=args.workers,
            batch_size=batch_size,
            watchdog=args.watchdog,
            max_attempts=args.max_attempts,
            kind=args.benchmarker,
            opts=opts,
            verbose=not args.quiet_tracks,
            progress=not args.no_progress,
        )
        total = time.perf_counter() - started
        # Tracks whose cache was missing were dropped from the chunks up front;
        # surface that here so a partial rebuild is obvious.
        for model in models:
            missing = counts[model]["total"] - counts[model]["cached"]
            if missing and model not in skipped:
                skipped[model] = f"{missing}/{counts[model]['total']} selected track(s) did not have a cache"
        pb = make_benchmarker(args.benchmarker, opts)
        if rows:
            write_raw_outputs(pb, rows, args.benchmarker)
            write_summary(pb, rows, summary_model_order(models, opts))
        print(f"\ndone: rebuilt {len(rows)} cached rows, {len(errors)} cache errors in {fmt_dur(total)}")
        if skipped:
            print("skipped/missing caches:")
            for model, reason in sorted(skipped.items()):
                print(f"  - {model}: {reason}")
        if errors:
            print("cache error tracks:")
            for model, dataset, track_id, _ in errors:
                print(f"  - {model} / {dataset} / {track_id}")
            return 1
        return 0

    print(
        f"\nstarting {total_selected} model/track pair(s) at {time.strftime('%Y-%m-%d %H:%M:%S')} ...\n",
        flush=True,
    )
    started = time.perf_counter()
    rows, errors, skipped = process_chunks(
        run_chunk,
        chunks,
        workers=args.workers,
        batch_size=batch_size,
        watchdog=args.watchdog,
        max_attempts=args.max_attempts,
        kind=args.benchmarker,
        opts=opts,
        verbose=not args.quiet_tracks,
        progress=not args.no_progress,
    )
    total = time.perf_counter() - started

    pb = make_benchmarker(args.benchmarker, opts)
    if rows:
        write_raw_outputs(pb, rows, args.benchmarker)
        write_summary(pb, rows, summary_model_order(models, opts))

    print(f"\ndone: {len(rows)} rows, {len(errors)} track errors, {len(skipped)} skipped model(s) in {fmt_dur(total)}")
    if skipped:
        print("skipped models:")
        for model, reason in sorted(skipped.items()):
            first_line = reason.splitlines()[0] if reason else "dependency unavailable"
            print(f"  - {model}: {first_line}")
    if errors:
        print("failed tracks:")
        for model, dataset, track_id, _ in errors:
            print(f"  - {model} / {dataset} / {track_id}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
