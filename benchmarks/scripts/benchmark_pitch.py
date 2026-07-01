#!/usr/bin/env python3
"""Parallel, resumable pitch benchmark runner.

Runs Attune pYIN plus third-party competitors over the selected pitch corpora.
Each method writes raw per-track CSVs under
``benchmarks/results/pitch/raw_outputs/<method>/`` and the final comparison table
to ``benchmarks/results/pitch/pitch_benchmarks.csv``.

The work is partitioned by model-aware chunks: heavy neural competitors keep all
selected tracks in one worker by default so the model is loaded once, while
lightweight/CPU-bound methods such as pYIN and Praat are split across workers.
Use ``--tracks-per-task`` to override that partitioning.
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
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
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
    PITCH_COMPUTE_COL,
    PITCH_REALTIME_COL,
    PitchBenchmarker,
)

TrackItem = tuple[str, str, str, str]  # dataset, track_id, wav_path, annot_path
WorkChunk = tuple[str, tuple[TrackItem, ...]]  # model, tracks

HEAVY_MODEL_LOADS = {"swiftf0", "crepe", "torchcrepe", "penn", "spice", "rmvpe"}


def make_benchmarker(kind: str, opts: dict[str, Any] | None = None):
    opts = opts or {}
    if kind == "coco":
        return CompetitorCocoBenchmarker(
            root=opts.get("root"),
            f0_fps=float(opts.get("f0_fps", F0_FPS_DEFAULT)),
        )
    return CompetitorPitchBenchmarker()


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
    ):
        for ds in datasets:
            records = pb.select_records(
                split=ds,
                per_stratum=opts.get("per_stratum"),
                seed=int(opts.get("seed", 0)),
                max_tracks=opts.get("max_tracks"),
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
    if kind != "coco" and opts.get("max_tracks") is not None:
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
        return bool(PYIN_MODELS[model] and bench.cache_path_for_wav(wav).exists())
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
    if model in HEAVY_MODEL_LOADS:
        return track_count
    return max(1, math.ceil(track_count / max(1, workers)))


def build_chunks(
    models: list[str],
    tracks: list[TrackItem],
    workers: int,
    tracks_per_task: int,
    kind: str,
    opts: dict[str, Any],
    skip_cached: bool,
) -> tuple[list[WorkChunk], dict[str, dict[str, int]]]:
    cache_bench = make_benchmarker(kind, opts)
    chunks: list[WorkChunk] = []
    counts: dict[str, dict[str, int]] = {}

    for model in models:
        cached = [item for item in tracks if is_cached_for_model(cache_bench, model, item, opts["no_cache"])]
        selected = [item for item in tracks if item not in cached] if skip_cached else list(tracks)
        counts[model] = {
            "total": len(tracks),
            "cached": len(cached),
            "selected": len(selected),
        }
        size = chunk_size_for(model, len(selected), workers, tracks_per_task)
        for start in range(0, len(selected), size):
            chunks.append((model, tuple(selected[start : start + size])))

    chunks.sort(key=lambda c: (c[0], c[1][0][0] if c[1] else "", c[1][0][1] if c[1] else ""))
    return chunks, counts


def run_chunk(
    chunk: WorkChunk,
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
) -> tuple[str, str, list[dict[str, Any]], list[tuple[str, str, str, str]], float, str | None]:
    model, items = chunk
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []

    try:
        bench = make_benchmarker(kind, opts)
        bench.model_name = model
        bench.use_cache = not opts["no_cache"]
        bench.tracker = make_tracker(model, **opts)
        if bench.tracker is not None:
            bench.tracker.ensure_available()
    except MissingDependency as exc:
        return ("skip", model, rows, errors, time.perf_counter() - started, str(exc))
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return ("err", model, rows, [(model, "", "", tb)], time.perf_counter() - started, None)

    for i, (dataset, track_id, wav, annot) in enumerate(items, start=1):
        try:
            row = bench.bench_pitch_track(wav, annot)
            row["track_id"] = track_id
            row["dataset"] = dataset
            row["model"] = model
            rows.append(row)
            if verbose:
                rtf = row.get("realtime_factor", float("nan"))
                print(
                    f"[{model}] {i:>4}/{len(items)} {dataset:18s} {track_id[:36]:36s} "
                    f"RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f} "
                    f"{rtf:.0f}xRT",
                    flush=True,
                )
        except MissingDependency as exc:
            return ("skip", model, rows, errors, time.perf_counter() - started, str(exc))
        except Exception as exc:  # noqa: BLE001 -- isolate bad tracks
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            errors.append((model, dataset, track_id, tb))
            print(f"[{model}] {dataset} / {track_id} ERROR: {exc!r}", file=sys.stderr, flush=True)

    return ("ok", model, rows, errors, time.perf_counter() - started, None)


def fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    i, n = (int(x) for x in text.split("/"))
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError(f"shard index {i} out of range for {n} shards")
    return i, n


def _force_teardown(ex: ProcessPoolExecutor) -> None:
    ex.shutdown(wait=False, cancel_futures=True)
    for child in multiprocessing.active_children():
        child.terminate()
    for child in multiprocessing.active_children():
        child.join(timeout=10)
        if child.is_alive():
            child.kill()


def process_chunks(
    chunks: list[WorkChunk],
    workers: int,
    batch_size: int,
    watchdog: float,
    max_attempts: int,
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, str]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []
    skipped: dict[str, str] = {}
    attempts: dict[str, int] = {}
    queue = list(chunks)
    total_tracks = sum(len(items) for _, items in chunks)
    completed_tracks = 0
    started = time.perf_counter()

    def chunk_key(chunk: WorkChunk) -> str:
        model, items = chunk
        if not items:
            return f"{model}:empty"
        return f"{model}:{items[0][0]}:{items[0][1]}:{items[-1][0]}:{items[-1][1]}:{len(items)}"

    def log(tag: str, model: str, n: int, dur: float, extra: str = "") -> None:
        nonlocal completed_tracks
        completed_tracks += n
        elapsed = time.perf_counter() - started
        rate = completed_tracks / elapsed if elapsed else 0.0
        eta = (total_tracks - completed_tracks) / rate if rate else 0.0
        print(
            f"[{completed_tracks:>4}/{total_tracks}] {tag:7s} {model:14s} "
            f"{n:>4} track(s) in {fmt_dur(dur)} {extra}"
            f"| elapsed {fmt_dur(elapsed)} eta {fmt_dur(eta)}",
            flush=True,
        )

    while queue:
        batch, queue = queue[:batch_size], queue[batch_size:]
        runnable: list[WorkChunk] = []
        for chunk in batch:
            key = chunk_key(chunk)
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] > max_attempts:
                model, items = chunk
                tb = f"gave up after {max_attempts} attempts (kept hanging/failing)"
                for dataset, track_id, *_ in items:
                    errors.append((model, dataset, track_id, tb))
                log("GIVEUP", model, len(items), 0.0)
            else:
                runnable.append(chunk)
        if not runnable:
            continue

        ex = ProcessPoolExecutor(max_workers=workers)
        futs = {ex.submit(run_chunk, chunk, kind, opts, verbose): chunk for chunk in runnable}
        try:
            pending = set(futs)
            while pending:
                done_set, pending = wait(pending, timeout=watchdog, return_when=FIRST_COMPLETED)
                if not done_set:
                    stuck = [futs[fut] for fut in pending]
                    print(
                        f"\n!! watchdog: no chunk finished in {fmt_dur(watchdog)} -- "
                        f"re-queueing {len(stuck)} in-flight chunk(s).",
                        file=sys.stderr,
                        flush=True,
                    )
                    for model, items in stuck:
                        first = items[0][1] if items else "empty"
                        print(f"   stuck: {model} / {first} ({len(items)} tracks)", file=sys.stderr, flush=True)
                    queue.extend(stuck)
                    break

                broke = False
                for fut in done_set:
                    chunk = futs[fut]
                    model, items = chunk
                    try:
                        status, result_model, chunk_rows, chunk_errors, dur, skip_msg = fut.result()
                    except BrokenProcessPool:
                        broke = True
                        break

                    if status == "skip":
                        rows.extend(chunk_rows)
                        errors.extend(chunk_errors)
                        skipped[result_model] = skip_msg or "dependency unavailable"
                        log("SKIP", result_model, len(items), dur)
                        print(f"[{result_model}] SKIPPED -- {skipped[result_model]}", flush=True)
                        continue

                    rows.extend(chunk_rows)
                    errors.extend(chunk_errors)
                    if status == "ok":
                        log("OK", model, len(chunk_rows) + len(chunk_errors), dur)
                    else:
                        log("ERR", model, len(items), dur)
                        for _, _, _, tb in chunk_errors:
                            print(tb.rstrip(), file=sys.stderr, flush=True)

                if broke:
                    stuck = [chunk, *[futs[fut] for fut in pending]]
                    print(
                        f"\n!! pool broke (a worker died) -- re-queueing {len(stuck)} unfinished chunk(s).",
                        file=sys.stderr,
                        flush=True,
                    )
                    queue.extend(stuck)
                    break
        finally:
            _force_teardown(ex)

    return rows, errors, skipped


def rebuild_from_caches(
    models: list[str],
    tracks: list[TrackItem],
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, str]], dict[str, str]]:
    """Re-score cached pitch estimates and write rows without running inference."""
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []
    skipped: dict[str, str] = {}

    cache_opts = dict(opts)
    cache_opts["no_cache"] = False

    for model in models:
        if model == "pyin":
            skipped[model] = (
                "pyin stage-1 output is not persisted; only pyin_smoothed "
                "PitchData and third-party competitor estimates can be rebuilt from cache."
            )
            print(f"[{model}] SKIPPED -- {skipped[model]}", flush=True)
            continue

        bench = make_benchmarker(kind, cache_opts)
        bench.model_name = model
        bench.use_cache = True
        bench.tracker = make_tracker(model, **cache_opts)

        missing = 0
        produced = 0
        for dataset, track_id, wav, annot in tracks:
            if not is_cached_for_model(bench, model, (dataset, track_id, wav, annot), no_cache=False):
                missing += 1
                continue
            try:
                row = (
                    score_cached_pyin_smoothed(bench, wav, annot)
                    if model == "pyin_smoothed"
                    else bench.bench_pitch_track(wav, annot)
                )
                row["track_id"] = track_id
                row["dataset"] = dataset
                row["model"] = model
                rows.append(row)
                produced += 1
                if verbose:
                    print(
                        f"[{model}] cache {produced:>4} {dataset:18s} {track_id[:36]:36s} "
                        f"RPA={row['Raw Pitch Accuracy']:.3f} OA={row['Overall Accuracy']:.3f}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 -- report corrupt/incompatible caches
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                errors.append((model, dataset, track_id, tb))
                print(f"[{model}] {dataset} / {track_id} CACHE ERROR: {exc!r}", file=sys.stderr, flush=True)

        if missing:
            skipped[model] = f"{missing}/{len(tracks)} selected track(s) did not have a cache"
        print(
            f"[{model}] rebuilt {produced}/{len(tracks)} cached row(s)"
            + (f"; missing {missing}" if missing else ""),
            flush=True,
        )

    return rows, errors, skipped


def score_cached_pyin_smoothed(
    bench: Any,
    wav_path: str,
    annot_path: str,
) -> dict[str, Any]:
    """Score cached smoothed Attune PitchData without constructing Recording."""
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
    pitch_data, metadata = bench.load_pitch_data(cache_path, cfg)
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
        "model": "pyin_smoothed",
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
    cols = [*PitchBenchmarker.PITCH_METRICS, PITCH_COMPUTE_COL, PITCH_REALTIME_COL]
    table = full.groupby("model")[[c for c in cols if c in full.columns]].mean(numeric_only=True)
    table.insert(0, "Tracks", full.groupby("model").size())
    table = table.reindex([m for m in model_order if m in table.index] + [m for m in table.index if m not in model_order])
    table.index.name = "model"

    print(f"\n{'=' * 72}\npitch_benchmarks.csv (mean over tracks)\n{'=' * 72}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(table.to_string())

    out = pb.pitch_summary_csv_path
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\nwrote summary -> {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 4),
        help="process pool size (default: cores - 4; BLAS/OpenMP threads are capped to 1 per worker)",
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
        help="seconds with NO chunk finishing before in-flight chunks are treated as stuck (default 1200)",
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
    p.add_argument("--no-cache", action="store_true", help="ignore + rewrite competitor estimate caches")
    p.add_argument("--confidence", type=float, default=None, help="override per-model voicing threshold")
    p.add_argument("--step", type=float, default=0.01, dest="step_seconds", help="frame hop seconds (default 0.01)")
    p.add_argument("--crepe-capacity", default="full", choices=["tiny", "small", "medium", "large", "full"])
    p.add_argument("--rmvpe-checkpoint", default=None, help="path to rmvpe.pt")
    p.add_argument("--rmvpe-module", default=None, help="dotted module exposing class RMVPE")
    p.add_argument("--root", default=None, help="CocoChorales dataset root override")
    p.add_argument("--f0-fps", type=float, default=F0_FPS_DEFAULT, help="CocoChorales f0 frame rate")
    p.add_argument("--quiet-tracks", action="store_true", help="suppress per-track worker progress lines")
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

    opts: dict[str, Any] = {
        "no_cache": args.no_cache,
        "confidence": args.confidence,
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
    )

    total_model_tracks = len(models) * len(tracks)
    total_cached = sum(c["cached"] for c in counts.values())
    total_selected = sum(c["selected"] for c in counts.values())
    batch_size = args.batch_size if args.batch_size > 0 else max(1, args.workers * 2)

    print(f"benchmarker: {args.benchmarker}")
    print(f"datasets:   {', '.join(args.datasets)}")
    print(f"models:     {', '.join(models)}")
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
        print("mode:       cache-only CSV rebuild; no pitch algorithms will run")
    else:
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
        started = time.perf_counter()
        rows, errors, skipped = rebuild_from_caches(
            models,
            tracks,
            kind=args.benchmarker,
            opts=opts,
            verbose=not args.quiet_tracks,
        )
        total = time.perf_counter() - started
        pb = make_benchmarker(args.benchmarker, opts)
        if rows:
            write_raw_outputs(pb, rows, args.benchmarker)
            write_summary(pb, rows, models)
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
        chunks,
        workers=args.workers,
        batch_size=batch_size,
        watchdog=args.watchdog,
        max_attempts=args.max_attempts,
        kind=args.benchmarker,
        opts=opts,
        verbose=not args.quiet_tracks,
    )
    total = time.perf_counter() - started

    pb = make_benchmarker(args.benchmarker, opts)
    if rows:
        write_raw_outputs(pb, rows, args.benchmarker)
        write_summary(pb, rows, models)

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
