#!/usr/bin/env python3
"""Parallel, resumable note-detection benchmark runner.

Mirrors ``benchmark_pitch.py``: it runs the note-detector method matrix over the
selected corpus in a fresh-pool-per-batch ``ProcessPoolExecutor`` with a live
``[i/N] <method>: <title> [/]`` progress display, a no-progress watchdog, and
per-method skip/error isolation. Each method writes a raw per-track CSV under
``benchmarks/results/note/raw_outputs/<method>/`` and the comparison table to
``benchmarks/results/note/note_benchmarks.csv``.

Unlike pitch, note detection consumes the ALREADY-CACHED ``pyin_smooth`` pitch
track (produced by ``benchmark_pitch.py``); pitch detection is never re-run and
never timed. The rows carry only the note detector's own compute time and
Audio(s)/Compute(s).

Corpora (``--benchmarker``):
  - ``coco``  : materialized CocoChorales stems (reference = stems_midi). Default.
  - ``etude`` : violin etudes (WAV synthesized from the reference MIDI).

Examples:
    python benchmarks/scripts/benchmark_notes.py --list-methods
    python benchmarks/scripts/benchmark_notes.py --benchmarker coco --instrument violin --per-stratum 10
    python benchmarks/scripts/benchmark_notes.py --benchmarker etude --dataset kayser --max-tracks 2
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
from benchmarks.modules.note.CocoNoteBenchmarker import CocoNoteBenchmarker  # noqa: E402
from benchmarks.modules.note.NoteBenchmarker import (  # noqa: E402
    _ALL_NOTE_METHODS,
    _EXTERNAL_NOTE_METHODS,
    NoteBenchmarker,
)
from benchmarks.modules.runner import (  # noqa: E402
    TrackItem,
    WorkChunk,
    fmt_dur,
    parse_shard,
    process_chunks,
)

ALL_METHODS: list[str] = list(_ALL_NOTE_METHODS)
CPD_METHODS: list[str] = [m for m in ALL_METHODS if m not in _EXTERNAL_NOTE_METHODS]


def _is_missing_dependency(exc: Exception) -> bool:
    """External baselines (basic-pitch / tony / crepe-notes) raise a RuntimeError
    whose message flags an unmet dependency. Treat those as a whole-method skip
    rather than one failure per track, matching the pitch runner's behaviour."""
    text = str(exc).lower()
    return isinstance(exc, (RuntimeError, ImportError, ModuleNotFoundError, FileNotFoundError)) and any(
        needle in text for needle in ("unavailable", "not found", "no module", "not installed", "missing")
    )


def make_benchmarker(kind: str, opts: dict[str, Any] | None = None):
    opts = opts or {}
    if kind == "coco":
        return CocoNoteBenchmarker(
            root=opts.get("root"),
            f0_fps=float(opts.get("f0_fps", F0_FPS_DEFAULT)),
            onset_tolerance=float(opts.get("onset_tolerance", 0.05)),
        )
    return NoteBenchmarker(onset_tolerance=float(opts.get("onset_tolerance", 0.05)))


def list_work(
    datasets: list[str],
    shard: tuple[int, int] | None = None,
    kind: str = "coco",
    opts: dict[str, Any] | None = None,
) -> list[TrackItem]:
    opts = opts or {}
    pb = make_benchmarker(kind, opts)
    work: list[TrackItem] = []
    max_tracks = opts.get("max_tracks")
    if kind == "coco":
        for ds in datasets:
            # NOTE: max_tracks is applied to the WORK list below, AFTER the
            # materialized-only filter -- not inside select_records. Otherwise a
            # cap would truncate the (mostly un-materialized) manifest first and
            # records_to_note_tracks could then drop every survivor, yielding 0
            # runnable tracks. When materializing, cap what we extract up front.
            records = pb.select_records(
                split=ds,
                per_stratum=opts.get("per_stratum"),
                seed=int(opts.get("seed", 0)),
                max_tracks=None,
                ensembles=opts.get("ensembles"),
                instruments=opts.get("instruments"),
                rebuild_manifest=False,
            )
            if opts.get("materialize"):
                to_extract = records[:max_tracks] if max_tracks is not None else records
                written = pb.materialize_records(
                    to_extract, force=bool(opts.get("force_materialize", False)),
                )
                print(f"materialized {len(written)} file(s) for {ds}", flush=True)
            for track_id, wav, midi in pb.records_to_note_tracks(records):
                work.append((ds, track_id, str(wav), str(midi)))
    else:
        for ds in datasets:
            for title, midi in pb.iter_etudes(ds):
                work.append((ds, title, str(midi), str(midi)))
    work.sort()
    if kind == "etude" and opts.get("instruments"):
        wanted = [str(name).lower() for name in opts["instruments"]]
        work = [item for item in work if any(name in item[1].lower() for name in wanted)]
    if max_tracks is not None:
        work = work[: int(max_tracks)]
    if shard is not None:
        i, n = shard
        work = [w for idx, w in enumerate(work) if idx % n == i]
    return work


def chunk_size_for(track_count: int, workers: int, tracks_per_task: int) -> int:
    if track_count <= 0:
        return 1
    if tracks_per_task > 0:
        return tracks_per_task
    return max(1, math.ceil(track_count / max(1, workers)))


def build_chunks(
    methods: list[str],
    tracks: list[TrackItem],
    workers: int,
    tracks_per_task: int,
) -> list[WorkChunk]:
    chunks: list[WorkChunk] = []
    progress_index = 1
    for method in methods:
        size = chunk_size_for(len(tracks), workers, tracks_per_task)
        for start in range(0, len(tracks), size):
            chunks.append((method, progress_index + start, tuple(tracks[start : start + size])))
        progress_index += len(tracks)
    chunks.sort(key=lambda c: (c[0], c[2][0][0] if c[2] else "", c[2][0][1] if c[2] else ""))
    return chunks


def run_chunk(
    chunk: WorkChunk,
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
    progress_queue: Any | None = None,
    progress_total: int = 0,
) -> tuple[str, str, list[dict[str, Any]], list[tuple[str, str, str, str]], float, str | None]:
    method, first_progress_index, items = chunk
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []

    def progress(event: str, index: int, dataset: str, track_id: str, ok: bool | None = None) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put({
                "event": event, "pid": os.getpid(), "index": index,
                "total": progress_total, "model": method,
                "dataset": dataset, "track_id": track_id, "ok": ok,
            })
        except (BrokenPipeError, EOFError, OSError):
            return

    try:
        bench = make_benchmarker(kind, opts)
        bench.algorithm_verbose = bool(opts.get("algorithm_verbose", False))
        method_config = bench.ALL_NOTE_METHODS[method]
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return ("err", method, rows, [(method, "", "", tb)], time.perf_counter() - started, None)

    for i, (dataset, track_id, primary, reference) in enumerate(items, start=1):
        progress_index = first_progress_index + i - 1
        progress("start", progress_index, dataset, track_id)
        try:
            row = bench.score_note_track(
                primary, reference, method, method_config,
                align=opts.get("align", "identity"),
                latency_align=bool(opts.get("latency_align", True)),
                trim_boundaries=bool(opts.get("trim_boundaries", True)),
                onset_tolerance=opts.get("onset_tolerance"),
            )
            row["track_id"] = track_id
            row["dataset"] = dataset
            row["method"] = method
            if kind == "coco":
                row.update(bench.note_row_meta(primary))
            rows.append(row)
            progress("done", progress_index, dataset, track_id, ok=(row.get("error") is None))
            if verbose and progress_queue is None:
                fval = row.get("F-measure", float("nan"))
                rtf = row.get("realtime_factor", float("nan"))
                print(
                    f"[{method}] {i:>4}/{len(items)} {dataset:16s} {track_id[:40]:40s} "
                    f"F={fval:.3f} {rtf:.0f}xRT",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 -- isolate a bad track
            progress("done", progress_index, dataset, track_id, ok=False)
            if _is_missing_dependency(exc):
                # skip the whole method instead of erroring on every track
                return ("skip", method, rows, errors, time.perf_counter() - started, str(exc))
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            errors.append((method, dataset, track_id, tb))
            print(f"[{method}] {dataset} / {track_id} ERROR: {exc!r}", file=sys.stderr, flush=True)

    return ("ok", method, rows, errors, time.perf_counter() - started, None)


def write_raw_outputs(pb: NoteBenchmarker, rows: list[dict[str, Any]], kind: str) -> None:
    import pandas as pd

    full = pd.DataFrame(rows)
    for method, method_df in full.groupby("method", sort=False):
        if kind == "coco" and "instrument" in method_df.columns:
            groups = ((f"coco_{instrument}", sub) for instrument, sub in method_df.groupby("instrument", sort=True))
        else:
            groups = ((dataset, sub) for dataset, sub in method_df.groupby("dataset", sort=True))
        for dataset_label, sub in groups:
            out_df = sub.drop(columns=["method", "dataset"], errors="ignore").set_index("track_id")
            out = pb.write_note_result(out_df, method, dataset_label)
            print(f"wrote {len(out_df)} rows -> {out}")


def write_summary(pb: NoteBenchmarker, rows: list[dict[str, Any]], method_order: list[str]) -> None:
    import pandas as pd

    full = pd.DataFrame(rows)
    # summary keeps ONLY the note-detection quality + Audio(s)/Compute(s); the raw
    # note compute time is intentionally left out (it lives in the per-track CSVs).
    metric_cols = [c for c in pb.NOTE_QUALITY_METRICS if c in full.columns]
    cols = [*metric_cols, "realtime_factor"]
    table = full.groupby("method")[[c for c in cols if c in full.columns]].mean(numeric_only=True)
    table.insert(0, "Tracks", full.groupby("method").size())
    table = table.rename(columns={"realtime_factor": pb.REALTIME_COL})
    table = table.reindex(
        [m for m in method_order if m in table.index]
        + [m for m in table.index if m not in method_order]
    )
    table.index.name = "method"

    print(f"\n{'=' * 72}\nnote_benchmarks.csv (mean over tracks)\n{'=' * 72}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(table.to_string())

    out = pb.note_summary_csv_path
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\nwrote summary -> {out}")


def select_methods(args: argparse.Namespace) -> list[str]:
    if args.methods:
        unknown = [m for m in args.methods if m not in _ALL_NOTE_METHODS]
        if unknown:
            raise SystemExit(f"unknown methods: {unknown}; choices: {ALL_METHODS}")
        methods = list(args.methods)
    elif args.cpd_only:
        methods = list(CPD_METHODS)
    else:
        methods = list(ALL_METHODS)
    if args.no_external:
        methods = [m for m in methods if m not in _EXTERNAL_NOTE_METHODS]
    return methods


def print_method_matrix() -> None:
    for label, config in _ALL_NOTE_METHODS.items():
        family = config.get("method")
        algo = config.get("ruptures_algorithm", "-")
        model = config.get("model", "-")
        post = config.get("postprocess_transitions", False)
        group = config.get("benchmark_group", "-")
        print(
            f"{label:22s} group={group:9s} family={family:12s} "
            f"algorithm={algo:9s} model={model:7s} prune+remedian={post}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarker", choices=["coco", "etude"], default="coco",
                   help="'coco' runs materialized CocoChorales stems (default); 'etude' runs violin etudes")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="splits/datasets. Default: 'test' for coco, all etude datasets for etude.")
    p.add_argument("--method", action="append", dest="methods",
                   help="method label to run; repeatable; default = the full matrix")
    p.add_argument("--cpd-only", action="store_true", help="run only the change-point families (no moreover baselines)")
    p.add_argument("--no-external", action="store_true", help="drop basic-pitch / tony / crepe-notes (unmet deps)")
    p.add_argument("--list-methods", action="store_true", help="print the method matrix and exit")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4),
                   help="process pool size (default: all logical CPUs; BLAS threads capped to 1/worker)")
    p.add_argument("--align", choices=["identity", "resize"], default="identity",
                   help="identity isolates note detection; resize mirrors the app pipeline")
    p.add_argument("--no-latency-align", action="store_true", help="disable the constant detector-lag shift (identity mode)")
    p.add_argument("--no-trim-boundaries", action="store_true", help="do not clamp first/last notes to reference durations")
    p.add_argument("--onset-tolerance", type=float, default=0.05, help="mir_eval onset match tolerance in seconds")
    # coco selection (mirrors benchmark_pitch.py)
    p.add_argument("--instrument", action="append", dest="instruments", help="filter tracks by instrument; repeatable")
    p.add_argument("--ensemble", action="append", dest="ensembles", help="CocoChorales ensemble filter; repeatable")
    p.add_argument("--max-tracks", type=int, default=None, help="cap selected tracks after filtering")
    p.add_argument("--per-stratum", type=int, default=None, help="CocoChorales sample cap per (ensemble,instrument)")
    p.add_argument("--seed", type=int, default=0, help="CocoChorales sampling seed")
    p.add_argument("--materialize", action="store_true", help="CocoChorales-only: extract selected stems before benchmarking")
    p.add_argument("--force-materialize", action="store_true", help="CocoChorales-only: overwrite materialized stems")
    p.add_argument("--root", default=None, help="CocoChorales dataset root override")
    p.add_argument("--f0-fps", type=float, default=F0_FPS_DEFAULT, help="CocoChorales f0 frame rate")
    # execution
    p.add_argument("--shard", type=parse_shard, default=None, help="run only shard i/N (e.g. 0/2)")
    p.add_argument("--tracks-per-task", type=int, default=0, help="tracks per worker task (0 = auto)")
    p.add_argument("--batch-size", type=int, default=0, help="chunks per fresh process pool (0 = workers*2)")
    p.add_argument("--watchdog", type=float, default=1200.0, help="seconds with no progress before in-flight chunks are re-queued")
    p.add_argument("--max-attempts", type=int, default=2, help="retries for a stuck/failed chunk before giving up")
    p.add_argument("--quiet-tracks", action="store_true", help="suppress legacy per-track worker lines")
    p.add_argument("--no-progress", action="store_true", help="disable the live per-track progress display")
    p.add_argument("--algorithm-verbose", action="store_true", help="let algorithms print their own diagnostics inside workers")
    p.add_argument("--dry-run", action="store_true", help="print the work plan and exit; detect nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_methods:
        print_method_matrix()
        return 0

    if args.datasets is None:
        args.datasets = ["test"] if args.benchmarker == "coco" else NoteBenchmarker.ETUDE_DATASETS

    methods = select_methods(args)

    opts: dict[str, Any] = {
        "root": args.root,
        "f0_fps": args.f0_fps,
        "onset_tolerance": args.onset_tolerance,
        "align": args.align,
        "latency_align": not args.no_latency_align,
        "trim_boundaries": not args.no_trim_boundaries,
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
    chunks = build_chunks(methods, tracks, workers=args.workers, tracks_per_task=args.tracks_per_task)
    batch_size = args.batch_size if args.batch_size > 0 else max(1, args.workers * 2)

    pb = make_benchmarker(args.benchmarker, opts)
    cached_input = 0
    if args.benchmarker == "coco":
        cached_input = sum(1 for _, _, wav, _ in tracks if pb.has_note_input_cache(wav))

    print(f"benchmarker: {args.benchmarker} (note detection)")
    print(f"datasets:   {', '.join(args.datasets)}")
    print(f"methods:    {len(methods)} | {', '.join(methods)}")
    if args.instruments:
        print(f"instruments:{' ' * 3}{', '.join(args.instruments)}")
    if args.ensembles:
        print(f"ensembles:  {', '.join(args.ensembles)}")
    print(f"align:      {args.align} (latency_align={not args.no_latency_align}, trim={not args.no_trim_boundaries})")
    print(f"shard:      {f'{args.shard[0]}/{args.shard[1]}' if args.shard else 'all'}")
    print(f"workers:    {args.workers}  (cpu_count={os.cpu_count()}, BLAS threads capped to 1/worker)")
    print(f"tracks:     {len(tracks)} corpus tracks | {len(methods) * len(tracks)} method/track pairs")
    if args.benchmarker == "coco":
        print(f"pitch input:{' ' * 1}{cached_input}/{len(tracks)} tracks have a cached pyin_smooth track")
    print(f"chunks:     {len(chunks)} total | batch={batch_size} | watchdog={fmt_dur(args.watchdog)}")

    if args.dry_run:
        print(f"\n[dry-run] would process {len(methods) * len(tracks)} method/track pair(s); no detection performed.")
        return 0
    if not tracks:
        print("\nno tracks selected.")
        if args.benchmarker == "coco":
            print("CocoChorales may need --materialize (extract selected stems first).")
        return 1
    if not chunks:
        print("\nnothing to do.")
        return 0

    print(f"\nstarting {len(methods) * len(tracks)} method/track pair(s) at "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')} ...\n", flush=True)
    started = time.perf_counter()
    rows, errors, skipped = process_chunks(
        run_chunk, chunks, workers=args.workers, batch_size=batch_size,
        watchdog=args.watchdog, max_attempts=args.max_attempts,
        kind=args.benchmarker, opts=opts, verbose=not args.quiet_tracks,
        progress=not args.no_progress,
    )
    total = time.perf_counter() - started

    if rows:
        write_raw_outputs(pb, rows, args.benchmarker)
        write_summary(pb, rows, methods)

    print(f"\ndone: {len(rows)} rows, {len(errors)} track errors, {len(skipped)} skipped method(s) in {fmt_dur(total)}")
    if skipped:
        print("skipped methods:")
        for method, reason in sorted(skipped.items()):
            first_line = reason.splitlines()[0] if reason else "dependency unavailable"
            print(f"  - {method}: {first_line}")
    if errors:
        print("failed tracks:")
        for method, dataset, track_id, _ in errors[:40]:
            print(f"  - {method} / {dataset} / {track_id}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
