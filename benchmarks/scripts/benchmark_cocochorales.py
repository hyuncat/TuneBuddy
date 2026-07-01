#!/usr/bin/env python3
from __future__ import annotations

"""Run CocoChorales benchmarks.

Examples:
    python benchmarks/scripts/benchmark_cocochorales.py --build-manifest --dry-run
    python benchmarks/scripts/benchmark_cocochorales.py --stage pitch --instrument violin --per-stratum 5 --materialize --dry-run
    python benchmarks/scripts/benchmark_cocochorales.py --stage pitch --instrument violin --shard 1.tar.bz2 --per-stratum 5 --materialize --write
"""

import argparse
import sys
from pathlib import Path

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

from benchmarks.modules.CocoChoralesBenchmarker import (  # noqa: E402
    F0_FPS_DEFAULT,
    CocoChoralesBenchmarker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["pitch"], default="pitch", help="benchmark stage to run")
    parser.add_argument("--root", default=None, help="CocoChorales dataset root")
    parser.add_argument("--split", default="test", choices=["test", "valid", "train", "all"])
    parser.add_argument("--f0-fps", type=float, default=F0_FPS_DEFAULT)
    parser.add_argument("--build-manifest", action="store_true", help="scan retained shards and write the manifest")
    parser.add_argument("--rebuild-manifest", action="store_true", help="ignore an existing manifest")
    parser.add_argument("--prune-f0", action="store_true", help="delete f0 pickles not referenced by the manifest")
    parser.add_argument("--materialize", action="store_true", help="extract selected stem wav/midi files only")
    parser.add_argument("--force-materialize", action="store_true", help="overwrite existing materialized files")
    parser.add_argument("--per-stratum", type=int, default=None, help="sample up to N stems per (ensemble,instrument)")
    parser.add_argument("--max-tracks", type=int, default=None, help="cap selected stems after sampling/filtering")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble", action="append", dest="ensembles", help="filter by ensemble; repeatable")
    parser.add_argument("--instrument", action="append", dest="instruments", help="filter by instrument; repeatable")
    parser.add_argument("--shard", action="append", dest="shards", help="filter by shard path or basename, e.g. 1.tar.bz2")
    parser.add_argument("--no-smooth", action="store_true", help="skip HMM smoothing and pitch cache writes")
    parser.add_argument("--write", action="store_true", help="write result CSV")
    parser.add_argument("--list", action="store_true", help="print the selected work plan")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and do not run detection")
    parser.add_argument("--probe", action="store_true", help="inspect dataset state")
    parser.add_argument("--probe-detect", action="store_true", help="run one materialized stem end to end")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pb = CocoChoralesBenchmarker(root=args.root, f0_fps=args.f0_fps)

    if args.probe or args.probe_detect:
        pb.probe(run_detect=args.probe_detect)
        return 0

    if args.build_manifest or args.rebuild_manifest:
        records = pb.load_or_build_manifest(args.split, rebuild=True)
        print(f"wrote {len(records)} manifest rows -> {pb.manifest_path(args.split)}")

    if args.prune_f0:
        deleted = pb.prune_f0_to_manifest(args.split, dry_run=args.dry_run)
        action = "would delete" if args.dry_run else "deleted"
        print(f"{action} {deleted} f0 pickle(s) not referenced by the manifest")

    records = pb.select_records(
        split=args.split,
        per_stratum=args.per_stratum,
        seed=args.seed,
        max_tracks=args.max_tracks,
        ensembles=args.ensembles,
        instruments=args.instruments,
        shards=args.shards,
        rebuild_manifest=False,
    )

    if args.materialize and not args.dry_run:
        written = pb.materialize_records(records, force=args.force_materialize)
        print(f"materialized {len(written)} file(s)")

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
        if args.materialize:
            missing = sum(1 for record in records if pb.local_wav_path(record) is None)
            print(f"\nmaterialize needed for {missing} selected stem(s)")
        return 0

    if args.stage == "pitch":
        pb.bench_records(records, smooth=not args.no_smooth, write=args.write)
        return 0

    raise ValueError(f"unsupported stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
