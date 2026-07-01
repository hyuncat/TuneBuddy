#!/usr/bin/env python3
from __future__ import annotations

"""Run Attune note-detection comparison benchmarks.

Examples:
    python benchmarks/scripts/benchmark_notes.py --list-methods
    python benchmarks/scripts/benchmark_notes.py --dataset kayser --max-tracks 2
    python benchmarks/scripts/benchmark_notes.py --preset full --include-rbf --list-methods
    python benchmarks/scripts/benchmark_notes.py --dataset kayser --method pelt-l2__exclude-transitions__raw --no-write
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

from benchmarks.modules.note.NoteBenchmarker import NoteBenchmarker, NoteMethodConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="etude dataset to run; repeatable; defaults to all etude datasets",
    )
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--onset-tolerance", type=float, default=None)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="method label to include; repeatable; defaults to the selected preset",
    )
    parser.add_argument(
        "--preset",
        choices=["default", "full"],
        default="default",
        help="default is the practical local subset; full adds oracle/kernel/window rows",
    )
    parser.add_argument(
        "--include-rbf",
        action="store_true",
        help="include RBF/kernel rows in preset-selected methods",
    )
    parser.add_argument(
        "--include-mt3",
        action="store_true",
        help="include the MT3 placeholder row; it reports an error unless wired locally",
    )
    parser.add_argument(
        "--no-basic-pitch",
        action="store_true",
        help="drop the optional Basic Pitch external baseline from the default matrix",
    )
    parser.add_argument(
        "--no-tony-pyin",
        action="store_true",
        help="drop the Sonic Annotator / Tony pYIN external baseline from the default matrix",
    )
    parser.add_argument(
        "--align",
        choices=["identity", "resize"],
        default="identity",
        help="identity isolates note detection; resize mirrors the app pipeline",
    )
    parser.add_argument("--no-latency-align", action="store_true")
    parser.add_argument("--no-trim-boundaries", action="store_true")
    parser.add_argument("--no-note-cache", action="store_true")
    parser.add_argument(
        "--write",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write result CSV files; enabled by default",
    )
    parser.add_argument("--list-methods", action="store_true")
    return parser.parse_args()


def selected_methods(args: argparse.Namespace, benchmarker: NoteBenchmarker) -> dict[str, NoteMethodConfig]:
    if not args.methods:
        methods = dict(
            benchmarker.ALL_NOTE_METHODS
            if args.preset == "full"
            else benchmarker.NOTE_METHODS
        )
        if not args.include_rbf:
            methods = {
                label: config
                for label, config in methods.items()
                if not _is_rbf_method(label, config)
            }
        if args.include_mt3:
            methods.update(benchmarker.OPTIONAL_NOTE_METHODS)
        if args.no_basic_pitch:
            methods.pop("basic-pitch", None)
        if args.no_tony_pyin:
            methods.pop("tony-pyin", None)
        return methods

    methods = dict(benchmarker.ALL_NOTE_METHODS)
    methods.update(benchmarker.OPTIONAL_NOTE_METHODS)
    if args.no_basic_pitch:
        methods.pop("basic-pitch", None)
    if args.no_tony_pyin:
        methods.pop("tony-pyin", None)
    unknown = [label for label in args.methods if label not in methods]
    if unknown:
        known = "\n  ".join(sorted(methods))
        raise SystemExit(
            "Unknown note method label(s): "
            + ", ".join(unknown)
            + "\nKnown labels:\n  "
            + known
        )
    return {label: methods[label] for label in args.methods}


def _is_rbf_method(label: str, config: NoteMethodConfig) -> bool:
    return (
        "rbf" in label
        or config.get("model") == "rbf"
        or config.get("cost") == "rbf"
        or config.get("ruptures_algorithm") == "kernelcpd"
    )


def main() -> int:
    args = parse_args()
    benchmarker = NoteBenchmarker()
    methods = selected_methods(args, benchmarker)

    if args.list_methods:
        for label, config in methods.items():
            print(
                f"{label}\t"
                f"family={config.get('method')} "
                f"algorithm={config.get('ruptures_algorithm')} "
                f"model={config.get('model')} "
                f"transition_excluding={config.get('exclude_transitions')} "
                f"refined_with_onsets={config.get('refined_with_onsets', False)}"
            )
        return 0

    datasets = args.datasets or benchmarker.ETUDE_DATASETS
    for dataset in datasets:
        df = benchmarker.bench_note_dataset(
            dataset,
            max_tracks=args.max_tracks,
            onset_tolerance=args.onset_tolerance,
            methods=methods,
            align=args.align,
            latency_align=not args.no_latency_align,
            trim_boundaries=not args.no_trim_boundaries,
            write_note_cache=not args.no_note_cache,
            write=args.write,
        )
        print(f"\n=== {dataset}: {len(df)} rows ===")
        if not df.empty:
            columns = [
                "method",
                "Precision",
                "Recall",
                "F-measure",
                "Estimated Notes",
                "refined_with_onsets",
                "transition_excluding",
                "error",
            ]
            print(df[[c for c in columns if c in df.columns]].to_string(index=False))
        if args.write:
            print(f"wrote {benchmarker.result_csv_path('note', dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
