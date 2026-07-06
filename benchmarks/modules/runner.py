from __future__ import annotations

"""Shared parallel benchmark execution machinery.

Extracted verbatim from ``benchmarks/scripts/benchmark_pitch.py`` so the pitch
and note runners share one implementation of:

  - ``ProgressRenderer`` -- live ``[i/N] <method>: <title> [/]`` lines,
  - ``process_chunks`` -- fresh ``ProcessPoolExecutor`` per batch (no in-place
    worker recycling), a no-progress watchdog that re-queues stuck chunks, and
    BrokenProcessPool recovery,
  - small helpers (``fmt_dur``, ``parse_shard``, ``_force_teardown``).

The chunk-executing function is passed in by the calling script and MUST be a
top-level function of that script: with the ``spawn`` start method the workers
unpickle it by re-importing the script module, which is what re-applies the
BLAS/OpenMP thread caps set at the top of the script before numpy loads.

Contract for ``run_chunk(chunk, kind, opts, verbose, progress_queue, total)``:
returns ``(status, method, rows, errors, duration_sec, skip_msg)`` where status
is one of ``"ok" | "skip" | "err"``, ``rows`` is a list of result dicts and
``errors`` a list of ``(method, dataset, track_id, traceback_text)`` tuples.
"""

import argparse
import multiprocessing
import os
import queue as queue_mod
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any

TrackItem = tuple[str, str, str, str]  # dataset, track_id, wav_path, annot/midi path
WorkChunk = tuple[str, int, tuple[TrackItem, ...]]  # method, first progress index, tracks
ChunkResult = tuple[str, str, list[dict[str, Any]], list[tuple[str, str, str, str]], float, str | None]
RunChunkFn = Callable[..., ChunkResult]


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
    # Only tear down this executor's workers.  multiprocessing.active_children()
    # also includes the Manager process that backs the progress queue; killing it
    # makes the next batch's spawned workers fail while unpickling that queue
    # proxy with ConnectionRefusedError.
    processes = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for child in processes:
        if child.is_alive():
            child.terminate()
    for child in processes:
        child.join(timeout=10)
        if child.is_alive():
            child.kill()


class ProgressRenderer:
    SPINNER = "|/-\\"

    def __init__(self, enabled: bool = True, live: bool = False) -> None:
        self.enabled = enabled
        self.is_tty = enabled and live and sys.stdout.isatty()
        self.active: dict[int, dict[str, Any]] = {}
        self.live_lines = 0
        self.last_render = 0.0
        self.transport_failed = False

    def close(self) -> None:
        self._clear_live()
        sys.stdout.flush()

    def drain(self, progress_queue: Any | None) -> bool:
        if not self.enabled or progress_queue is None:
            return False
        changed = False
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue_mod.Empty:
                break
            except (BrokenPipeError, EOFError, OSError):
                self.transport_failed = True
                self.enabled = False
                self._clear_live()
                return changed
            self._handle(event)
            changed = True
        if changed or (self.is_tty and self.active):
            self.render()
        return changed

    def _handle(self, event: dict[str, Any]) -> None:
        index = int(event.get("index", 0) or 0)
        if event.get("event") == "start":
            self.active[index] = {
                "index": index,
                "total": int(event.get("total", 0) or 0),
                "model": str(event.get("model", "")),
                "track_id": str(event.get("track_id", "")),
                "started": time.perf_counter(),
            }
            return

        if event.get("event") == "done":
            item = self.active.pop(index, None) or {
                "index": index,
                "total": int(event.get("total", 0) or 0),
                "model": str(event.get("model", "")),
                "track_id": str(event.get("track_id", "")),
            }
            ok = bool(event.get("ok"))
            self._clear_live()
            print(self._line(item, "✓" if ok else "X"), flush=True)

    def render(self) -> None:
        if not self.is_tty:
            return
        now = time.perf_counter()
        if now - self.last_render < 0.08 and self.live_lines:
            return
        self._clear_live()
        lines = [self._line(item, self._spinner(now)) for item in self.active.values()]
        if lines:
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
        self.live_lines = len(lines)
        self.last_render = now

    def _clear_live(self) -> None:
        if not self.is_tty or self.live_lines <= 0:
            self.live_lines = 0
            return
        for _ in range(self.live_lines):
            sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()
        self.live_lines = 0

    def _spinner(self, now: float) -> str:
        return self.SPINNER[int(now * 10) % len(self.SPINNER)]

    def _line(self, item: dict[str, Any], status: str) -> str:
        total = int(item.get("total", 0) or 0)
        index = int(item.get("index", 0) or 0)
        model = str(item.get("model", ""))
        title = str(item.get("track_id", ""))
        line = f"[{index}/{total}] {model}: {title} [{status}]"
        width = shutil.get_terminal_size((120, 20)).columns
        if width > 20 and len(line) > width:
            line = line[: width - 1]
        return line


def process_chunks(
    run_chunk: RunChunkFn,
    chunks: Sequence[WorkChunk],
    workers: int,
    batch_size: int,
    watchdog: float,
    max_attempts: int,
    kind: str,
    opts: dict[str, Any],
    verbose: bool,
    progress: bool,
    progress_live: bool = True,
    on_result: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, str]], dict[str, str]]:
    """``on_result`` (if given) is called on the MAIN process with each finished
    chunk's rows, so callers can persist results incrementally instead of only
    after the whole run."""
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str, str]] = []
    skipped: dict[str, str] = {}
    attempts: dict[str, int] = {}
    queue = list(chunks)
    total_tracks = sum(len(items) for _, _, items in chunks)
    completed_tracks = 0
    started = time.perf_counter()
    manager = multiprocessing.Manager() if progress else None
    progress_queue = manager.Queue() if manager is not None else None
    renderer = ProgressRenderer(enabled=progress, live=progress_live)

    def chunk_key(chunk: WorkChunk) -> str:
        model, _, items = chunk
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

    try:
        while queue:
            batch, queue = queue[:batch_size], queue[batch_size:]
            runnable: list[WorkChunk] = []
            for chunk in batch:
                key = chunk_key(chunk)
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] > max_attempts:
                    model, _, items = chunk
                    tb = f"gave up after {max_attempts} attempts (kept hanging/failing)"
                    for dataset, track_id, *_ in items:
                        errors.append((model, dataset, track_id, tb))
                    renderer.close()
                    log("GIVEUP", model, len(items), 0.0)
                else:
                    runnable.append(chunk)
            if not runnable:
                continue

            ex = ProcessPoolExecutor(max_workers=workers)
            futs = {
                ex.submit(
                    run_chunk,
                    chunk,
                    kind,
                    opts,
                    verbose,
                    progress_queue,
                    total_tracks,
                ): chunk
                for chunk in runnable
            }
            try:
                pending = set(futs)
                last_activity = time.perf_counter()
                while pending:
                    done_set, pending = wait(
                        pending,
                        timeout=0.1 if progress_queue is not None else watchdog,
                        return_when=FIRST_COMPLETED,
                    )
                    if renderer.drain(progress_queue):
                        last_activity = time.perf_counter()
                    if renderer.transport_failed:
                        progress_queue = None
                    if not done_set:
                        if time.perf_counter() - last_activity < watchdog:
                            continue
                        stuck = [futs[fut] for fut in pending]
                        renderer.close()
                        watchdog_reason = "no track/chunk progress" if progress else "no chunk finished"
                        print(
                            f"\n!! watchdog: {watchdog_reason} in {fmt_dur(watchdog)} -- "
                            f"re-queueing {len(stuck)} in-flight chunk(s).",
                            file=sys.stderr,
                            flush=True,
                        )
                        for model, _, items in stuck:
                            first = items[0][1] if items else "empty"
                            print(f"   stuck: {model} / {first} ({len(items)} tracks)", file=sys.stderr, flush=True)
                        queue.extend(stuck)
                        break

                    last_activity = time.perf_counter()
                    broke = False
                    for fut in done_set:
                        chunk = futs[fut]
                        model, _, items = chunk
                        try:
                            status, result_model, chunk_rows, chunk_errors, dur, skip_msg = fut.result()
                        except BrokenProcessPool:
                            broke = True
                            break

                        if status == "skip":
                            rows.extend(chunk_rows)
                            errors.extend(chunk_errors)
                            if on_result is not None and chunk_rows:
                                on_result(chunk_rows)
                            skipped[result_model] = skip_msg or "dependency unavailable"
                            renderer.close()
                            log("SKIP", result_model, len(items), dur)
                            print(f"[{result_model}] SKIPPED -- {skipped[result_model]}", flush=True)
                            continue

                        rows.extend(chunk_rows)
                        errors.extend(chunk_errors)
                        if on_result is not None and chunk_rows:
                            on_result(chunk_rows)
                        if status == "ok":
                            if not progress:
                                log("OK", model, len(chunk_rows) + len(chunk_errors), dur)
                        else:
                            renderer.close()
                            log("ERR", model, len(items), dur)
                            for _, _, _, tb in chunk_errors:
                                print(tb.rstrip(), file=sys.stderr, flush=True)

                    if broke:
                        stuck = [chunk, *[futs[fut] for fut in pending]]
                        renderer.close()
                        print(
                            f"\n!! pool broke (a worker died) -- re-queueing {len(stuck)} unfinished chunk(s).",
                            file=sys.stderr,
                            flush=True,
                        )
                        queue.extend(stuck)
                        break
            finally:
                renderer.drain(progress_queue)
                renderer.close()
                _force_teardown(ex)
    finally:
        renderer.close()
        if manager is not None:
            try:
                manager.shutdown()
            except (BrokenPipeError, EOFError, OSError):
                pass

    return rows, errors, skipped


__all__ = [
    "ChunkResult",
    "ProgressRenderer",
    "TrackItem",
    "WorkChunk",
    "fmt_dur",
    "parse_shard",
    "process_chunks",
    "_force_teardown",
]
