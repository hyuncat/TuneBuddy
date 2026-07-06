from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import ruptures as rpt

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording

Run = Sequence[Pitch]


class RupturesDetector:
    """ruptures-based note segmentation over pre-split pitch runs.

    Sizing (min segment, penalty) and the retrace (median pitch per segment,
    merge across sub-threshold jumps, boundaries at the midpoint over dropped
    frames) mirror the production ``NoteDetector.detect_notes``, so the methods
    differ ONLY in how breakpoints are searched. Two params are benchmark-fixed
    (``JUMP``, ``MIN_NOTE_MS``) rather than derived from the live Config. Run
    splitting and transition exclusion are the caller's job
    (``NoteBenchmarker.pitch_runs``).

    The penalty is COST-AWARE: each cost model gets the cost drop that a
    pitch_thresh mean shift buys across two min-size segments, so every method
    applies the same "jumps under pitch_thresh are one note" decision boundary
    (l2's version is what the production detector uses). ``penalty_scale``
    multiplies that baseline for sweeps.
    """

    #: benchmark-fixed changepoint stride (frames). Independent of Config.h2 so
    #: the search granularity stays constant across configs/datasets.
    JUMP: int = 5

    #: minimum note duration (ms); shorter segments are dropped as over-
    #: segmentation. The external baselines all min-duration filter (crepe-notes
    #: min_duration, basic-pitch min_note_len), so change-point methods matched
    #: here to not be uniquely penalized for vibrato/portamento splits. Tuned on
    #: the coco violin set (FP -15%, accuracy +1pt); the peak was flat 90-150ms.
    MIN_NOTE_MS: float = 90.0

    #: cosine embedding scale: this many semitones map to a half-turn (antipodal
    #: vectors), so similarity falls monotonically out to two octaves and only
    #: aliases at four (a within-run jump that large doesn't occur).
    COSINE_HALF_TURN_ST: float = 24.0

    #: frames kept (evenly strided) for the rbf median-heuristic estimate
    RBF_GAMMA_MAX_FRAMES: int = 512

    def __init__(self, recording: Recording) -> None:
        self.config = recording.config
        self.min_size = max(1, round(
            recording.note_detector.MIN_NOTE_FACTOR
            * self.config.get_min_note_length(type="frames")
        ))
        self.jump = self.JUMP

    def detect_pelt(
        self, runs: Sequence[Run], model: str = "l2", penalty_scale: float = 1.0,
    ) -> NoteData:
        return self._detect(runs, lambda n: rpt.Pelt(
            model=model, min_size=self.min_size, jump=self.jump),
            cost=model, penalty_scale=penalty_scale)

    def detect_kernel(
        self, runs: Sequence[Run], kernel: str = "linear", penalty_scale: float = 1.0,
    ) -> NoteData:
        return self._detect(runs, lambda n: rpt.KernelCPD(
            kernel=kernel, min_size=self.min_size, jump=self.jump),
            cost=kernel, penalty_scale=penalty_scale)

    def detect_bottom_up(
        self, runs: Sequence[Run], model: str = "l2", penalty_scale: float = 1.0,
    ) -> NoteData:
        return self._detect(runs, lambda n: rpt.BottomUp(
            model=model, min_size=self.min_size, jump=self.jump),
            cost=model, penalty_scale=penalty_scale)

    def detect_windowed(
        self, runs: Sequence[Run], model: str = "l2", width: int | None = None,
        penalty_scale: float = 1.0,
    ) -> NoteData:
        return self._detect(runs, lambda n: rpt.Window(
            width=max(2, min(int(width or 5 * self.min_size), n - 1)),
            model=model, min_size=self.min_size, jump=self.jump),
            cost=model, penalty_scale=penalty_scale)

    # ------------------------------------------------------------------- core
    def _detect(
        self, runs: Sequence[Run], make_algo: Callable[[int], Any],
        cost: str = "l2", penalty_scale: float = 1.0,
    ) -> NoteData:
        note_data = NoteData()
        for run in runs:
            values = np.asarray([p.value for p in run], dtype=float)
            penalty = penalty_scale * self._penalty(cost, values)
            bkps = self._breakpoints(self._signal(values, cost), make_algo, penalty)
            self._write_segments(note_data, run, bkps)
        return self._prune_short(note_data)

    def _signal(self, values: np.ndarray, cost: str) -> np.ndarray:
        """Cosine similarity is degenerate on a raw 1-D pitch track (all values
        positive => every pair's similarity is 1), so for the cosine cost the
        pitch is embedded as a unit vector whose ANGLE is the pitch: similarity
        becomes cos(pitch difference / COSINE_HALF_TURN_ST * pi) -- an actual
        pitch-distance kernel. Everything else takes the raw column vector."""
        if cost != "cosine":
            return values.reshape(-1, 1)
        theta = values * (np.pi / self.COSINE_HALF_TURN_ST)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    def _penalty(self, cost: str, values: np.ndarray) -> float:
        """Cost drop a pitch_thresh mean shift buys across two min-size segments."""
        m, delta = self.min_size, self.config.pitch_thresh
        if cost == "l1":
            # split saves |value - median| for every frame on the far level
            return m * delta
        if cost == "rbf":
            # bounded kernel: a clean split saves at most ~m; a delta-sized shift
            # saves m*(1 - k(delta)) under ruptures' own median-heuristic
            # bandwidth (incl. its clip), estimated from this run's signal
            scaled = np.clip(self._rbf_gamma(values) * delta ** 2, 1e-2, 1e2)
            return m * float(-np.expm1(-scaled))
        if cost == "cosine":
            # unit-vector embedding (see _signal): cross-segment similarity for
            # a delta shift is cos(alpha * delta)
            return m * (1.0 - np.cos(np.pi * delta / self.COSINE_HALF_TURN_ST))
        # l2 / linear kernel (identical costs)
        return 0.5 * m * delta ** 2

    def _rbf_gamma(self, values: np.ndarray) -> float:
        """ruptures' median heuristic (gamma = 1/median pairwise sq. distance),
        on an evenly-strided subsample so long runs stay cheap."""
        sample = values[:: max(1, len(values) // self.RBF_GAMMA_MAX_FRAMES)]
        if len(sample) < 2:
            return 1.0
        sq_dists = (sample[:, None] - sample[None, :]) ** 2
        median = float(np.median(sq_dists[np.triu_indices_from(sq_dists, k=1)]))
        return 1.0 / median if median > 0 else 1.0

    def _prune_short(self, note_data: NoteData) -> NoteData:
        min_sec = self.MIN_NOTE_MS / 1000.0
        out = NoteData()
        for note in note_data.read(i=0, j=len(note_data.times)):
            if note.end_time - note.start_time >= min_sec:
                note.id = len(out.times)
                out.write_note(note)
        return out

    def _breakpoints(self, signal: np.ndarray, make_algo, penalty: float) -> list[int]:
        n = len(signal)
        if n < 2 * self.min_size:  # too short to split into > 1 note
            return [n]
        try:
            bkps = make_algo(n).fit(signal).predict(pen=penalty)
        except (rpt.exceptions.BadSegmentationParameters, ValueError):
            return [n]
        bkps = sorted({int(b) for b in bkps if 0 < int(b) <= n})
        return bkps if bkps and bkps[-1] == n else [*bkps, n]

    def _write_segments(self, note_data: NoteData, run: Run, bkps: list[int]) -> None:
        prev, first_in_run = 0, True
        for bkp in bkps:
            end = min(int(bkp), len(run))
            if end <= prev:
                continue
            midi_num = float(np.median([p.value for p in run[prev:end]]))
            start_time = self._boundary_time(run, prev)
            end_time = self._boundary_time(run, end)
            prev = end
            if end_time <= start_time:
                continue
            last = (
                note_data.read_note(i=len(note_data.times) - 1)
                if not first_in_run else None
            )
            if last is not None and abs(last.midi_num[0] - midi_num) < self.config.pitch_thresh:
                last.end_time = end_time  # too close in pitch to be two notes
            else:
                note_data.write_note(Note(
                    i=len(note_data.times),
                    start_time=start_time,
                    end_time=end_time,
                    midi_num=[midi_num],
                ))
            first_in_run = False

    @staticmethod
    def _boundary_time(run: Run, i: int) -> float:
        """midpoint over the dropped gap/slide, so adjacent notes meet mid-transition"""
        if i <= 0:
            return run[0].time
        if i >= len(run):
            return run[-1].time
        return 0.5 * (run[i - 1].time + run[i].time)
