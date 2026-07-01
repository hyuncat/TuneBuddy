from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np


def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app.py").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    raise RuntimeError("could not locate Attune repo root")


_BOOTSTRAP_ROOT = _bootstrap_repo_root()
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from app_logic.NoteData import Note, NoteData  # noqa: E402
from benchmarks.modules.pitch.PitchBenchmarker import PathLike, PitchBenchmarker  # noqa: E402
from benchmarks.paths import REPO_ROOT, ensure_repo_on_path  # noqa: E402

ensure_repo_on_path()
ROOT = REPO_ROOT
MistakeType = Literal["substitution", "deletion", "insertion"]
_SOURCE_SCORE_ID_UNSET = object()


class TruthEvent(TypedDict, total=False):
    type: MistakeType
    score_note_id: int
    time: float


class MistakeInjector:
    """
    Interface for injecting mistakes into scores, symbolically or re-synthesized through audio.
    Mirrors the PolyTune methods 'augment_mistakes' and 'add_screwups' sampling from 
    'lambda_occur ~ U(0.1, 0.4)' and applying +/-50% count jitter to select error indices.
    
    Rk: When synthesized, the first and last notes are artificially longer due to reverb.
    We hard-correct for this by truncating any excess for those two edge notes only, and likewise
    we also omit injecting further mistakes into those.
    """

    POLYTUNE_LAMBDA_RANGE = (0.1, 0.4)
    POLYTUNE_PITCH_STD = 1.0
    POLYTUNE_DURATION_MEAN = 1.0
    POLYTUNE_DURATION_STD = 0.02
    POLYTUNE_TIMING_MEAN_MS = 0.0
    POLYTUNE_TIMING_STD_MS = 300.0
    POLYTUNE_SCREWUP_TYPES = tuple(range(16))

    def __init__(
        self,
        lambda_range: tuple[float, float] = POLYTUNE_LAMBDA_RANGE,
        pitch_offset_std: float = POLYTUNE_PITCH_STD,
        duration_mean: float = POLYTUNE_DURATION_MEAN,
        duration_std: float = POLYTUNE_DURATION_STD,
        timing_mean_ms: float = POLYTUNE_TIMING_MEAN_MS,
        timing_std_ms: float = POLYTUNE_TIMING_STD_MS,
        allow_overlap: bool = True,
        protect_boundary_notes: bool = True,
        mistake_rate: float | None = None,
        weights: Sequence[float] | None = None,
        screwup_type_weights: Sequence[float] | None = None,
        out_dir: PathLike | None = None,
    ) -> None:
        if mistake_rate is not None:
            lambda_range = (float(mistake_rate), float(mistake_rate))
        self.lambda_range = (float(lambda_range[0]), float(lambda_range[1]))

        if screwup_type_weights is not None:
            if len(screwup_type_weights) != len(self.POLYTUNE_SCREWUP_TYPES):
                raise ValueError("screwup_type_weights must have length 16")
            self.screwup_type_weights = self._normalize_weights(screwup_type_weights)
        elif weights is not None:
            # Compatibility for old notebooks using (substitution, deletion,
            # insertion). Timing-only codes keep zero probability in this mode.
            if len(weights) != 3:
                raise ValueError("weights must have length 3")
            probs = np.zeros(16, dtype=float)
            probs[1] = float(weights[0])
            probs[0] = float(weights[1])
            probs[3] = float(weights[2])
            self.screwup_type_weights = self._normalize_weights(probs)
        else:
            self.screwup_type_weights = tuple([1.0 / 16.0] * 16)

        self.pitch_offset_std = float(pitch_offset_std)
        self.duration_mean = float(duration_mean)
        self.duration_std = float(duration_std)
        self.timing_mean_ms = float(timing_mean_ms)
        self.timing_std_ms = float(timing_std_ms)
        self.allow_overlap = allow_overlap
        self.protect_boundary_notes = protect_boundary_notes
        self.out_dir = (
            Path(out_dir)
            if out_dir is not None
            else ROOT / "benchmarks" / "datasets" / "mistake-db"
        )
        self.last_metadata: dict[str, Any] = {}

    # @staticmethod
    def _normalize_weights(self, weights: Sequence[float]) -> tuple[float, ...]:
        arr = np.asarray(list(weights), dtype=float)
        if np.any(arr < 0):
            raise ValueError("weights must be non-negative")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        return tuple(float(x / total) for x in arr)

    @staticmethod
    def _write_note_unique(note_data: NoteData, note: Note) -> None:
        while note.start_time in note_data.data:
            note.start_time += 1e-6
            note.end_time += 1e-6
        note_data.write_note(note)

    @staticmethod
    def _copy_note(
        note: Note,
        note_id: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        midi_num: Sequence[float] | None = None,
        source_score_id: Any = _SOURCE_SCORE_ID_UNSET,
    ) -> Note:
        out = Note(
            i=note.id if note_id is None else note_id,
            start_time=float(note.start_time if start_time is None else start_time),
            end_time=float(note.end_time if end_time is None else end_time),
            midi_num=list(note.midi_num if midi_num is None else midi_num),
            velocity=note.velocity,
            instrument=note.instrument,
        )
        if source_score_id is _SOURCE_SCORE_ID_UNSET:
            source_score_id = getattr(note, "source_score_id", note.id)
        out.source_score_id = source_score_id
        return out

    def _sample_pitch_delta(self, random_generator: np.random.Generator) -> int:
        """GitHub generator samples int(normal(0, stdev_pitch_delta))."""
        for _ in range(50):
            delta = int(random_generator.normal(loc=0.0, scale=self.pitch_offset_std))
            if delta <= -1 or delta >= 1:
                return delta
        return int(random_generator.choice([-1, 1]))

    def _sample_timing_delta(self, random_generator: np.random.Generator) -> float:
        return float(
            random_generator.normal(
                loc=self.timing_mean_ms / 1000.0,
                scale=self.timing_std_ms / 1000.0,
            )
        )

    def _sample_duration(self, duration: float, random_generator: np.random.Generator) -> float:
        variance = self.duration_std ** 2
        if duration <= 0 or variance <= 0 or self.duration_mean <= 0:
            return max(1e-6, duration)
        shape = self.duration_mean ** 2 / variance
        scale = variance / self.duration_mean
        duration_var = random_generator.gamma(shape, scale)
        out = max(0.5 * duration, duration * duration_var)
        return float(min(out, 2.0 * duration))

    @staticmethod
    def _clamp_midi(midi_num: float) -> int:
        return int(np.clip(round(midi_num), 0, 127))

    @staticmethod
    def _protect_span(
        notes: Sequence[Note],
        note_index: int,
        start: float,
        end: float,
    ) -> tuple[float, float]:
        """Keep generated events off the first and last notes.

        Interior mistakes remain free to overlap other interior notes, but the
        adjacent boundary windows are preserved so synth attack/release artifacts
        do not contaminate the first/last note benchmark anchors.
        """
        if not notes or note_index <= 0 or note_index >= len(notes) - 1:
            return start, end

        duration = max(1e-6, end - start)
        if note_index == 1:
            min_start = notes[0].end_time
            if start < min_start:
                start = min_start
                end = start + duration
        if note_index == len(notes) - 2:
            max_end = notes[-1].start_time
            if end > max_end:
                end = max_end
                start = end - duration
        if start < 0:
            end -= start
            start = 0.0
        if end <= start:
            end = start + 1e-6
        return float(start), float(end)

    def _choose_error_indices(
        self,
        note_count: int,
        random_generator: np.random.Generator,
    ) -> tuple[float, set[int]]:
        lambda_value = float(
            random_generator.uniform(self.lambda_range[0], self.lambda_range[1])
        )
        if self.protect_boundary_notes and note_count > 2:
            candidates = np.arange(1, note_count - 1)
        elif self.protect_boundary_notes:
            candidates = np.asarray([], dtype=int)
        else:
            candidates = np.arange(note_count)
        if candidates.size == 0:
            return lambda_value, set()
        base_size = min(int(np.ceil(lambda_value * candidates.size)), candidates.size)
        half_range = base_size // 2
        size_adjustment = int(random_generator.integers(-half_range, half_range + 1))
        size = max(0, min(candidates.size, base_size + size_adjustment))
        selected = random_generator.choice(candidates, size=size, replace=False)
        return lambda_value, {int(i) for i in selected}

    def _timed_span(
        self,
        notes: Sequence[Note],
        note_index: int,
        random_generator: np.random.Generator,
    ) -> tuple[float, float]:
        note = notes[note_index]
        original_start = float(note.start_time)
        original_end = float(note.end_time)
        original_duration = max(1e-6, original_end - original_start)
        duration = self._sample_duration(original_duration, random_generator)
        start = original_start + self._sample_timing_delta(random_generator)

        if not self.allow_overlap:
            if 0 < note_index < len(notes) - 1:
                prev_start = notes[note_index - 1].start_time
                next_start = notes[note_index + 1].start_time
                attempts = 0
                while (
                    (start - original_start >= next_start - original_start)
                    or (start - original_start <= prev_start - original_start)
                ) and attempts < 100:
                    start = original_start + self._sample_timing_delta(random_generator)
                    attempts += 1
            elif note_index == 0 and len(notes) > 1:
                next_start = notes[1].start_time
                attempts = 0
                while start - original_start >= next_start - original_start and attempts < 100:
                    start = original_start + self._sample_timing_delta(random_generator)
                    attempts += 1
            elif note_index == len(notes) - 1 and len(notes) > 1:
                prev_start = notes[-2].start_time
                attempts = 0
                while start - original_start <= prev_start - original_start and attempts < 100:
                    start = original_start + self._sample_timing_delta(random_generator)
                    attempts += 1

        end = start + duration
        if not self.allow_overlap:
            if note_index > 0:
                start = max(start, notes[note_index - 1].end_time)
            if note_index < len(notes) - 1:
                end = min(end, notes[note_index + 1].start_time)
            if end <= start:
                end = start + 1e-6

        start, end = self._protect_span(notes, note_index, start, end)
        # PolyTune applies onset/duration jitter as performance realism only: the
        # shifted note still lands in midi_correct_notes (never labeled a timing
        # mistake), and their evaluator scores onset+pitch with offset_ratio=None.
        # So we emit NO timing truth here -- the jittered span is all we return.
        return start, end

    def _extra_note_start(
        self,
        notes: Sequence[Note],
        note_index: int,
        note: Note,
        random_generator: np.random.Generator,
    ) -> float:
        if note_index < len(notes) - 1:
            next_start = notes[note_index + 1].start_time
        else:
            next_start = note.end_time
        next_start = max(float(note.start_time), float(next_start))
        if next_start <= note.start_time:
            return float(note.start_time)
        return float(random_generator.uniform(low=note.start_time, high=next_start))

    def inject(
        self,
        reference_notes: NoteData,
        random_generator: np.random.Generator | None = None,
    ) -> tuple[NoteData, list[TruthEvent]]:
        random_generator = (
            random_generator
            if random_generator is not None
            else np.random.default_rng()
        )
        notes = reference_notes.read(i=0, j=len(reference_notes.times))
        lambda_value, error_indices = self._choose_error_indices(
            len(notes),
            random_generator,
        )
        selected_metadata: list[dict[str, Any]] = []
        out_notes: list[Note] = []
        performance_notes = NoteData()
        truth: list[TruthEvent] = []
        for note_index, note in enumerate(notes):
            if note_index not in error_indices:
                out_notes.append(self._copy_note(note))
                continue

            screwup_type = int(
                random_generator.choice(
                    self.POLYTUNE_SCREWUP_TYPES,
                    p=self.screwup_type_weights,
                )
            )
            selected_metadata.append(
                {
                    "score_note_index": int(note_index),
                    "score_note_id": int(note.id),
                    "screwup_type": screwup_type,
                }
            )

            if screwup_type == 0:
                truth.append(
                    dict(
                        type="deletion",
                        score_note_id=note.id,
                        time=note.start_time,
                    )
                )
                continue

            start_time, end_time = self._timed_span(
                notes,
                note_index,
                random_generator,
            )

            if screwup_type == 1:
                pitch_delta = self._sample_pitch_delta(random_generator)
                new_pitch = self._clamp_midi(note.midi_num[0] + pitch_delta)
                out_notes.append(
                    self._copy_note(
                        note,
                        start_time=start_time,
                        end_time=end_time,
                        midi_num=[new_pitch],
                    )
                )
                truth.append(
                    dict(
                        type="substitution",
                        score_note_id=note.id,
                        time=note.start_time,
                    )
                )
                continue

            if screwup_type == 2:
                pitch_delta = self._sample_pitch_delta(random_generator)
                new_pitch = self._clamp_midi(note.midi_num[0] + pitch_delta)
                original_duration = max(1e-6, note.end_time - note.start_time)
                initial_duration = original_duration / 8.0 + random_generator.uniform(
                    low=-original_duration / 32.0,
                    high=(original_duration / 8.0) * 3.0,
                )
                initial_duration = max(1e-6, min(initial_duration, end_time - start_time))
                wrong_end = start_time + initial_duration
                out_notes.append(
                    self._copy_note(
                        note,
                        start_time=start_time,
                        end_time=wrong_end,
                        midi_num=[new_pitch],
                        source_score_id=None,
                    )
                )
                if wrong_end < end_time:
                    out_notes.append(
                        self._copy_note(
                            note,
                            start_time=wrong_end,
                            end_time=end_time,
                        )
                    )
                truth.append(dict(type="insertion", time=start_time))
                continue

            if screwup_type == 3:
                timed_note = self._copy_note(
                    note,
                    start_time=start_time,
                    end_time=end_time,
                )
                out_notes.append(timed_note)
                extra_duration = self._sample_duration(
                    max(1e-6, timed_note.end_time - timed_note.start_time),
                    random_generator,
                )
                extra_start = self._extra_note_start(
                    notes,
                    note_index,
                    timed_note,
                    random_generator,
                )
                extra_start, extra_end = self._protect_span(
                    notes,
                    note_index,
                    extra_start,
                    extra_start + extra_duration,
                )
                pitch_delta = self._sample_pitch_delta(random_generator)
                inserted_pitch = self._clamp_midi(timed_note.midi_num[0] + pitch_delta)
                out_notes.append(
                    self._copy_note(
                        timed_note,
                        start_time=extra_start,
                        end_time=extra_end,
                        midi_num=[inserted_pitch],
                        source_score_id=None,
                    )
                )
                truth.append(dict(type="insertion", time=extra_start))
                continue

            # GitHub codes 4..15 are timing/duration-only placeholders: PolyTune
            # applies the jittered span as realism but labels no mistake, so the
            # note lands in the "correct" bucket and we emit no truth event.
            out_notes.append(
                self._copy_note(note, start_time=start_time, end_time=end_time)
            )

        for output_index, note in enumerate(
            sorted(out_notes, key=lambda n: (n.start_time, n.end_time, n.midi_num[0]))
        ):
            note.id = output_index
            self._write_note_unique(performance_notes, note)

        self.last_metadata = {
            "method": "polytune-github-add_screwups",
            "lambda": lambda_value,
            "lambda_range": list(self.lambda_range),
            "screwup_types": list(self.POLYTUNE_SCREWUP_TYPES),
            "screwup_type_weights": list(self.screwup_type_weights),
            "pitch_offset_std": self.pitch_offset_std,
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
            "timing_mean_ms": self.timing_mean_ms,
            "timing_std_ms": self.timing_std_ms,
            "allow_overlap": self.allow_overlap,
            "protect_boundary_notes": self.protect_boundary_notes,
            "selected_note_indices": sorted(error_indices),
            "selected": selected_metadata,
        }
        return performance_notes, truth

    def synth(self, performance_notes: NoteData, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        midi_path = self.out_dir / f"{name}.mid"
        PitchBenchmarker.notedata_to_pm(performance_notes).write(str(midi_path))
        return PitchBenchmarker().synth_midi(midi_path, out_dir=self.out_dir, force=True)
