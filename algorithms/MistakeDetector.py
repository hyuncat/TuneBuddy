from __future__ import annotations

import time
from typing import TYPE_CHECKING, Sequence

import numpy as np

from algorithms.Config import Config
from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import Note, NoteData

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording


class MistakeDetector:
    # index constants for dp backpointer
    DELETION = 0
    SUBSTITUTION = 1
    INSERTION = 2

    def __init__(self, recording: Recording=None, config: Config=None, verbose: bool=False):
        if isinstance(recording, Config) and config is None:
            config = recording
            recording = None
        self.recording = recording
        self.config = recording.config if recording else config
        self.verbose = verbose

    def update_config(self, config: Config):
        self.config = config

    def detect_mistakes(self, user_notes: NoteData, score_notes: NoteData, verbose: bool=None) -> Alignment:
        """Align played notes to the score and build every resulting mistake."""
        if verbose is None:
            verbose = self.verbose
        start = time.perf_counter()

        user_string = [
            user_notes.data[t]
            for t in user_notes.times
            if user_notes.data[t].midi_num[0] != -1
        ]
        midi_string = [score_notes.data[t] for t in score_notes.times]

        if verbose:
            print(
                f"[MistakeDetector] aligning {len(user_string)} user note(s) "
                f"to {len(midi_string)} score note(s)",
                flush=True,
            )

        alignment = self.get_string_edit_alignment(user_string, midi_string)

        if verbose:
            print(
                f"[MistakeDetector] done: "
                f"{len(alignment.pitch_mistakes)} pitch mistake(s), "
                f"{len(alignment.timing_mistakes)} timing mistake(s), "
                f"{len(alignment.pairs)} aligned pair(s) in "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )
        return alignment

    # phase 1: dp string edit
    def string_edit(self, user_string: Sequence[Note], midi_string: Sequence[Note]) -> np.ndarray:
        """Return the cheapest edit operation at every dynamic-programming cell."""
        user_count = len(user_string)
        score_count = len(midi_string)
        backpointer = np.zeros(
            (score_count + 1, user_count + 1),
            dtype=np.uint8,
        )

        # get all insertion and deletion costs in advance for filling in DP matrix
        insertion_costs = np.fromiter(
            (self.get_insertion_cost(note) for note in user_string),
            dtype=np.float64,
            count=user_count,
        )
        deletion_costs = np.fromiter(
            (self.get_deletion_cost(note) for note in midi_string),
            dtype=np.float64,
            count=score_count,
        )

        previous = np.zeros(user_count + 1, dtype=np.float64)
        if user_count:
            previous[1:] = np.cumsum(insertion_costs)
            backpointer[0, 1:] = self.INSERTION

        # fill in DP matrix row by row
        for score_index, score_note in enumerate(midi_string, start=1):
            current = np.empty(user_count + 1, dtype=np.float64)
            deletion_cost = deletion_costs[score_index - 1]
            current[0] = previous[0] + deletion_cost
            backpointer[score_index, 0] = self.DELETION

            for user_index, user_note in enumerate(user_string, start=1):
                delete_total = previous[user_index] + deletion_cost
                substitute_total = (
                    previous[user_index - 1]
                    + self.get_substitution_cost(user_note, score_note)
                )
                insert_total = (
                    current[user_index - 1]
                    + insertion_costs[user_index - 1]
                )
                # preserve numpy.argmin's deletion/substitution/insertion tie order
                if delete_total <= substitute_total and delete_total <= insert_total:
                    current[user_index] = delete_total
                    operation = self.DELETION
                elif substitute_total <= insert_total:
                    current[user_index] = substitute_total
                    operation = self.SUBSTITUTION
                else:
                    current[user_index] = insert_total
                    operation = self.INSERTION
                backpointer[score_index, user_index] = operation
            previous = current

        return backpointer

    def get_string_edit_alignment(
        self,
        user_string: Sequence[Note],
        score_string: Sequence[Note],
    ) -> Alignment:
        """Align two already-ordered note sequences with the production costs."""
        backpointer = self.string_edit(user_string, score_string)
        return self.build_mistakes(
            backpointer,
            user_string,
            score_string,
        )

    # phase 2: build mistakes / alignment through tracing DP matrix
    def build_mistakes(self, backpointer: np.ndarray, user_string: Sequence[Note],
                       midi_string: Sequence[Note]) -> Alignment:
        """Trace an edit path and return a complete pitch/timing alignment."""
        # init variables
        user_index, score_index = len(user_string), len(midi_string)
        reversed_pairs: list[tuple[Note | None, Note | None]] = []
        indexed_pitch_mistakes: list[tuple[int, Mistake]] = []
        indexed_timing_mistakes: list[tuple[int, Mistake]] = []
        pitch_tolerance = max(0.0, float(self.config.pitch_tolerance))
        timing_tolerance = max(0.0, float(self.config.timing_tolerance))

        # retrace backwards
        while score_index > 0 or user_index > 0:
            # when we hit the top or left edge, must insert/delete
            if score_index == 0:
                operation = self.INSERTION
            elif user_index == 0:
                operation = self.DELETION
            else: # else take operation denoted by DP backpointer
                operation = int(backpointer[score_index, user_index])

            reverse_pair_index = len(reversed_pairs)
            if operation == self.DELETION:
                score_note = midi_string[score_index - 1]
                reversed_pairs.append((None, score_note))
                indexed_pitch_mistakes.append(
                    (
                        reverse_pair_index,
                        Mistake(
                            type="deletion",
                            user_note=None,
                            midi_note=score_note,
                        ),
                    )
                )
                score_index -= 1
                continue

            if operation == self.INSERTION:
                user_note = user_string[user_index - 1]
                reversed_pairs.append((user_note, None))
                indexed_pitch_mistakes.append(
                    (
                        reverse_pair_index,
                        Mistake(
                            type="insertion",
                            user_note=user_note,
                            midi_note=None,
                        ),
                    )
                )
                user_index -= 1
                continue

            user_note = user_string[user_index - 1]
            score_note = midi_string[score_index - 1]
            reversed_pairs.append((user_note, score_note))

            if self.get_pitch_distance(user_note, score_note) >= pitch_tolerance:
                indexed_pitch_mistakes.append(
                    (
                        reverse_pair_index,
                        Mistake(
                            type="substitution",
                            user_note=user_note,
                            midi_note=score_note,
                        ),
                    )
                )

            onset_offset = user_note.start_time - score_note.start_time
            if abs(onset_offset) > timing_tolerance:
                mistake = Mistake(
                    type="late" if onset_offset > 0 else "early",
                    user_note=user_note,
                    midi_note=score_note,
                )
                mistake.info = f"{onset_offset:+.2f}s"
                indexed_timing_mistakes.append((reverse_pair_index, mistake))

            duration_offset = user_note.duration() - score_note.duration()
            if abs(duration_offset) > timing_tolerance:
                mistake = Mistake(
                    type="long" if duration_offset > 0 else "short",
                    user_note=user_note,
                    midi_note=score_note,
                )
                mistake.info = f"{duration_offset:+.2f}s"
                indexed_timing_mistakes.append((reverse_pair_index, mistake))

            user_index -= 1
            score_index -= 1

        pairs = list(reversed(reversed_pairs))

        def finish(
            indexed_mistakes: list[tuple[int, Mistake]],
        ) -> list[Mistake]:
            for reverse_pair_index, mistake in indexed_mistakes:
                mistake.set_pair_index(len(pairs) - 1 - reverse_pair_index)
            indexed_mistakes.sort(key=lambda item: item[1].pair_index)
            return [mistake for _, mistake in indexed_mistakes]

        # create alignment object and return
        alignment = Alignment(config=self.config, notes=pairs)
        alignment.pitch_mistakes = finish(indexed_pitch_mistakes)
        alignment.timing_mistakes = finish(indexed_timing_mistakes)
        return alignment

    # --- distance / cost functions ---
    def get_pitch_distance(self, user_note: Note, score_note: Note) -> float:
        """Monophonic pitch distance in semitones."""
        return float(abs(user_note.midi_num[0] - score_note.midi_num[0]))

    def get_timing_distance(self, user_note: Note, score_note: Note) -> float:
        """Weighted absolute onset and duration distance in seconds."""
        onset_distance = abs(user_note.start_time - score_note.start_time)
        duration_distance = abs(user_note.duration() - score_note.duration())
        return (
            self.config.alignment_alpha_onset * onset_distance
            + self.config.alignment_alpha_duration * duration_distance
        )

    def get_deletion_cost(self, score_note: Note) -> float:
        """Cost of omitting one score note."""
        return (
            self.config.del_cost
            + self.config.alignment_gamma_time * score_note.duration()
        )

    def get_insertion_cost(self, user_note: Note) -> float:
        """Cost of playing one unmatched note."""
        return (
            self.config.ins_cost
            + self.config.alignment_gamma_time * user_note.duration()
        )

    def get_substitution_cost(self, user_note: Note, score_note: Note) -> float:
        """Cost of pairing two notes, whether the pair is correct or substituted."""
        return (
            self.config.alignment_gamma_pitch
            * self.get_pitch_distance(user_note, score_note)
            + self.config.alignment_gamma_time
            * self.get_timing_distance(user_note, score_note)
        )

    def get_alignment_cost(
        self,
        alignment: Alignment
        | Sequence[tuple[Note | None, Note | None]],
    ) -> float:
        """Return the exact string-edit cost of an alignment path."""
        pairs = alignment.pairs if isinstance(alignment, Alignment) else alignment
        cost = 0.0
        for user_note, score_note in pairs:
            if user_note is None and score_note is not None:
                cost += self.get_deletion_cost(score_note)
            elif user_note is not None and score_note is None:
                cost += self.get_insertion_cost(user_note)
            elif user_note is not None and score_note is not None:
                cost += self.get_substitution_cost(user_note, score_note)
        return float(cost)
