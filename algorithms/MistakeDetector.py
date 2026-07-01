from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from algorithms.Config import Config
from app_logic.Alignment import Alignment, Mistake
from app_logic.NoteData import NoteData, Note

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording


@dataclass(frozen=True)
class _CostModel:
    """Onset-aware DP costs as negative log-priors of the mistake model (anchored so
    a correct on-time match costs 0), plus the Gaussian onset/pitch spreads."""
    c_match: float
    c_sub: float
    c_ins: float
    c_del: float
    onset_sigma: float
    pitch_sigma: float


class MistakeDetector:
    def __init__(self, recording: Recording=None, config: Config=None):
        if isinstance(recording, Config) and config is None:
            config = recording
            recording = None
        self.recording = recording
        self.config = recording.config if recording else config

        # pitch mistake edit costs
        self.INSERTION_COST = self.config.ins_cost
        self.DELETION_COST = self.config.del_cost
        self.TOLERANCE = self.config.pitch_tolerance

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.INSERTION_COST = self.config.ins_cost
        self.DELETION_COST = self.config.del_cost
        self.TOLERANCE = self.config.pitch_tolerance

    def detect_pitch_mistakes(self, user_string: NoteData, midi_string: NoteData):
        """Pitch-only Levenshtein alignment of user vs score notes (returns
        alignment pairs + pitch mistakes). The substitution cost depends ONLY on
        pitch distance, so the minimal-edit path ignores WHEN each note was played.
        This is the original aligner and the A/B baseline for
        detect_pitch_mistakes_onset_aware()."""
        return self._align(user_string, midi_string, onset_aware=False)

    def detect_pitch_mistakes_onset_aware(self, user_string: NoteData, midi_string: NoteData):
        """Onset-aware alignment: same DP, but the costs are the negative log-priors
        of the configured mistake model (Config.alignment_priors) plus Gaussian
        onset/pitch penalties, instead of the pitch-only magic numbers. A match pays
        only its onset penalty; a substitution adds the wrong-note prior plus
        UNBOUNDED quadratic onset and pitch penalties — so a pairing that is far in
        time (an off-by-one alignment shift around an indel) or implausibly wrong in
        pitch costs more than an insertion+deletion and is split instead of becoming
        a spurious substitution. This anchors the alignment in time (among
        equal-pitch candidates a user note pairs with the score note nearest in
        onset) without hand-tuned costs. A/B against detect_pitch_mistakes
        (pitch-only)."""
        return self._align(user_string, midi_string, onset_aware=True)

    def _align_cost_model(self) -> _CostModel:
        """Derive the onset-aware DP costs as negative log-priors of the configured
        mistake model, anchored so a correct on-time match costs 0. Automated: the
        substitution/insertion/deletion costs come straight from
        Config.alignment_priors(), so they track mistake_rate without hand-tuning."""
        p_correct, p_sub, p_ins, p_del = self.config.alignment_priors()
        base = -np.log(max(p_correct, 1e-9))
        nll = lambda p: float(-np.log(max(p, 1e-9)) - base)
        return _CostModel(
            c_match=0.0,
            c_sub=nll(p_sub),
            c_ins=nll(p_ins),
            c_del=nll(p_del),
            onset_sigma=max(float(self.config.align_onset_sigma), 1e-3),
            pitch_sigma=max(float(self.config.align_pitch_sigma), 1e-3),
        )

    def _substitution_cost(self, user_note: Note, midi_note: Note, model: _CostModel | None) -> float:
        """Cost of matching/substituting user_note against midi_note.
        Pitch-only (model is None): 0 within tolerance ('same note'), else the
        semitone distance clamped to 10 (the original cost).
        Onset-aware (model given): a within-tolerance match costs only its Gaussian
        onset penalty; a substitution adds the wrong-note prior plus unbounded
        quadratic onset AND pitch penalties, so an implausibly far/wrong pairing
        loses to insertion+deletion."""
        d = abs(self.get_distance(user_note, midi_note))
        if model is None:
            return 0.0 if d < self.TOLERANCE else min(d, 10)
        onset_pen = 0.5 * ((user_note.start_time - midi_note.start_time) / model.onset_sigma) ** 2
        if d < self.TOLERANCE:
            return model.c_match + onset_pen
        return model.c_sub + onset_pen + 0.5 * (d / model.pitch_sigma) ** 2

    def _align(self, user_string: NoteData, midi_string: NoteData, onset_aware: bool):
        """Shared edit-distance core for the pitch-only and onset-aware aligners.
        Builds the DP, traces back, and returns (notes, mistakes). Only the
        operation COSTS differ between the two (pitch-only magic numbers vs the
        onset-aware negative-log-prior model); mistake CLASSIFICATION below stays
        pitch-only — onset never invents a pitch mistake, it only steers which notes
        pair up."""
        user_notes = list(user_string.data.values())
        user_notes = [n for n in user_notes if n.midi_num[0] != -1]

        # cost model: onset-aware uses negative log-priors (ins/del costs included);
        # pitch-only keeps the original flat ins/del costs.
        model = self._align_cost_model() if onset_aware else None
        ins_cost = model.c_ins if model else self.INSERTION_COST
        del_cost = model.c_del if model else self.DELETION_COST

        # setup dp matrix
        N = len(midi_string.times)
        M = len(user_notes)

        mat = np.zeros([N+1, M+1], dtype=np.float64)
        backpointer = np.zeros([N+1, M+1], dtype=np.int64)

        # initialize first row / column
        mat[0, :] = np.cumsum([0]+[ins_cost]*M) # all insertions
        mat[:, 0] = np.cumsum([0]+[del_cost]*N) # all deletions

        for i in range(1, N+1): # midi index
            midi_note = midi_string.read_note(i=i-1)
            for j in range(1, M+1): # user index
                user_note = user_notes[j-1]
                SUB_COST = self._substitution_cost(user_note, midi_note, model)

                top_three = np.array([
                    mat[i-1, j] + del_cost,
                    mat[i-1, j-1] + SUB_COST,
                    mat[i, j-1] + ins_cost
                ])
                mat[i, j] = np.min(top_three)
                backpointer[i, j] = np.argmin(top_three) # eg, 0=del, 1=sub, 2=ins

        # traceback the backpointer
        # print("starting pitch mistake traceback...")
        i = N
        j = M

        mistakes = []
        notes = []
        mistakes_to_reverse_position = {}
        while i>0 or j>0:
            # on the boundaries the backpointer is unset (0), so force the only
            # legal move: all-insertions along the top row, all-deletions down
            # the left column. otherwise the earliest notes get silently dropped.
            if i == 0:
                mistake_type = 2  # only user notes remain -> insertion
            elif j == 0:
                mistake_type = 0  # only score notes remain -> deletion
            else:
                mistake_type = backpointer[i, j]
            midi_note = midi_string.read_note(i=i-1) if i > 0 else None
            user_note = user_notes[j-1] if j > 0 else None

            # 0: deletion
            if mistake_type==0 and i>0:
                # print(f"--> DELETION at i={i}, j={j}")
                mistake = Mistake(type="deletion", user_note=user_note, midi_note=midi_note)
                mistakes_to_reverse_position[mistake] = len(notes)
                mistakes.append(mistake)
                notes.append((None, midi_note))
                i -= 1

            # 1: substitution / no change
            elif mistake_type==1 and i>0 and j>0:
                note_distance = self.get_distance(user_note, midi_note)
                if abs(note_distance) >= self.TOLERANCE:
                    # print(f"--> SUBSTITUTION at i={i}, j={j} (distance={note_distance})")
                    mistake = Mistake(type="substitution", user_note=user_note, midi_note=midi_note)
                    mistakes_to_reverse_position[mistake] = len(notes)
                    mistakes.append(mistake)
                notes.append((user_note, midi_note))
                i -= 1
                j -= 1

            # 2: insertion
            elif mistake_type==2 and j>0:
                # print(f"--> INSERTION at i={i}, j={j}")
                mistake = Mistake(type="insertion", user_note=user_note, midi_note=midi_note)
                mistakes_to_reverse_position[mistake] = len(notes)
                mistakes.append(mistake)
                j -= 1
                notes.append((user_note, None))
            else:
                # fallback to prevent infinite loop
                # print(f"[warning] Invalid state at i={i}, j={j}, backpointer={mistake_type}")
                break

        notes = list(reversed(notes))
        mistakes = list(reversed(mistakes))
        # print(f"Done! Took {time.time() - start:.2f} seconds")
        for mistake in mistakes:
            mistake.set_pair_index(len(notes) - 1 - mistakes_to_reverse_position[mistake])
        return notes, mistakes

    def detect_timing_mistakes(self, alignment: Alignment | None = None) -> list[Mistake]:
        """Derive early/late/short/long timing mistakes from alignment pairs.

        This is post-alignment analysis: it reads the master `alignment.pairs`
        list, stores the derived list on `alignment.timing_mistakes`, and returns
        that list. Only matched pairs can produce timing mistakes.
        """
        alignment = alignment or (self.recording.alignment if self.recording else None)
        if alignment is None:
            return []

        timing_tol = max(0.0, float(self.config.timing_tolerance))
        mistakes: list[Mistake] = []
        for pair_index, (user_note, score_note) in enumerate(alignment.pairs):
            if user_note is None or score_note is None:
                continue

            onset_off = user_note.start_time - score_note.start_time
            user_dur = max(1e-9, user_note.end_time - user_note.start_time)
            score_dur = max(1e-9, score_note.end_time - score_note.start_time)
            dur_off = user_dur - score_dur

            if abs(onset_off) > timing_tol:
                m = Mistake(
                    type="late" if onset_off > 0 else "early",
                    user_note=user_note,
                    midi_note=score_note,
                )
                m.info = f"{onset_off:+.2f}s"
                m.set_pair_index(pair_index)
                mistakes.append(m)

            if abs(dur_off) > timing_tol:
                m = Mistake(
                    type="long" if dur_off > 0 else "short",
                    user_note=user_note,
                    midi_note=score_note,
                )
                m.info = f"{dur_off:+.2f}s"
                m.set_pair_index(pair_index)
                mistakes.append(m)

        alignment.timing_mistakes = mistakes
        alignment.reindex_mistakes(mistakes)
        return mistakes
    
    def get_distance(self, user_note: Note, midi_note: Note):
        """Return the closest pitch distance between the user note and score note.
        When score note is a chord, user matches the nearest chord member."""
        return min(abs(u - m) for u in user_note.midi_num for m in midi_note.midi_num)
        
