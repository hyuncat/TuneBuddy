import numpy as np

from algorithms.Config import Config
from app_logic.user.ds.Recording import Recording
from app_logic.NoteData import Note, NoteData
from app_logic.Alignment import Alignment, Mistake


class MistakeChecker:
    """Post-processes a recording's alignment to fix mistakes caused by
    *incomplete note detection* — e.g. two played notes detected as one, which
    string editing then reports as a deletion. Splits the merged note back into
    two and re-aligns. Returns brand new NoteData / Alignment (never mutates the
    recording's own data)."""

    def __init__(self, recording: Recording = None, config: Config = None, verbose: bool = False):
        self.recording = recording
        self.config = recording.config if recording else config
        self.pd = self.nd = None
        self.verbose = verbose  # print step-by-step edit diagnostics
        self.MIN_CLOSE = self.config.min_close  # min # of close pitch frames to consider a split viable

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.MIN_CLOSE = self.config.min_close

    def check_mistakes(self, recording: Recording=None) -> tuple[NoteData, Alignment]:
        """Walk the mistakes, gather note edits, apply them, and re-align.
        Returns the new (NoteData, Alignment)."""
        if recording is not None:
            self.recording = recording
        if not self.recording or not self.recording.alignment:
            return None, None
        self.nd, self.pd = self.recording.note_data, self.recording.pitch_data

        mistakes = self.recording.alignment.mistakes
        edits = []  # each edit is (notes_to_remove, notes_to_add)
        for mistake in mistakes:
            edit = None
            if mistake.type == "deletion":
                edit = self.handle_deletion(mistake)
            elif mistake.type == "insertion":
                edit = self.handle_insertion(mistake)
            else:
                continue  # substitutions aren't correctable here

            if edit is not None:
                edits.append(edit)

            if self.verbose:
                if edit is None:
                    print(f"  [check_mistakes] no edit for {mistake}")
                else:
                    removed, added = edit
                    print(f"  [check_mistakes] edit for {mistake}: "
                          f"remove {[round(n.start_time, 2) for n in removed]} -> "
                          f"add {[round(n.start_time, 2) for n in added]}")

        if self.verbose:
            print(f"[check_mistakes] {len(mistakes)} mistakes -> {len(edits)} edits applied")
        new_nd = self._apply_edits(edits)
        new_alignment = self._realign(new_nd)
        return new_nd, new_alignment

    # --- mistake handlers ---------------------------------------------------

    def handle_deletion(self, mistake: Mistake):
        """A deletion means a score note had no match. If the user actually played
        it but it got merged into a neighbor, split that neighbor in two.
        Returns an edit ([merged_note], [first, second]) or None."""
        intended = mistake.midi_note
        prev_note = mistake.user_note  # None for a deletion at the very start of the piece
        next_id = (prev_note.id + 1) if prev_note else 0
        next_note = self.nd.read_note(i=next_id)

        host, targets = self._pick_host(prev_note, next_note, intended)
        if host is None:
            return None

        pitches = self.pd.read(start_time=host.start_time, end_time=host.end_time, clean=True)
        split = self._split_note(host, pitches, *targets)
        if split is None:
            if self.verbose:
                print(f"      [split-fail] host@{host.start_time:.2f}: too few frames ({len(pitches)})")
            return None
        ok = self._split_resembles_score(split, targets)
        if self.verbose:
            print(f"      [split] host@{host.start_time:.2f} -> "
                  f"halves=[{split[0].midi_num[0]:.1f}, {split[1].midi_num[0]:.1f}] "
                  f"targets=({targets[0]:.1f}, {targets[1]:.1f}) resembles={ok} (tol={self.config.tolerance})")
        if not ok:
            return None
        return [host], split

    def handle_insertion(self, mistake: Mistake):
        """An insertion means a user note had no match in the score. If the note
        detector over-segmented one played note into several, the spurious fragment
        sits right beside (and at the same pitch as) the note it broke off from.
        Merge it back into whichever neighbor it's closer to in pitch.
        Returns an edit ([inserted, neighbor], [merged]) or None."""
        inserted = mistake.user_note
        prev_note = self.nd.read_note(i=inserted.id - 1) if inserted.id > 0 else None
        next_note = self.nd.read_note(i=inserted.id + 1)

        neighbor = self._pick_merge_neighbor(inserted, prev_note, next_note)
        if neighbor is None:
            return None

        merged = self._merge_notes(inserted, neighbor)
        return [inserted, neighbor], [merged]


    # --- helpers ------------------------------------------------------------

    def _pick_host(self, prev_note: Note, next_note: Note, intended: Note):
        """Decide which neighbor (if any) actually swallowed the intended note,
        based on how many of its pitch frames land on the intended pitch.
        Returns (host_note, (target_first, target_second)) or (None, None).
        Both targets are *score* pitches: the score note the host aligns to and
        the deleted score note. (Targeting the score, not the host's own observed
        pitch, places the split boundary where the score says the two notes meet.)"""
        n_prev = self._count_close(prev_note, intended) if prev_note else 0
        n_next = self._count_close(next_note, intended) if next_note else 0

        if self.verbose:
            pp = f"{prev_note.midi_num[0]:.1f}@{prev_note.start_time:.2f}" if prev_note else "—"
            nn = f"{next_note.midi_num[0]:.1f}@{next_note.start_time:.2f}" if next_note else "—"
            print(f"      [split?] intended={intended.midi_num[0]:.1f} | prev({pp}) close={n_prev}"
                  f" next({nn}) close={n_next} | need>={self.MIN_CLOSE}")

        if max(n_prev, n_next) < self.MIN_CLOSE:
            return None, None

        if n_prev >= n_next:
            # intended note merged into the END of prev_note
            return prev_note, (self._score_pitch(prev_note), intended.midi_num[0])
        # intended note merged into the START of next_note
        return next_note, (intended.midi_num[0], self._score_pitch(next_note))

    def _score_pitch(self, user_note: Note) -> float:
        """The pitch of the score note `user_note` aligns to, or its own observed
        pitch if it has no match (e.g. the host is itself an insertion)."""
        score_note = self.recording.alignment.get_match(user_note=user_note)
        return score_note.midi_num[0] if score_note else user_note.midi_num[0]

    def _pick_merge_neighbor(self, inserted: Note, prev_note: Note, next_note: Note):
        """Decide which neighbor (if any) the inserted note most likely broke off
        from: the closest in pitch among neighbors that are both
            (a) within tolerance of the inserted note (the 'same note'), and
            (b) themselves a good match to the score note they align to.
        Requiring (b) means a genuinely extra note isn't force-merged into a
        neighbor just because their pitches happen to line up — we only dissolve a
        fragment back into a note we're confident was played correctly.
        An unvoiced neighbor is excluded by both checks. Returns the neighbor or None."""
        pitch = inserted.midi_num[0]
        candidates = []
        for label, neighbor in (("prev", prev_note), ("next", next_note)):
            if neighbor is None:
                if self.verbose:
                    print(f"      [merge?] inserted={pitch:.1f} | {label}=— ")
                continue
            dist = abs(neighbor.midi_num[0] - pitch)
            aligned = self._is_well_aligned(neighbor)
            if self.verbose:
                print(f"      [merge?] inserted={pitch:.1f} | {label}({neighbor.midi_num[0]:.1f}"
                      f"@{neighbor.start_time:.2f}) dist={dist:.2f} (<={self.config.tolerance}?)"
                      f" well_aligned={aligned}")
            if dist <= self.config.tolerance and aligned:
                candidates.append((dist, neighbor))

        if not candidates:
            return None
        return min(candidates, key=lambda c: c[0])[1]

    def _is_well_aligned(self, user_note: Note) -> bool:
        """True if `note` is paired with a score note and sits within tolerance of
        it — a 'good' match, not a substitution, insertion, or unvoiced note. We
        only merge an insertion into such a note (a fragment that broke off a
        correctly-played note); if the neighbor is itself off, the insertion may be
        a genuine extra note, so we leave it alone."""
        score_note = self.recording.alignment.get_match(user_note=user_note)
        if score_note is None:
            return False
        return abs(user_note.midi_num[0] - score_note.midi_num[0]) <= self.config.tolerance

    def _count_close(self, note: Note, intended: Note) -> int:
        """How many of `note`'s voiced pitch frames sit within tolerance of the
        intended pitch. (clean=True drops unvoiced frames, which have no
        candidates after pitch smoothing.)"""
        pitches = self.pd.read(start_time=note.start_time, end_time=note.end_time, clean=True)
        target = intended.midi_num[0]
        return sum(1 for p in pitches
                   if p.candidates and abs(p.candidates[0][0] - target) <= self.config.tolerance)

    def _split_resembles_score(self, split: list, targets: tuple) -> bool:
        """Accept a split only if each half resembles the score note it would
        align to. `targets` are the (host, intended) / (intended, host) pitches
        in the same order as the split halves, so each half must land within
        tolerance of its target. (`< tol` matches StringEditor's "same note"
        convention.)"""
        tol = self.config.tolerance
        return all(abs(note.midi_num[0] - target) <= tol
                   for note, target in zip(split, targets))

    def _split_note(self, note: Note, pitches: list, target_first: float, target_second: float):
        """Split one user note into two at the optimal pitch boundary between the
        two targets (single-breakpoint, minimizing summed pitch distance).
        Returns [first, second] or None if there aren't enough voiced frames."""
        if len(pitches) < 2:  # need >= 2 frames to place a boundary between them
            return None

        midis = np.array([p.candidates[0][0] for p in pitches])
        # cost[k] = sum|midi - target_first| over frames < k  +  sum|midi - target_second| over frames >= k
        left_cost = np.concatenate([[0.0], np.cumsum(np.abs(midis - target_first))])
        right_cost = np.concatenate([np.cumsum(np.abs(midis - target_second)[::-1])[::-1], [0.0]])
        cost = left_cost + right_cost
        k = int(np.argmin(cost[1:-1]) + 1)  # exclude k=0 / k=N so both notes stay non-empty
        split_time = pitches[k].time

        first = Note(
            i=-1, 
            start_time=note.start_time, 
            end_time=split_time,
            midi_num=[float(np.median(midis[:k]))],
            velocity=note.velocity, 
            instrument=note.instrument
        )
        second = Note(
            i=-1, 
            start_time=split_time, 
            end_time=note.end_time,
            midi_num=[float(np.median(midis[k:]))],
            velocity=note.velocity,
            instrument=note.instrument
        )
        return [first, second]

    def _merge_notes(self, inserted: Note, neighbor: Note) -> Note:
        """Fuse the inserted note and its neighbor into one note spanning both
        (they're contiguous, so the union is just min start → max end). The merged
        pitch is the median over the combined voiced frames, so the longer/dominant
        note's pitch wins; falls back to the neighbor's pitch if no voiced frames."""
        start_time = min(inserted.start_time, neighbor.start_time)
        end_time = max(inserted.end_time, neighbor.end_time)

        pitches = self.pd.read(start_time=start_time, end_time=end_time, clean=True)
        midis = [p.candidates[0][0] for p in pitches]
        midi_num = float(np.median(midis)) if midis else neighbor.midi_num[0]

        return Note(
            i=-1,
            start_time=start_time,
            end_time=end_time,
            midi_num=[midi_num],
            velocity=neighbor.velocity,
            instrument=neighbor.instrument,
        )

    def _apply_edits(self, edits: list) -> NoteData:
        """Build a fresh NoteData: drop the notes each edit replaces, add its
        replacements, then re-sort by time and re-assign sequential ids."""
        removed = {n.start_time for removed_notes, _ in edits for n in removed_notes}
        notes = [n for n in self.nd.data.values() if n.start_time not in removed]
        for _, added in edits:
            notes.extend(added)

        notes.sort(key=lambda n: n.start_time)
        new_nd = NoteData()
        for i, note in enumerate(notes):
            note.id = i
            new_nd.write_note(note)
        return new_nd

    def _realign(self, note_data: NoteData) -> Alignment:
        """Re-run string editing against the score to produce a fresh Alignment."""
        midi_notes = self.recording.score_data.note_datas[self.recording.active_instrument]
        notes, mistakes = self.recording.string_editor.string_edit(
            user_string=note_data, midi_string=midi_notes)
        alignment = Alignment(self.config)
        alignment.load_alignment(notes, mistakes)
        return alignment
