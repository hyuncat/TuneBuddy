import numpy as np
import ruptures as rpt
import time
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from PyQt6.QtCore import QObject

from app_logic.user.ds.Recording import Recording
from app_logic.user.ds.PitchData import PitchData
from algorithms.Config import Config

class NoteDetector(QObject):
    ONSET_REFINE_RADIUS = 29
    TRANSITION_WINDOW = 9
    TRANSITION_HOP = 7
    TRANSITION_SLOPE_THRESH = 0.5 / TRANSITION_WINDOW
    SPECTRAL_ONSET_GUARD_FRAMES = 3

    # note-detector shortest-note factors + onset-refinement gates (these used to
    # live on Config; they're algorithm-internal, so they're class constants here).
    MIN_NOTE_FACTOR = 0.6            # PELT min segment = this * Config.min_note_length
    REFINE_WITH_ONSETS = False       # run the spectral-onset repeated-note pass
    ONSET_MIN_NOTE_FACTOR = 1.0      # smallest child note a spectral split may create
    ONSET_MIN_STABLE_RATIO = 0.8     # min voiced/stable fraction on each side of a split

    def __init__(self, recording: Recording=None, config: Config=None, parent: QObject|None=None):
        """initialize the note detection algorithm parameters"""
        super().__init__(parent)

        # algorithm params
        self.recording = recording
        self.config = recording.config if recording else config
        self.PITCH_THRESH = self.config.pitch_thresh
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.PITCH_THRESH = self.config.pitch_thresh
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens

    def get_slope(self, pitches: list[Pitch]):
        """get slope of all voiced pitches in the window"""
        # select only voiced x and y values
        mask  = np.array([p.unvoiced_prob < self.UNV_THRESH if p else False for p in pitches]) # boolean mask

        all_x = np.linspace(start=0, stop=len(pitches), num=len(pitches))
        x_voiced = all_x[mask]
        y_voiced = np.array([p.value for p, m in zip(pitches, mask) if m])

        if x_voiced.size == 0:
            return 0.0, 0.0

        # get slope + intercept of only voiced pitches
        A = np.vstack([x_voiced, np.ones_like(x_voiced)]).T
        slope, intercept = np.linalg.lstsq(A, y_voiced, rcond=None)[0]

        return slope, intercept
    
    def get_median_pitches(self, pitches: list[Pitch]):
        """return median pitches of whatever exists in the candidate
        slots for indices 0:2"""
        N = 3
        medians = [-1] * N

        # select only voiced frames
        voiced = [p for p in pitches if p and p.value != -1 and p.unvoiced_prob < self.UNV_THRESH]
        if not voiced:
            return medians

        # collect candidates in each column
        cols = [[] for _ in range(N)]

        for p in voiced:
            cols[0].append(p.value)
            for i in range(1, min(N, len(p.candidate_pitches))):
                pitch_val = p.candidate_pitches[i][0]
                if pitch_val != -1:
                    cols[i].append(pitch_val)

        # compute medians
        for i in range(N):
            if cols[i]:
                medians[i] = float(np.median(cols[i]))

        return medians

    # ------------------------------------------------------------------ #
    # offline detector (ruptures PELT / L2)
    # ------------------------------------------------------------------ #
    def _pelt_min_size_from_score(self) -> int:
        """Minimum PELT segment size in pitch frames.

        Read the shortest note directly from the active score/clip, apply the
        note-detector factor, and convert seconds to pitch frames via sr / h1.
        """
        seconds = getattr(self.config, "min_note_length", Config.DEFAULT_MIN_NOTE_LENGTH)
        if self.recording is not None:
            try:
                note_data = self.recording.score_data.clipped_note_data(
                    channel=self.recording.active_instrument
                )
                seconds = note_data.get_min_note_length(default=float(seconds), clean=True)
            except (AttributeError, KeyError, TypeError):
                pass
        frame_rate = self.config.sr / self.config.h1
        return max(1, int(np.ceil(max(0.0, float(seconds) * self.MIN_NOTE_FACTOR) * frame_rate)))

    def _pelt_penalty(self, min_size: int) -> float:
        """Penalty tuned to the existing pitch-change threshold.

        For two equal min-size segments, a mean shift of PITCH_THRESH reduces L2
        cost by roughly 0.5 * min_size * PITCH_THRESH^2, so use that as the PELT
        split threshold.
        """
        return 0.5 * min_size * (self.PITCH_THRESH ** 2)

    def _pelt_jump(self, jump: int | None = None) -> int:
        """PELT candidate-boundary stride in pitch frames."""
        if jump is None:
            jump = self.config.h2
        return max(1, int(jump))

    def _is_pelt_frame(self, p: Pitch | None) -> bool:
        """Whether a pitch frame should participate in PELT segmentation."""
        midi = self._frame_pitch(p)
        return (
            p is not None
            and not getattr(p, "is_transition", False)
            and p.unvoiced_prob < self.UNV_THRESH
            and midi not in (None, -1)
        )

    def _pelt_signal(self, pitches: list[Pitch]) -> np.ndarray:
        """Primary MIDI pitch per voiced frame."""
        values = []
        for p in pitches:
            values.append(float(self._frame_pitch(p)))
        return np.asarray(values, dtype=float).reshape(-1, 1)

    def _pelt_runs(self, pitch_data: PitchData, min_gap_frames: int) -> list[list[Pitch]]:
        """Stable voiced PELT runs.

        Transition frames are explicit non-note material, so they split runs even
        when the slide is shorter than the generic unvoiced/ignored-gap threshold.
        Other ignored frames still need to be sustained before they break a run,
        which keeps isolated noisy frames from fragmenting notes. Very short
        voiced islands are discarded here, before PELT is called, because they do
        not have enough evidence to stand alone as notes.
        """
        runs = []
        run = []
        gap_frames = 0
        transition_gap = False
        min_run_frames = max(1, int(np.ceil(0.4 * min_gap_frames)))

        def append_run():
            nonlocal run
            if len(run) >= min_run_frames:
                runs.append(run)
            run = []

        for p in pitch_data.data:
            if self._is_pelt_frame(p):
                if run and (transition_gap or gap_frames >= min_gap_frames):
                    append_run()
                run.append(p)
                gap_frames = 0
                transition_gap = False
            elif run:
                gap_frames += 1
                if p is not None and getattr(p, "is_transition", False):
                    transition_gap = True

        if run:
            append_run()

        return runs

    def _pelt_segment_pitch(self, pitches: list[Pitch]) -> list[float]:
        """Summarize a PELT segment as up to three median pitch candidates."""
        if not pitches:
            return [-1, -1, -1]
        return self.get_median_pitches(pitches)

    def _pelt_boundary_time(self, pitches: list[Pitch], i: int) -> float:
        """Time for the boundary before frame i; i == len gives one frame past end."""
        if i <= 0:
            return pitches[0].time
        if i < len(pitches):
            return pitches[i].time
        return pitches[-1].time + (self.config.h1 / self.config.sr)

    def _same_note_pitch(self, a: Note, b_midi: list[float]) -> bool:
        if a is None or not a.midi_num or not b_midi:
            return False
        a0, b0 = a.midi_num[0], b_midi[0]
        if a0 == -1 and b0 == -1:
            return True
        if a0 == -1 or b0 == -1:
            return False
        return abs(a0 - b0) <= self.PITCH_THRESH

    def _spectral_onset_refinement_requested(
        self, refine_with_onsets: bool | None,
    ) -> bool:
        if refine_with_onsets is None:
            refine_with_onsets = self.REFINE_WITH_ONSETS
        return bool(refine_with_onsets)

    def _detect_spectral_onsets(self):
        """Run the librosa-backed OnsetDetector for this recording, if available."""
        if self.recording is None or getattr(self.recording, "audio_data", None) is None:
            return None
        if getattr(self.recording.audio_data, "end_index", 0) <= 0:
            return None
        existing = getattr(self.recording, "onset_data", None)
        if existing is not None:
            return existing

        from app_logic.user.ds.OnsetData import OnsetDetector

        detector = getattr(self.recording, "onset_detector", None)
        if detector is None:
            detector = OnsetDetector(self.recording)
            self.recording.onset_detector = detector
        detector.update_config(self.config)
        onset_data = detector.detect()
        self.recording.onset_data = onset_data
        return onset_data

    @staticmethod
    def _nearest_boundary_distance(t: float, boundaries: list[float]) -> float:
        if not boundaries:
            return float("inf")
        i = np.searchsorted(boundaries, t)
        candidates = []
        if i > 0:
            candidates.append(abs(t - boundaries[i - 1]))
        if i < len(boundaries):
            candidates.append(abs(t - boundaries[i]))
        return min(candidates) if candidates else float("inf")

    def _spectral_onset_min_split_seconds(self) -> float:
        """Smallest allowed child note created by spectral-onset refinement."""
        factor = self.ONSET_MIN_NOTE_FACTOR
        seconds = getattr(self.config, "min_note_length", Config.DEFAULT_MIN_NOTE_LENGTH)
        if self.recording is not None:
            try:
                note_data = self.recording.score_data.clipped_note_data(
                    channel=self.recording.active_instrument
                )
                seconds = note_data.get_min_note_length(default=float(seconds), clean=True)
            except (AttributeError, KeyError, TypeError):
                pass
        return max(self.config.h1 / self.config.sr, max(0.0, float(seconds) * factor))

    def _spectral_onset_min_split_frames(self) -> int:
        seconds = getattr(self.config, "min_note_length", Config.DEFAULT_MIN_NOTE_LENGTH)
        if self.recording is not None:
            try:
                note_data = self.recording.score_data.clipped_note_data(
                    channel=self.recording.active_instrument
                )
                seconds = note_data.get_min_note_length(default=float(seconds), clean=True)
            except (AttributeError, KeyError, TypeError):
                pass
        frame_rate = self.config.sr / self.config.h1
        return max(1, int(np.ceil(max(0.0, float(seconds) * self.ONSET_MIN_NOTE_FACTOR) * frame_rate)))

    def _is_stable_spectral_onset(self, pitch_data: PitchData, t: float) -> bool:
        """Reject spectral peaks that fall inside a rest/gap or slide transition."""
        i = pitch_data.time_to_index(t)
        if i < 0 or i >= len(pitch_data.data):
            return False
        lo = max(0, i - self.SPECTRAL_ONSET_GUARD_FRAMES)
        hi = min(len(pitch_data.data), i + self.SPECTRAL_ONSET_GUARD_FRAMES + 1)
        if lo >= hi:
            return False

        window = pitch_data.data[lo:hi]
        return bool(window) and all(self._is_pelt_frame(p) for p in window)

    def _stable_span_frames(
        self,
        pitch_data: PitchData,
        start_time: float,
        end_time: float,
    ) -> tuple[int, int]:
        frames = pitch_data.read(
            start_time=start_time,
            end_time=end_time,
            clean=False,
        )
        total = len(frames)
        stable = sum(1 for p in frames if self._is_pelt_frame(p))
        return stable, total

    def _has_stable_split_support(
        self,
        pitch_data: PitchData,
        start_time: float,
        split_time: float,
        end_time: float,
    ) -> bool:
        min_frames = self._spectral_onset_min_split_frames()
        min_ratio = max(0.0, min(1.0, float(self.ONSET_MIN_STABLE_RATIO)))

        for a, b in ((start_time, split_time), (split_time, end_time)):
            stable, total = self._stable_span_frames(pitch_data, a, b)
            if stable < min_frames:
                return False
            if total <= 0 or (stable / total) < min_ratio:
                return False
        return True

    def _span_pitch(
        self,
        pitch_data: PitchData,
        start_time: float,
        end_time: float,
        fallback: list[float],
    ) -> list[float]:
        frames = pitch_data.read(
            start_time=start_time,
            end_time=end_time,
            clean=False,
        )
        midi_num = self._pelt_segment_pitch([
            p for p in frames if self._is_pelt_frame(p)
        ])
        return midi_num if midi_num and midi_num[0] != -1 else list(fallback)

    @staticmethod
    def _copy_note_span(
        note: Note,
        note_id: int,
        start_time: float,
        end_time: float,
        midi_num: list[float],
    ) -> Note:
        return Note(
            i=note_id,
            start_time=float(start_time),
            end_time=float(end_time),
            midi_num=list(midi_num),
            velocity=note.velocity,
            instrument=note.instrument,
        )

    def detect_notes(
        self,
        pitch_data: PitchData,
        model: str = "l2",
        pen: float | None = None,
        jump: int | None = None,
        refine_with_onsets: bool | None = None,
        onset_data=None,
        verbose: bool = False,
    ) -> NoteData:
        """Offline note detection using ruptures' PELT change-point detector.

        `model` selects the ruptures cost function ("l2" is the production default;
        "l1"/"rbf"/"normal" exist for benchmarking — see notebooks/benchmark_notes).
        `pen` overrides the PELT penalty; None derives it from PITCH_THRESH via
        `_pelt_penalty` (tuned for L2, a reasonable baseline for the others).
        `jump` controls the candidate-boundary stride; None uses Config.h2.
        `refine_with_onsets` enables a second pass that inserts missing repeated-
        note boundaries from spectral OnsetData; None follows
        NoteDetector.REFINE_WITH_ONSETS.

        Unvoiced frames and pitch-transition (slide) frames are excluded from the
        signal. A sustained run of excluded frames splits the PELT input, so notes
        can stop during real gaps instead of being forced into one contiguous
        voiced-only timeline. Transition frames are excluded when
        detect_transitions() has flagged them; the onset-refined path initializes
        those flags itself if needed.
        """
        start = time.perf_counter()
        refine_onsets = self._spectral_onset_refinement_requested(refine_with_onsets)
        if verbose:
            print(
                f"[NoteDetector] detecting notes "
                f"(model={model}, refine_onsets={refine_onsets})",
                flush=True,
            )
        if refine_onsets and any(p is not None and p.is_transition is None for p in pitch_data.data):
            self.detect_transitions(pitch_data, verbose=verbose)

        nd = NoteData()
        min_size = self._pelt_min_size_from_score()
        runs = self._pelt_runs(pitch_data, min_gap_frames=min_size)
        if not runs:
            if verbose:
                print("[NoteDetector] done: 0 runs, 0 note(s)", flush=True)
            return nd

        penalty = self._pelt_penalty(min_size) if pen is None else pen
        pelt_jump = self._pelt_jump(jump)
        note_index = 0
        if verbose:
            total_frames = sum(len(run) for run in runs)
            print(
                f"[NoteDetector] {len(runs)} voiced run(s), {total_frames} frame(s), "
                f"min_size={min_size}, penalty={penalty:.3f}, jump={pelt_jump}",
                flush=True,
            )

        for pitches in runs:
            signal = self._pelt_signal(pitches)
            n_frames = len(pitches)

            if n_frames < 2 * min_size:
                bkps = [n_frames]
            else:
                try:
                    bkps = (
                        rpt.Pelt(model=model, min_size=min_size, jump=pelt_jump)
                        .fit(signal)
                        .predict(pen=penalty)
                    )
                except rpt.exceptions.BadSegmentationParameters:
                    bkps = [n_frames]

            prev = 0
            first_segment_in_run = True
            for bkp in bkps:
                end = min(int(bkp), n_frames)
                if end <= prev:
                    continue

                segment = pitches[prev:end]
                midi_num = self._pelt_segment_pitch(segment)
                start_time = self._pelt_boundary_time(pitches, prev)
                end_time = self._pelt_boundary_time(pitches, end)
                if end_time <= start_time:
                    prev = end
                    continue

                last = (
                    nd.read_note(i=len(nd.times) - 1)
                    if nd.times and not first_segment_in_run
                    else None
                )
                if self._same_note_pitch(last, midi_num):
                    last.end_time = end_time
                else:
                    nd.write_note(Note(
                        i=note_index,
                        start_time=start_time,
                        end_time=end_time,
                        midi_num=midi_num,
                    ))
                    note_index += 1

                prev = end
                first_segment_in_run = False

        if refine_onsets:
            onset_data = onset_data if onset_data is not None else self._detect_spectral_onsets()
            nd = self.refine_with_spectral_onsets(
                nd,
                pitch_data,
                onset_data,
                verbose=verbose,
            )

        if verbose:
            print(
                f"[NoteDetector] done: {len(nd.times)} note(s) in "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )
        return nd

    def refine_with_spectral_onsets(
        self,
        note_data: NoteData,
        pitch_data: PitchData,
        onset_data,
        verbose: bool = False,
    ) -> NoteData:
        """Split same-pitch PELT notes at trustworthy spectral onsets.

        This catches repeated notes that PELT merges because their pitch means are
        effectively identical. A spectral onset is used only when both resulting
        child notes are at least the score-derived minimum note length and the
        surrounding pitch frames are voiced, non-transition frames.
        """
        if onset_data is None or len(onset_data) == 0 or not note_data.times:
            return note_data

        notes = note_data.read(i=0, j=len(note_data.times))
        if not notes:
            return note_data

        min_split_seconds = self._spectral_onset_min_split_seconds()
        frame_dt = self.config.h1 / self.config.sr
        split_count = 0

        for onset_time in onset_data.read():
            t = float(onset_time)
            if not np.isfinite(t):
                continue

            boundaries = sorted(
                [n.start_time for n in notes] + [n.end_time for n in notes]
            )
            if self._nearest_boundary_distance(t, boundaries) < min_split_seconds:
                continue
            if not self._is_stable_spectral_onset(pitch_data, t):
                continue

            note_index = None
            for i, note in enumerate(notes):
                if note.start_time + frame_dt < t < note.end_time - frame_dt:
                    note_index = i
                    break
            if note_index is None:
                continue

            note = notes[note_index]
            if not note.midi_num or note.midi_num[0] == -1:
                continue
            if (
                t - note.start_time < min_split_seconds
                or note.end_time - t < min_split_seconds
            ):
                continue
            if not self._has_stable_split_support(
                pitch_data, note.start_time, t, note.end_time
            ):
                continue

            left_midi = self._span_pitch(
                pitch_data, note.start_time, t, fallback=note.midi_num
            )
            right_midi = self._span_pitch(
                pitch_data, t, note.end_time, fallback=note.midi_num
            )
            notes[note_index:note_index + 1] = [
                self._copy_note_span(
                    note, note.id, note.start_time, t, left_midi
                ),
                self._copy_note_span(
                    note, note.id + 1, t, note.end_time, right_midi
                ),
            ]
            split_count += 1

        if split_count == 0:
            if verbose:
                print("[NoteDetector] spectral onset refinement: 0 split(s)", flush=True)
            return note_data

        refined = NoteData()
        for idx, note in enumerate(notes):
            note.id = idx
            refined.write_note(note)
        if verbose:
            print(
                f"[NoteDetector] spectral onset refinement: {split_count} split(s)",
                flush=True,
            )
        return refined

    # ------------------------------------------------------------------ #
    # onset refinement (Method 1: pitch-transition + voicing fallback)
    # ------------------------------------------------------------------ #
    def _frame_pitch(self, p: Pitch):
        """primary midi of a frame, or None if the frame is missing / empty."""
        if p is None or p.value == -1:
            return None
        return p.value

    def _changepoint(self, signal: np.ndarray) -> int | None:
        """single best mean-shift split of a 1-D window via ruptures (L2 cost):
        the index of the first sample of the second segment, or None when the
        window is too short or too flat to split. jump=1 keeps the split at full
        frame resolution — ruptures' default jump=5 would quantise it and throw
        away the precision this whole pass exists to recover."""
        if len(signal) < 2 or np.ptp(signal) == 0:
            return None
        algo = rpt.Dynp(model="l2", min_size=1, jump=1).fit(signal.reshape(-1, 1))
        return int(algo.predict(n_bkps=1)[0])

    def _find_pitch_crossing(self, pitches: list[Pitch], lo: int, hi: int) -> int | None:
        """voiced<->voiced: the boundary is the change-point (mean shift) of the
        window's pitch track. Unvoiced/gap frames carry no pitch, so they're
        dropped and the split is mapped back onto real frame indices."""
        idx = [k for k in range(lo, hi)
               if self._frame_pitch(pitches[k]) not in (None, -1)]
        if len(idx) < 2:
            return None
        sig = np.array([self._frame_pitch(pitches[k]) for k in idx], dtype=float)
        s = self._changepoint(sig)
        return idx[s] if s is not None else None

    def _find_voicing_change(self, pitches: list[Pitch], lo: int, hi: int) -> int | None:
        """rest<->note: the boundary is the change-point of the window's voiced
        indicator (1 where the frame has a pitch, else 0) — derived purely from
        the pitch track, no unvoiced-probability threshold needed."""
        sig = np.array([0.0 if self._frame_pitch(pitches[k]) in (None, -1) else 1.0
                        for k in range(lo, hi)], dtype=float)
        s = self._changepoint(sig)
        return lo + s if s is not None else None

    def refine_onsets(self, note_data: NoteData, pitch_data: PitchData) -> NoteData:
        """relocate the hop-quantized note boundaries onto their true onsets.

        For each *shared* boundary between consecutive notes we search nearby
        pitch frames and move the split to single-frame (h1, ~3 ms) detail:
          - voiced<->voiced : change-point (mean shift) of the pitch track
          - rest<->note     : change-point of the voiced/unvoiced indicator
          - rest<->rest     : left untouched (no cue)
        The original boundary is kept whenever no confident split is found, and
        the split is clamped strictly inside the pair so notes can't collapse.
        """
        notes = note_data.read(i=0, j=len(note_data.times))
        if len(notes) < 2:
            return note_data

        pitches = pitch_data.data
        n_frames = len(pitches)
        radius = self.ONSET_REFINE_RADIUS
        frame_dt = self.config.h1 / self.config.sr

        for a, b in zip(notes, notes[1:]):
            med_a, med_b = a.midi_num[0], b.midi_num[0]

            # clamp the search to this pair's own extent so we never wander into
            # neighbouring notes (a.start_time may already have been refined)
            bound_idx = pitch_data.time_to_index(a.end_time)
            a_start = pitch_data.time_to_index(a.start_time)
            b_end = pitch_data.time_to_index(b.end_time)
            lo = max(a_start + 1, bound_idx - radius)
            hi = min(b_end, bound_idx + radius, n_frames)
            if lo >= hi:
                continue

            # find best onset
            a_voiced, b_voiced = med_a != -1, med_b != -1
            # if both are voiced
            if a_voiced and b_voiced:
                k = self._find_pitch_crossing(pitches, lo, hi)
            # if one is not voiced
            elif a_voiced != b_voiced:
                k = self._find_voicing_change(pitches, lo, hi)
            else:
                k = None  # rest<->rest: nothing to refine
            if k is None:
                continue # no confident split found (at boundaries)

            new_t = pitch_data.t_origin + k * frame_dt
            if a.start_time < new_t < b.end_time:
                a.end_time = new_t
                b.start_time = new_t

        # rebuild so NoteData's dict/times index reflects the new start_times
        refined = NoteData()
        for idx, n in enumerate(notes):
            n.id = idx
            refined.write_note(n)
        return refined

    # ------------------------------------------------------------------ #
    # transition flagging (post-pass, run after onset refinement)
    # ------------------------------------------------------------------ #
    def detect_transitions(self, pitch_data: PitchData, verbose: bool = False) -> None:
        """Mark every high-slope (pitch-transition) frame in `pitch_data`.

        This pass slides a short window over the whole pitch track and
        sets `Pitch.is_transition = True` for any frame inside a window whose
        slope looks like a slide. Downstream, update_alignment_distances leaves
        these frames' distances unset (grey) so they aren't scored.

        Order: it only reads the pitch track (no note dependency), so analyze()
        runs it BEFORE detect_notes2() (PELT), which excludes the flagged slide
        frames from segmentation. It must precede recompute_note_pitches(),
        prune_transition_notes() and update_alignment_distances()."""
        pitches = pitch_data.data
        start = time.perf_counter()

        # default every present frame to non-transition first (idempotent re-runs)
        for p in pitches:
            if p is not None:
                p.is_transition = False

        windows = 0
        for i in range(0, len(pitches) - self.TRANSITION_WINDOW, self.TRANSITION_HOP):
            window = pitches[i:i + self.TRANSITION_WINDOW]
            if window[0] is None:
                continue
            slope, _ = self.get_slope(window)
            if abs(slope) >= self.TRANSITION_SLOPE_THRESH:
                windows += 1
                for p in window:
                    if p is not None:
                        p.is_transition = True
        if verbose:
            transition_frames = sum(1 for p in pitches if p is not None and p.is_transition)
            print(
                f"[NoteDetector] transitions: {transition_frames} frame(s) "
                f"from {windows} window(s) in {time.perf_counter() - start:.2f}s",
                flush=True,
            )

    def recompute_note_pitches(
        self,
        note_data: NoteData,
        pitch_data: PitchData,
        verbose: bool = False,
    ) -> None:
        """Re-median each note's pitch over only its non-transition frames.

        A note's midi_num is the median of its frames, but onset refinement pulls
        high-slope slide frames into the note's span; for a descending slide those
        frames sit above the settled pitch and drag the median sharp (e.g. an F5
        reads F#5 -> a false 'too sharp' substitution). Excluding transition frames
        (flagged by detect_transitions) restores the settled pitch. A note with no
        non-transition voiced frames (rests, all-slide blips) keeps its original
        midi_num.

        Run after detect_transitions() and before detect_mistakes()."""
        changed = 0
        for note in note_data.data.values():
            if note is None or note.midi_num[0] == -1:
                continue  # leave rests alone
            frames = pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=False
            )
            kept = [p for p in frames if p is not None and not p.is_transition]
            med = self.get_median_pitches(kept)
            if med[0] != -1:  # only overwrite when a voiced estimate survives
                if med != note.midi_num:
                    changed += 1
                note.midi_num = med
        if verbose:
            print(f"[NoteDetector] recomputed pitch for {changed} note(s)", flush=True)

    def prune_transition_notes(self, note_data: NoteData, pitch_data: PitchData,
                               frac_thresh: float = 0.5,
                               verbose: bool = False) -> NoteData:
        """Drop notes that are almost entirely transition frames or tiny blips.

        Once detect_transitions flags slide frames, any note whose voiced frames
        are more than `frac_thresh` transition is a phantom of that slide ->
        remove it. Very short standalone voiced islands are filtered earlier in
        _pelt_runs(), before PELT is called. Notes with no voiced frames (rests)
        are left untouched.
        Returns a rebuilt, reindexed NoteData.

        Run after detect_transitions(), before detect_mistakes()."""
        survivors = []
        notes = note_data.read(i=0, j=len(note_data.times))
        for note in notes:
            if note is None:
                continue
            voiced = pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=True
            )
            n_trans = sum(1 for p in voiced if p.is_transition)
            if voiced and n_trans > frac_thresh * len(voiced):
                continue  # phantom slide note -> drop
            survivors.append(note)

        # rebuild so NoteData's dict/times index reflects the dropped notes
        pruned = NoteData()
        for idx, n in enumerate(survivors):
            n.id = idx
            pruned.write_note(n)
        if verbose:
            print(
                f"[NoteDetector] pruned {len(notes) - len(survivors)} transition note(s)",
                flush=True,
            )
        return pruned
