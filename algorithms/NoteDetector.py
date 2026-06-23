import numpy as np
from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch
from PyQt6.QtCore import pyqtSignal, QObject
import threading

from app_logic.user.ds.Recording import Recording
from app_logic.user.ds.PitchData import PitchData
from algorithms.Config import Config

class NoteDetector(QObject):
    note_detected = pyqtSignal(float)
    
    def __init__(self, recording: Recording=None, config: Config=None, parent: QObject|None=None):
        """initialize the note detection algorithm parameters"""
        super().__init__(parent)

        # algorithm params
        self.recording = recording
        self.config = recording.config if recording else config
        self.w = self.config.w2
        self.hop = self.config.h2
        self.PITCH_THRESH = self.config.pitch_thresh
        self.SLOPE_THRESH = self.config.slope_thresh
        
        self.UNVOICED_PROP = self.config.unv_ratio # if more than 50% of pitches are unvoiced
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens

        # refine_onsets: a relocated boundary must hold for this many consecutive
        # pitch frames before we trust it (resists a lone vibrato / noise frame)
        self.ONSET_SUSTAIN = 3

        # threading variables
        self.nda_thread: threading.Thread = None
        self.stop_event = threading.Event()

    def update_config(self, config: Config):
        """update the config and all relevant parameters"""
        self.config = config
        self.w = self.config.w2
        self.hop = self.config.h2
        self.PITCH_THRESH = self.config.pitch_thresh
        self.SLOPE_THRESH = self.config.slope_thresh
        
        self.UNVOICED_PROP = self.config.unv_ratio # if more than 50% of pitches are unvoiced
        self.UNV_THRESH = self.config.unv_thresh # unvoiced pitches have unv_prob > sens

    def stop(self):
        if self.nda_thread and self.nda_thread.is_alive():
            self.stop_event.set()
            self.nda_thread.join() # pause the main thread until recording thread recognizes the stop event

    def get_slope(self, pitches: list[Pitch]):
        """get slope of all voiced pitches in the window"""
        # select only voiced x and y values
        mask  = np.array([p.unvoiced_prob < self.UNV_THRESH if p else False for p in pitches]) # boolean mask

        all_x = np.linspace(start=0, stop=len(pitches), num=len(pitches))
        x_voiced = all_x[mask]
        y_voiced = np.array([p.candidates[0][0] for p, m in zip(pitches, mask) if m])

        if x_voiced.size == 0:
            return 0.0, 0.0

        # get slope + intercept of only voiced pitches
        A = np.vstack([x_voiced, np.ones_like(x_voiced)]).T
        slope, intercept = np.linalg.lstsq(A, y_voiced, rcond=None)[0]

        return slope, intercept
    
    def is_unvoiced(self, unvoiced_probs: list[float]) -> bool:
        """returns whether the window is voiced or not
        based on whether the proportion of unvoiced pitches
        exceeds the UNVOICED_PROP threshold
        """
        arr = [p > self.UNV_THRESH for p in unvoiced_probs]
        if sum(arr) > self.UNVOICED_PROP*len(arr):
            return True
        return False
    
    def get_median_pitches(self, pitches: list[Pitch]):
        """return median pitches of whatever exists in the candidate
        slots for indices 0:2"""
        N = 3
        medians = [-1] * N

        # select only voiced frames
        voiced = [p for p in pitches if p and p.unvoiced_prob < self.UNV_THRESH]
        if not voiced:
            return medians

        # collect candidates in each column
        cols = [[] for _ in range(N)]

        for p in voiced:
            # p.candidates should be a list of (midi, prob)
            for i in range(min(N, len(p.candidates))):
                pitch_val = p.candidates[i][0]
                if pitch_val != -1:
                    cols[i].append(pitch_val)

        # compute medians
        for i in range(N):
            if cols[i]:
                medians[i] = float(np.median(cols[i]))

        return medians
        
    
    def handle_window(self, pitches: list[Pitch]):
        """
        returns key results about the window used for note processing
            (1) is_flat, (2) is_unv, (3) median_pitch, (4) start_time
        """
        unvoiced_probs = [p.unvoiced_prob if p else 1.0 for p in pitches]
        slope, _ = self.get_slope(pitches) 

        # key results
        is_flat = slope < self.SLOPE_THRESH
        is_unv = self.is_unvoiced(unvoiced_probs)
        med_pitches = self.get_median_pitches(pitches)
        
        # print(f"t({pitches[0].time:.4f}): slope({slope:.2f}), is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitches[0]:.2f})")
        
        return is_flat, is_unv, med_pitches
    
    def detect_notes(self, pitch_data: PitchData ) -> NoteData:
        """writes all notes completely offline"""
        nd = NoteData()
        prev_note = None
        prev_time = None
        # prev_good_time = None
        note_index = 0

        # iterate through all pitches
        for i in range(0, len(pitch_data.data)-self.w-1, self.hop):
            x = pitch_data.read(i=i, j=i+self.w, clean=False)

            if x[0] is None:
                continue
            
            t = x[0].time
            is_flat, is_unv, med_pitches = self.handle_window(x)

            if prev_note is None:
                if is_unv:
                    prev_note = [-1, -1, -1]
                elif is_flat:
                    prev_note = med_pitches
                prev_time = t
                # prev_good_time = t
            else:
                # if different enough...
                if abs(prev_note[0] - med_pitches[0]) > self.PITCH_THRESH:
                    if not is_flat and not is_unv:
                        # it's okay to not be 'flat' if unvoiced
                        # in the case it's not, skip to next window
                        # prev_time = t
                        continue
                    # and FLAT !
                    n = Note(
                        i=note_index,
                        start_time=prev_time, 
                        end_time=t,
                        midi_num=prev_note
                    )
                    nd.write_note(n)
                    # update iteration variables
                    prev_note = [-1, -1, -1] if is_unv else med_pitches
                    note_index += 1
                    # prev_good_time = t
                
                    prev_time = t

        # write the last note! :,)
        n = Note(
            i=i,
            start_time=prev_time,
            end_time=t,
            midi_num=prev_note
        )
        nd.write_note(n)

        # post-process: pull the hop-quantized boundaries onto the true onsets
        nd = self.refine_onsets(nd, pitch_data)
        return nd

    # ------------------------------------------------------------------ #
    # onset refinement (Method 1: pitch-transition + voicing fallback)
    # ------------------------------------------------------------------ #
    def _frame_pitch(self, p: Pitch):
        """primary midi of a frame, or None if the frame is missing / empty."""
        if p is None or not p.candidates:
            return None
        return p.candidates[0][0]

    def _frame_unvoiced(self, p: Pitch) -> float:
        """unvoiced prob of a frame; a missing frame counts as fully unvoiced."""
        return p.unvoiced_prob if p is not None else 1.0

    def _find_pitch_crossing(self, pitches: list[Pitch], lo: int, hi: int,
                             med_a: float, med_b: float) -> int | None:
        """voiced<->voiced: smallest frame in [lo, hi) where the per-frame pitch
        has crossed onto note B's side of the A/B midpoint and *stays* there for
        ONSET_SUSTAIN frames. returns the start of that run, or None."""
        midpoint = 0.5 * (med_a + med_b)
        going_up = med_b > med_a
        run, run_start = 0, None
        for k in range(lo, hi):
            v = self._frame_pitch(pitches[k])
            on_b = v is not None and ((v > midpoint) if going_up else (v < midpoint))
            if on_b:
                if run == 0:
                    run_start = k
                run += 1
                if run >= self.ONSET_SUSTAIN:
                    return run_start
            else:
                run, run_start = 0, None
        return None

    def _find_voicing_change(self, pitches: list[Pitch], lo: int, hi: int,
                             to_voiced: bool) -> int | None:
        """rest<->note: smallest frame in [lo, hi) where voicing flips to the
        target state and holds for ONSET_SUSTAIN frames. returns its start, or
        None."""
        run, run_start = 0, None
        for k in range(lo, hi):
            unv = self._frame_unvoiced(pitches[k])
            match = (unv < self.UNV_THRESH) if to_voiced else (unv >= self.UNV_THRESH)
            if match:
                if run == 0:
                    run_start = k
                run += 1
                if run >= self.ONSET_SUSTAIN:
                    return run_start
            else:
                run, run_start = 0, None
        return None

    def refine_onsets(self, note_data: NoteData, pitch_data: PitchData) -> NoteData:
        """relocate the hop-quantized note boundaries onto their true onsets.

        The detector reports every boundary at an h2-frame grid point (~55 ms).
        For each *shared* boundary between consecutive notes we search +-w pitch
        frames around it and move the split to single-frame (h1, ~3 ms) detail:
          - voiced<->voiced : first sustained crossing of the A/B pitch midpoint
          - rest<->note     : first sustained voicing change
          - rest<->rest     : left untouched (no cue)
        The original boundary is kept whenever no confident split is found, and
        the split is clamped strictly inside the pair so notes can't collapse.
        """
        notes = note_data.read(i=0, j=len(note_data.times))
        if len(notes) < 2:
            return note_data

        pitches = pitch_data.data
        n_frames = len(pitches)
        radius = self.w  # detection-window width; the true onset lies within it
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

            a_voiced, b_voiced = med_a != -1, med_b != -1
            if a_voiced and b_voiced:
                k = self._find_pitch_crossing(pitches, lo, hi, med_a, med_b)
            elif a_voiced != b_voiced:
                k = self._find_voicing_change(pitches, lo, hi, to_voiced=b_voiced)
            else:
                k = None  # rest<->rest: nothing to refine

            if k is None:
                continue

            new_t = k * frame_dt
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
    def detect_transitions(self, pitch_data: PitchData) -> None:
        """Mark every high-slope (pitch-transition) frame in `pitch_data`.

        Note detection deliberately skips windows whose pitch slope exceeds
        SLOPE_THRESH (slides between notes), but onset refinement then pulls those
        boundaries back so the transition frames end up *inside* a note's span.
        Left in, they drag a note's median pitch toward the neighbour it's sliding
        from/to (e.g. a slide up reads "too sharp").

        This pass slides the same detection window over the whole pitch track and
        sets `Pitch.is_transition = True` for any frame inside a window whose
        |slope| >= SLOPE_THRESH, else False. Downstream, update_alignment_distances
        leaves these frames' distances unset (grey) so they aren't scored.
        Call after detect_notes() (i.e. after refine_onsets) and before
        Recording.update_alignment_distances()."""
        pitches = pitch_data.data

        # default every present frame to non-transition first (idempotent re-runs)
        for p in pitches:
            if p is not None:
                p.is_transition = False

        # slide the detection window; flag all frames inside a high-slope window.
        # uses a dedicated small window (transitions are short-timescale events),
        # independent of the note-segmentation window self.w.
        w = 9
        h = 7
        slope_thresh = 0.75 / w
        for i in range(0, len(pitches) - w, h):
            window = pitches[i:i + w]
            if window[0] is None:
                continue
            slope, _ = self.get_slope(window)
            if abs(slope) >= slope_thresh:
                for p in window:
                    if p is not None:
                        p.is_transition = True

    def recompute_note_pitches(self, note_data: NoteData, pitch_data: PitchData) -> None:
        """Re-median each note's pitch over only its non-transition frames.

        A note's midi_num is the median of its frames, but onset refinement pulls
        high-slope slide frames into the note's span; for a descending slide those
        frames sit above the settled pitch and drag the median sharp (e.g. an F5
        reads F#5 -> a false 'too sharp' substitution). Excluding transition frames
        (flagged by detect_transitions) restores the settled pitch. A note with no
        non-transition voiced frames (rests, all-slide blips) keeps its original
        midi_num.

        Run after detect_transitions() and before detect_mistakes()."""
        for note in note_data.data.values():
            if note is None or note.midi_num[0] == -1:
                continue  # leave rests alone
            frames = pitch_data.read(
                start_time=note.start_time, end_time=note.end_time, clean=False
            )
            kept = [p for p in frames if p is not None and not p.is_transition]
            med = self.get_median_pitches(kept)
            if med[0] != -1:  # only overwrite when a voiced estimate survives
                note.midi_num = med



    def run(self, start_time: float=None):
        self.stop()
        self.stop_event.clear()
        self.recording.p2n_queue.init_start_time(start_time)
        self.nda_thread = threading.Thread(
            target=self._run, daemon=True
        )
        self.nda_thread.start()

    def _run(self) -> None:
        """the note detection algorithm for real time processing.

        an onset-based approach, where a window is an *onset* if
            - it's flat enough and voiced
            - or if it's mostly unvoiced
        
        and if a window is an onset,
        we compare it to the last valid onset
            - if it's different, it's a new note
            - if it's the same, it's not a new note
        """
        prev_note = None
        prev_time = None
        i = 0
        while not self.stop_event.is_set():
            try:
                x, t = self.recording.p2n_queue.pop(self.w, self.hop)
                if x is None or t < 0: # if invalid data read, skip frame
                    continue

                is_flat, is_unv, med_pitch = self.handle_window(x)

                # print(f"this window: is_flat({is_flat}), is_unv({is_unv}), med_pitch({med_pitch}), t({t})")

                # --- finding the first note phase ---
                if prev_note is None:
                    prev_note = -1 if is_unv else med_pitch
                    prev_time = t
                    continue

                # --- the second note and beyond ---
                if abs(prev_note - med_pitch) < self.PITCH_THRESH:
                    continue

                # ignore if the current window is unvoiced or flat
                prev_time = t
                if not is_flat and not is_unv:
                    # but still advance prev_time so we stay contiguous
                    # prev_note = -1 if is_unv else med_pitch
                    continue

                # ---> if we reach here, we have a NEW NOTE!
                # print(f"NEW NOTE! pitch={prev_note}, start={prev_time}, end={t}")
                n = Note(
                    i=i,
                    start_time=prev_time, 
                    end_time=t,
                    midi_num=prev_note
                )
                self.recording.note_data.write_note(n)
                i += 1

                # update iteration variables
                prev_note = -1 if is_unv else med_pitch
                self.note_detected.emit(n.start_time)

            except Exception as e:
                print(f"[NoteDetector] frame skipped due to error: {e}")
                continue
