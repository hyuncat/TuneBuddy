import numpy as np
import threading
from PyQt6.QtCore import pyqtSignal, QObject

try:
    import ruptures as rpt
except ImportError:
    rpt = None

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch, PitchData
from app_logic.user.ds.Recording import Recording
from algorithms.Config import Config
from app_logic.midi.ScoreData import ScoreData


class NoteDetector2(QObject):
    """
    Alternative note detector using the `ruptures` library for change-point
    detection over the pitch series.

    Where the original NoteDetector uses an onset-based sliding window with
    slope/voicing heuristics, this treats note detection as a 1-D signal
    segmentation problem: a note is a contiguous run of frames between two
    consecutive change points in the MIDI-pitch series.

    Public API mirrors NoteDetector exactly:
      - __init__(recording=..., config=..., parent=...)  (+ algorithm kwargs)
      - detect_notes(pitch_data) -> NoteData             (offline)
      - run(start_time)                                  (real-time)
      - stop()
      - note_detected = pyqtSignal(float)                (start_time of new note)
    """
    note_detected = pyqtSignal(float)

    # ruptures algorithms exposed by name
    _ALGOS = {
        "pelt":   lambda model, min_size, jump: rpt.Pelt(model=model, min_size=min_size, jump=jump),
        "binseg": lambda model, min_size, jump: rpt.Binseg(model=model, min_size=min_size, jump=jump),
        "window": lambda model, min_size, jump: rpt.Window(model=model, min_size=min_size, jump=jump, width=max(2*min_size, 10)),
    }

    def __init__(self,
                 recording: Recording = None,
                 config: Config = None,
                 algo: str = "pelt",
                 model: str = "l2",
                 penalty: float = 3.0,
                 min_size: int = None,
                 jump: int = 1,
                 parent: QObject | None = None):
        """
        Parameters
        ----------
        recording, config : same as NoteDetector. One of them must be supplied.
        algo     : ruptures algorithm — "pelt" | "binseg" | "window".
        model    : cost model — "l1" | "l2" | "rbf" | "normal".
        penalty  : penalty for Pelt / Window. Larger => fewer change points.
                   (For Binseg this is repurposed as `pen` in `predict`.)
        min_size : minimum segment length in frames. Defaults to config.h2.
        jump     : ruptures `jump` parameter (frame stride). 1 = full resolution.
        """
        super().__init__(parent)
        if rpt is None:
            raise ImportError(
                "`ruptures` is required for NoteDetector2. "
                "Install it with `pip install ruptures`."
            )
        if recording is None and config is None:
            raise ValueError("Must provide either a recording or a config.")

        self.recording = recording
        self.config = recording.config if recording else config

        # shared note-detection thresholds (keep API parity w/ NoteDetector)
        self.PITCH_THRESH = self.config.pitch_thresh
        self.UNV_THRESH = self.config.unv_thresh

        # ruptures-specific params
        if algo not in self._ALGOS:
            raise ValueError(f"Unknown algo '{algo}'. Choose from {list(self._ALGOS)}.")
        self.algo_name = algo
        self.model = model
        self.penalty = float(penalty)
        self.min_size = int(min_size) if min_size is not None else max(2, self.config.h2)
        self.jump = max(1, int(jump))

        # threading
        self.nda_thread: threading.Thread = None
        self.stop_event = threading.Event()

    # ----------------------------------------------------------------- helpers

    def _pitch_to_signal(self, pitches: list[Pitch]):
        """
        Convert a list of Pitch frames -> (signal, unv_mask, times).

        - signal    : 1-D float array of MIDI numbers, with unvoiced/missing
                      frames forward-filled from the most recent valid pitch.
                      This avoids spurious change points at every voicing gap.
        - unv_mask  : boolean array, True where the original frame was
                      unvoiced or missing.
        - times     : list of frame timestamps (None for missing frames).
        """
        signal, unv_mask, times = [], [], []
        last_valid = 0.0
        for p in pitches:
            if p is None:
                signal.append(last_valid)
                unv_mask.append(True)
                times.append(None)
                continue

            unv = p.unvoiced_prob >= self.UNV_THRESH
            midi = p.candidates[0][0] if p.candidates else -1

            if unv or midi == -1:
                signal.append(last_valid)
                unv_mask.append(True)
            else:
                last_valid = float(midi)
                signal.append(last_valid)
                unv_mask.append(False)
            times.append(p.time)

        return np.asarray(signal, dtype=float), np.asarray(unv_mask), times

    def _segment_summary(self, signal: np.ndarray, unv_mask: np.ndarray,
                         start: int, end: int) -> list[float]:
        """
        Summarize frames [start, end) into the existing 3-slot midi_num format.
        Returns [-1, -1, -1] for unvoiced segments (>50% unvoiced frames).
        """
        if end <= start:
            return [-1, -1, -1]

        seg_unv = unv_mask[start:end]
        if seg_unv.mean() > 0.5:
            return [-1, -1, -1]

        seg = signal[start:end][~seg_unv]
        if seg.size == 0:
            return [-1, -1, -1]

        m = float(np.median(seg))
        return [m, m, m]

    def _detect_changepoints(self, signal: np.ndarray) -> list[int]:
        """
        Run the configured ruptures algorithm over `signal` and return the
        list of change-point indices. The list always ends with `len(signal)`.
        """
        n = signal.size
        if n < self.min_size * 2:
            return [n]

        algo = self._ALGOS[self.algo_name](self.model, self.min_size, self.jump).fit(signal)
        try:
            bkps = algo.predict(pen=self.penalty)
        except TypeError:
            # some algorithms (e.g. Binseg) may want n_bkps instead;
            # fall back to an estimate based on signal length / min_size
            n_bkps = max(1, n // (4 * self.min_size))
            bkps = algo.predict(n_bkps=n_bkps)
        return bkps

    # ----------------------------------------------------------------- offline

    def detect_notes(self, pitch_data: PitchData) -> NoteData:
        """
        Offline note detection via change-point segmentation of the MIDI series.
        """
        nd = NoteData()

        # take all valid (non-None) pitches in time order
        pitches = [p for p in pitch_data.data if p is not None]
        if not pitches:
            return nd

        signal, unv_mask, _ = self._pitch_to_signal(pitches)
        bkps = self._detect_changepoints(signal)

        prev = 0
        note_idx = 0
        for end in bkps:
            if end <= prev:
                continue

            mids = self._segment_summary(signal, unv_mask, prev, end)
            start_t = pitches[prev].time
            end_t = pitches[min(end - 1, len(pitches) - 1)].time

            # merge with previous note if pitch is within PITCH_THRESH —
            # ruptures occasionally emits adjacent boundaries on the same note
            last = nd.read_note(i=len(nd.times) - 1) if nd.times else None
            same_pitch = (
                last is not None
                and last.midi_num[0] != -1
                and mids[0] != -1
                and abs(last.midi_num[0] - mids[0]) <= self.PITCH_THRESH
            )
            if same_pitch:
                last.end_time = end_t
            else:
                n = Note(i=note_idx, start_time=start_t, end_time=end_t, midi_num=mids)
                nd.write_note(n)
                note_idx += 1
            prev = end

        return nd

    # --------------------------------------------------------------- real-time

    def stop(self):
        if self.nda_thread and self.nda_thread.is_alive():
            self.stop_event.set()
            self.nda_thread.join()

    def run(self, start_time: float = None):
        if self.recording is None:
            raise RuntimeError("NoteDetector2 needs a recording for real-time mode.")
        self.stop()
        self.stop_event.clear()
        self.recording.p2n_queue.init_start_time(start_time)
        self.nda_thread = threading.Thread(target=self._run, daemon=True)
        self.nda_thread.start()

    def _run(self) -> None:
        """
        Real-time change-point detection on a growing buffer.

        We pop windows from the p2n_queue (matching the original NoteDetector's
        cadence), append them to a buffer, and re-run ruptures. Boundaries
        before the most recent change point are considered 'confirmed' and
        flushed as Note objects; the final boundary (== len(buffer)) is left
        floating for the next iteration.
        """
        buffer_signal: list[float] = []
        buffer_unv: list[bool] = []
        buffer_t: list[float] = []
        last_confirmed = 0
        note_idx = 0

        w, h = self.config.w2, self.config.h2

        while not self.stop_event.is_set():
            try:
                x, _t = self.recording.p2n_queue.pop(w, h)
                if x is None or _t < 0:
                    continue

                sig, unv, ts = self._pitch_to_signal(x)
                buffer_signal.extend(sig.tolist())
                buffer_unv.extend(unv.tolist())
                buffer_t.extend(ts)

                arr = np.asarray(buffer_signal, dtype=float)
                unv_arr = np.asarray(buffer_unv)

                if arr.size < self.min_size * 2:
                    continue

                bkps = self._detect_changepoints(arr)
                # `bkps` ends with len(arr) — that boundary is provisional.
                stable = [b for b in bkps[:-1] if b > last_confirmed]

                for end in stable:
                    mids = self._segment_summary(arr, unv_arr, last_confirmed, end)
                    start_t = buffer_t[last_confirmed]
                    end_t = buffer_t[end - 1]
                    if start_t is None:
                        # forward-fill: walk forward to find a real timestamp
                        for k in range(last_confirmed, end):
                            if buffer_t[k] is not None:
                                start_t = buffer_t[k]
                                break
                    if end_t is None:
                        for k in range(end - 1, last_confirmed - 1, -1):
                            if buffer_t[k] is not None:
                                end_t = buffer_t[k]
                                break
                    if start_t is None or end_t is None:
                        last_confirmed = end
                        continue

                    n = Note(
                        i=note_idx,
                        start_time=start_t,
                        end_time=end_t,
                        midi_num=mids,
                    )
                    self.recording.note_data.write_note(n)
                    self.note_detected.emit(n.start_time)
                    note_idx += 1
                    last_confirmed = end

            except Exception as e:
                print(f"[NoteDetector2] frame skipped due to error: {e}")
                continue



class NoteDetector3(QObject):
    """
    Score-guided note detector (offline).

    Walks the pitch frames and the MIDI score notes with two pointers
    (j over frames, i_score over score notes). For each frame:
      1. The 'expected' MIDI is read from score[i_score].
      2. The pitch candidate closest in probability that lies within
         `prox_window` semitones of `expected` is chosen; otherwise we
         fall back to the most probable candidate.
      3. Segmentation is done string-edit-style:
            - chosen ~= expected           -> extend current segment
            - chosen ~= score[i_score+1]   -> user advanced; flush + i_score++
            - chosen far from both         -> insertion; flush, don't advance
            - sustained unvoiced           -> flush as rest
            - frame time > score[i_score].end_time + grace
              with no progress             -> deletion; advance i_score silently
            - chosen drifts from segment median by > pitch_thresh
                                           -> safety-net flush (score-blind)
    """

    note_detected = pyqtSignal(float)

    def __init__(self, recording: Recording = None, config: Config = None,
                 parent: QObject | None = None):
        super().__init__(parent)
        if recording is None and config is None:
            raise ValueError("Must provide either a recording or a config.")

        self.recording = recording
        self.config = recording.config if recording else config

        self.UNV_THRESH = self.config.unv_thresh
        self.PITCH_THRESH = self.config.pitch_thresh
        self.TOLERANCE = self.config.tolerance
        self.PROX_WINDOW = self.config.prox_window

        # frames of sustained-unvoiced before flushing the current segment
        self.MIN_UNV_RUN = max(2, self.config.h2 // 2)
        # extra time past a score note's end before we consider it 'deleted'
        self.DELETION_GRACE = 0.15  # seconds

    # ------------------------------------------------------------- selection

    def _pick_candidate(self, p: Pitch, expected: float | None) -> float:
        """
        Pick the MIDI value for frame `p` given an `expected` MIDI from the
        score (or None when no score note covers this frame).

        - Unvoiced -> -1
        - If `expected` is given, prefer the most-probable candidate within
          PROX_WINDOW semitones of expected.
        - Otherwise fall back to the top candidate.
        """
        if p is None or not p.candidates:
            return -1.0
        if p.unvoiced_prob > self.UNV_THRESH:
            return -1.0

        if expected is not None:
            # candidates are already sorted by descending probability
            for midi, _prob in p.candidates:
                if midi == -1:
                    continue
                if abs(midi - expected) <= self.PROX_WINDOW:
                    return float(midi)

        return float(p.candidates[0][0])

    # ----------------------------------------------------------- segmentation

    def _flush(self, nd: NoteData, note_idx: int,
               seg_pitches: list[float], seg_start_t: float,
               end_t: float) -> int:
        """Write the current segment to nd and return the next note_idx."""
        if seg_start_t is None or end_t is None or end_t <= seg_start_t:
            return note_idx

        if not seg_pitches or all(m == -1 for m in seg_pitches):
            midi_num = [-1.0, -1.0, -1.0]
        else:
            voiced = [m for m in seg_pitches if m != -1]
            m = float(np.median(voiced)) if voiced else -1.0
            midi_num = [m, m, m]

        n = Note(i=note_idx, start_time=seg_start_t,
                 end_time=end_t, midi_num=midi_num)
        nd.write_note(n)
        return note_idx + 1

    # ------------------------------------------------------------- detection

    def detect_notes(self, pitch_data: PitchData) -> NoteData:
        """
        Offline score-guided note detection.

        If no score is available on the recording, falls back to a
        score-blind pass that simply picks `candidates[0]` per frame.
        """
        nd = NoteData()
        frames: list[Pitch] = [p for p in pitch_data.data if p is not None]
        if not frames:
            return nd

        # pull score notes for the active instrument, if any
        score_notes: NoteData | None = None
        if self.recording is not None and self.recording.score_data is not None:
            sd: ScoreData = self.recording.score_data
            channel = self.recording.active_instrument
            score_notes = sd.note_datas.get(channel) if sd.note_datas else None
        n_score = len(score_notes.times) if score_notes else 0

        i_score = 0  # pointer into score_notes
        note_idx = 0

        seg_pitches: list[float] = []
        seg_start_t: float | None = None
        seg_match: int | None = None  # which score note this segment belongs to
        unv_run = 0  # consecutive unvoiced frames

        def expected_at(i: int) -> float | None:
            if score_notes is None or i >= n_score:
                return None
            return score_notes.read_note(i=i).midi_num[0]

        for p in frames:
            t = p.time
            expected = expected_at(i_score)
            next_expected = expected_at(i_score + 1)

            # time-based catch-up (deletion): advance score pointer past notes
            # whose window has clearly passed while we made no contact with them.
            while (score_notes is not None
                   and i_score < n_score
                   and t > score_notes.read_note(i=i_score).end_time + self.DELETION_GRACE
                   and (seg_match is None or seg_match != i_score)):
                i_score += 1
                expected = expected_at(i_score)
                next_expected = expected_at(i_score + 1)

            chosen = self._pick_candidate(p, expected)

            # --- start-of-stream ---
            if seg_start_t is None:
                seg_start_t = t
                seg_pitches = [chosen]
                if chosen != -1 and expected is not None and abs(chosen - expected) <= self.TOLERANCE:
                    seg_match = i_score
                else:
                    seg_match = None
                unv_run = 1 if chosen == -1 else 0
                continue

            # --- unvoiced frame ---
            if chosen == -1:
                unv_run += 1
                if unv_run >= self.MIN_UNV_RUN and any(m != -1 for m in seg_pitches):
                    # close out the previous voiced segment, start an unvoiced one
                    note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                    if seg_match is not None:
                        i_score = max(i_score, seg_match + 1)
                    seg_pitches = [-1.0]
                    seg_start_t = t
                    seg_match = None
                else:
                    seg_pitches.append(chosen)
                continue
            unv_run = 0

            # --- voiced frame: classify against expected / next expected ---
            close_to_curr = (expected is not None
                             and abs(chosen - expected) <= self.TOLERANCE)
            close_to_next = (next_expected is not None
                             and abs(chosen - next_expected) <= self.TOLERANCE)

            # running median of voiced pitches in the segment (safety net)
            voiced_seg = [m for m in seg_pitches if m != -1]
            seg_med = float(np.median(voiced_seg)) if voiced_seg else None
            drifted = (seg_med is not None
                       and abs(chosen - seg_med) > self.PITCH_THRESH)

            if close_to_curr and seg_match == i_score:
                # extending the current matched segment
                seg_pitches.append(chosen)
                continue

            if close_to_next:
                # user advanced -> flush, advance pointer, start fresh
                note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                i_score += 1
                seg_pitches = [chosen]
                seg_start_t = t
                seg_match = i_score  # now locked onto the new expected
                continue

            if close_to_curr:
                # segment wasn't matched yet but this frame matches expected;
                # close any in-progress mismatch (insertion) and lock on
                if seg_pitches and seg_med is not None and not drifted:
                    seg_pitches.append(chosen)
                    seg_match = i_score
                else:
                    note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                    seg_pitches = [chosen]
                    seg_start_t = t
                    seg_match = i_score
                continue

            # far from both expected and next_expected
            if drifted:
                # safety net: pitch boundary even without score guidance
                note_idx = self._flush(nd, note_idx, seg_pitches, seg_start_t, t)
                if seg_match is not None:
                    # we'd been tracking score[i_score]; that note is done
                    i_score = max(i_score, seg_match + 1)
                seg_pitches = [chosen]
                seg_start_t = t
                seg_match = None  # insertion until proven otherwise
            else:
                # noisy frame but not a boundary; just accumulate
                seg_pitches.append(chosen)

        # --- flush the final segment ---
        end_t = frames[-1].time
        self._flush(nd, note_idx, seg_pitches, seg_start_t, end_t)
        return nd

    # ----------------------------------------------------- real-time (unused)

    def run(self, start_time: float = None):
        raise NotImplementedError("NoteDetector3 is offline-only.")

    def stop(self):
        pass
