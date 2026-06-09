import math

from PyQt6.QtCore import pyqtSignal, QObject

from app_logic.NoteData import Note, NoteData
from app_logic.user.ds.PitchData import Pitch, PitchData
from app_logic.user.ds.Recording import Recording
from app_logic.midi.ScoreData import ScoreData
from app_logic.Alignment import Mistake
from algorithms.Config import Config


# HMM_DETECTOR_MARKER_V1
REST = -1.0
_EPS = 1e-9
_LOG_EPS = math.log(_EPS)


class NoteDetector3(QObject):
    """Score-guided HMM note detector.

    Hidden state at frame t is (i, m): i is the score pointer into the active
    instrument's expected MIDI sequence, m is the user's produced pitch (-1 for
    rest). Per-frame observations are the existing Pitch.candidates probability
    list. A log-space Viterbi pass decodes the most likely (i, m) trajectory,
    which is then turned into both a NoteData segmentation and an alignment
    (pairs + mistakes) compatible with the existing Alignment contract.
    """
    note_detected = pyqtSignal(float)

    def __init__(self, recording: Recording = None, config: Config = None, parent: QObject | None = None):
        super().__init__(parent)
        self.recording = recording
        self.config = recording.config if recording else config

        c = self.config
        self.SELF_LOOP_LOGP = math.log(c.hmm_self_loop_prob)
        self.ADVANCE_LOGP = math.log(c.hmm_advance_prob)
        # advancing the score pointer onto a wrong pitch (substitution) is still allowed,
        # just at a fraction of the correct-advance probability
        self.ADVANCE_WRONG_LOGP = math.log(c.hmm_advance_prob * 0.2)
        self.SKIP_LOGP = math.log(c.hmm_skip_prob)
        self.INSERT_LOGP = math.log(c.hmm_insert_prob)
        self.MATCH_WIN = c.hmm_match_window
        self.BEAM = c.hmm_beam_width
        self.TOP_K = c.hmm_top_k
        self.TOLERANCE = c.tolerance

        self._alignment_pairs: list[tuple[Note | None, Note | None]] = []
        self._mistakes: list[Mistake] = []

    # ---------- public API ----------

    def detect_notes(self, pitch_data: PitchData = None) -> NoteData:
        if pitch_data is None:
            if not self.recording or not self.recording.pitch_data:
                print("[NoteDetector3] no pitch data available")
                return NoteData()
            pitch_data = self.recording.pitch_data

        pitches = [p for p in pitch_data.data if p is not None]
        if not pitches:
            print("[NoteDetector3] no valid pitches")
            return NoteData()

        expected = self._build_expected_sequence()

        path = self._viterbi(pitches, expected)
        note_data = self._path_to_notedata(path, pitches)
        self._alignment_pairs, self._mistakes = self._path_to_alignment(
            path, pitches, expected, note_data
        )

        for t in note_data.times:
            self.note_detected.emit(t)

        return note_data

    def get_alignment_result(self) -> tuple[list[tuple[Note | None, Note | None]], list[Mistake]]:
        return self._alignment_pairs, self._mistakes

    # ---------- state space ----------

    def _build_expected_sequence(self) -> list[Note]:
        if not self.recording or not self.recording.score_data:
            return []
        sd: ScoreData = self.recording.score_data
        nd = sd.note_datas.get(sd.active_instrument)
        if nd is None or not nd.times:
            return []
        return [nd.read_note(i=k) for k in range(len(nd.times))]

    def _state_pitches(self, i: int, pitch: Pitch, expected: list[Note]) -> list[float]:
        """Per-frame pitch set M_t for a given score pointer i."""
        seen: set = set()
        out: list[float] = []

        def add(m: float):
            key = round(m, 1)
            if key not in seen:
                seen.add(key)
                out.append(round(m, 2))

        if expected:
            if 0 <= i < len(expected):
                add(expected[i].midi_num[0])
            if 0 <= i + 1 < len(expected):
                add(expected[i + 1].midi_num[0])

        for c, _ in pitch.candidates[:3]:
            if c > 0:
                add(c)

        add(REST)
        return out

    # ---------- emission / transition ----------

    def _emit_logp(self, m: float, pitch: Pitch) -> float:
        if m < 0:  # rest state
            return math.log(max(pitch.unvoiced_prob, _EPS))
        voiced = max(1.0 - pitch.unvoiced_prob, _EPS)
        best_prob = 0.0
        for c, p in pitch.candidates:
            if abs(c - m) < self.MATCH_WIN and p > best_prob:
                best_prob = p
        if best_prob <= 0:
            return _LOG_EPS
        return math.log(voiced * best_prob + _EPS)

    def _trans_logp(self, i_prev: int, m_prev: float, i: int, m: float, expected: list[Note]) -> float:
        di = i - i_prev
        if di < 0 or di > 2:
            return _LOG_EPS

        same_m = (round(m, 1) == round(m_prev, 1))

        if di == 0:
            return self.SELF_LOOP_LOGP if same_m else self.INSERT_LOGP

        if di == 1:
            if expected and 0 <= i < len(expected):
                expected_m = expected[i].midi_num[0]
                if m >= 0 and abs(m - expected_m) < self.MATCH_WIN:
                    return self.ADVANCE_LOGP
            return self.ADVANCE_WRONG_LOGP

        # di == 2 — skipping one score note (deletion)
        return self.SKIP_LOGP

    # ---------- viterbi ----------

    def _prune(self, delta: dict) -> dict:
        if len(delta) <= self.TOP_K:
            return delta
        items = sorted(delta.items(), key=lambda kv: kv[1], reverse=True)
        return dict(items[:self.TOP_K])

    def _viterbi(self, pitches: list[Pitch], expected: list[Note]) -> list[tuple[int, float]]:
        n = len(pitches)
        N = len(expected)
        max_i = (N - 1) if N > 0 else 0

        delta_seq: list[dict] = [None] * n
        bp_seq: list[dict] = [None] * n

        # t = 0: uniform prior over a small initial set of i values
        delta: dict[tuple[int, float], float] = {}
        init_is = [0, 1] if N >= 2 else [0]
        for i0 in init_is:
            for m in self._state_pitches(i0, pitches[0], expected):
                key = (i0, m)
                e = self._emit_logp(m, pitches[0])
                if key not in delta or e > delta[key]:
                    delta[key] = e
        delta = self._prune(delta)
        delta_seq[0] = delta
        bp_seq[0] = {}

        for t in range(1, n):
            pitch_t = pitches[t]
            prev_items = list(delta.items())
            if not prev_items:
                break

            # build candidate i values for this frame: previous i's + advances, bounded by beam
            prev_is = sorted({k[0] for k, _ in prev_items})
            best_logp_by_i: dict[int, float] = {}
            for (ip, _mp), lp in prev_items:
                if ip not in best_logp_by_i or lp > best_logp_by_i[ip]:
                    best_logp_by_i[ip] = lp
            best_i = max(best_logp_by_i, key=lambda ip: best_logp_by_i[ip])

            cand_is_set: set = set()
            for ip in prev_is:
                cand_is_set.add(ip)
                cand_is_set.add(ip + 1)
                cand_is_set.add(ip + 2)
            cand_is = sorted(
                i for i in cand_is_set
                if abs(i - best_i) <= self.BEAM and (N == 0 or 0 <= i <= max_i)
            )

            new_delta: dict[tuple[int, float], float] = {}
            new_bp: dict[tuple[int, float], tuple[int, float]] = {}

            for i in cand_is:
                for m in self._state_pitches(i, pitch_t, expected):
                    e = self._emit_logp(m, pitch_t)
                    best_logp = -math.inf
                    best_prev = None
                    for prev_key, prev_logp in prev_items:
                        ip, prev_m = prev_key
                        tr = self._trans_logp(ip, prev_m, i, m, expected)
                        cand = prev_logp + tr + e
                        if cand > best_logp:
                            best_logp = cand
                            best_prev = prev_key
                    if best_prev is None or best_logp == -math.inf:
                        continue
                    key = (i, m)
                    if key not in new_delta or best_logp > new_delta[key]:
                        new_delta[key] = best_logp
                        new_bp[key] = best_prev

            if not new_delta:
                # dead end — sustain previous frame's frontier so Viterbi can continue
                new_delta = dict(delta)
                new_bp = {k: k for k in delta.keys()}

            new_delta = self._prune(new_delta)
            new_bp = {k: v for k, v in new_bp.items() if k in new_delta}

            delta = new_delta
            delta_seq[t] = delta
            bp_seq[t] = new_bp

        # find last frame with non-empty frontier
        last_t = n - 1
        while last_t >= 0 and not delta_seq[last_t]:
            last_t -= 1
        if last_t < 0:
            return []

        # backtrace from best final state
        final = delta_seq[last_t]
        best_key = max(final.keys(), key=lambda k: final[k])
        path_rev: list[tuple[int, float]] = [best_key]
        cur = best_key
        for t in range(last_t, 0, -1):
            prev = bp_seq[t].get(cur) if bp_seq[t] else None
            if prev is None:
                break
            path_rev.append(prev)
            cur = prev

        path = list(reversed(path_rev))
        # pad with last state if backtrace stopped short of n
        while len(path) < n:
            path.append(path[-1])
        return path[:n]

    # ---------- path -> outputs ----------

    def _path_to_notedata(self, path: list[tuple[int, float]], pitches: list[Pitch]) -> NoteData:
        note_data = NoteData()
        if not path:
            return note_data

        n = len(path)
        cur_m = path[0][1]
        cur_start = 0
        next_id = 0

        def emit(m: float, fs: int, fe: int):
            nonlocal next_id
            if m < 0:
                return
            start_time = pitches[fs].time
            end_time = pitches[fe].time if fe < len(pitches) else pitches[-1].time
            if end_time <= start_time:
                end_time = start_time + 1e-3
            note = Note(
                i=next_id,
                start_time=start_time,
                end_time=end_time,
                midi_num=[m],
            )
            note_data.write_note(note)
            next_id += 1

        for t in range(1, n):
            if path[t][1] != cur_m:
                emit(cur_m, cur_start, t - 1)
                cur_m = path[t][1]
                cur_start = t
        emit(cur_m, cur_start, n - 1)

        return note_data

    def _path_to_alignment(
        self,
        path: list[tuple[int, float]],
        pitches: list[Pitch],
        expected: list[Note],
        note_data: NoteData,
    ) -> tuple[list[tuple[Note | None, Note | None]], list[Mistake]]:
        pairs: list[tuple[Note | None, Note | None]] = []
        mistakes: list[Mistake] = []

        if not path:
            for note in expected:
                pairs.append((None, note))
                mistakes.append(Mistake(type="deletion", user_note=None, midi_note=note))
            return pairs, mistakes

        if not expected:
            # no score: just list all user notes as insertions
            for t in note_data.times:
                un = note_data.data[t]
                pairs.append((un, None))
                mistakes.append(Mistake(type="insertion", user_note=un, midi_note=None))
            return pairs, mistakes

        n = len(path)

        # chunks split on (m, i) changes — each chunk lives in exactly one i-run
        chunks: list[tuple[float, int, int, int]] = []  # (m, i, frame_start, frame_end)
        cur_m, cur_i = path[0][1], path[0][0]
        cur_start = 0
        for t in range(1, n):
            if path[t][1] != cur_m or path[t][0] != cur_i:
                chunks.append((cur_m, cur_i, cur_start, t - 1))
                cur_m, cur_i = path[t][1], path[t][0]
                cur_start = t
        chunks.append((cur_m, cur_i, cur_start, n - 1))

        # group consecutive non-rest chunks of same m into user notes (matches _path_to_notedata)
        user_notes_objs = [note_data.data[t] for t in note_data.times]
        note_obj_for_chunk: dict[int, Note] = {}
        note_idx = 0
        prev_m_for_note = None
        for c_idx, (m, _i, _fs, _fe) in enumerate(chunks):
            if m < 0:
                prev_m_for_note = None
                continue
            if prev_m_for_note is None or m != prev_m_for_note:
                if prev_m_for_note is not None:
                    note_idx += 1
                prev_m_for_note = m
            if note_idx < len(user_notes_objs):
                note_obj_for_chunk[c_idx] = user_notes_objs[note_idx]

        # i-runs from path
        i_runs: list[tuple[int, int, int]] = []  # (i, frame_start, frame_end)
        cur_i, cur_start = path[0][0], 0
        for t in range(1, n):
            if path[t][0] != cur_i:
                i_runs.append((cur_i, cur_start, t - 1))
                cur_i, cur_start = path[t][0], t
        i_runs.append((cur_i, cur_start, n - 1))

        consumed: set[int] = set()  # id() of user notes already paired
        last_i = -1

        for (i_val, fs, fe) in i_runs:
            # deletions for any expected indices we skipped over
            for skipped_idx in range(last_i + 1, i_val):
                if 0 <= skipped_idx < len(expected):
                    pairs.append((None, expected[skipped_idx]))
                    mistakes.append(Mistake(
                        type="deletion", user_note=None, midi_note=expected[skipped_idx]
                    ))

            # user notes whose chunks live inside this i-run
            notes_in_run: list[Note] = []
            seen_ids: set[int] = set()
            for c_idx, (_m, c_i, c_fs, c_fe) in enumerate(chunks):
                if c_i != i_val:
                    continue
                if c_fs < fs or c_fe > fe:
                    continue
                n_obj = note_obj_for_chunk.get(c_idx)
                if n_obj is None:
                    continue
                nid = id(n_obj)
                if nid in consumed or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                notes_in_run.append(n_obj)

            expected_note = expected[i_val] if 0 <= i_val < len(expected) else None

            if notes_in_run and expected_note is not None:
                for un in notes_in_run[:-1]:
                    pairs.append((un, None))
                    mistakes.append(Mistake(type="insertion", user_note=un, midi_note=None))
                    consumed.add(id(un))
                last_un = notes_in_run[-1]
                pairs.append((last_un, expected_note))
                if abs(last_un.midi_num[0] - expected_note.midi_num[0]) >= self.TOLERANCE:
                    mistakes.append(Mistake(
                        type="substitution", user_note=last_un, midi_note=expected_note
                    ))
                consumed.add(id(last_un))
            elif notes_in_run and expected_note is None:
                for un in notes_in_run:
                    pairs.append((un, None))
                    mistakes.append(Mistake(type="insertion", user_note=un, midi_note=None))
                    consumed.add(id(un))
            elif expected_note is not None:
                pairs.append((None, expected_note))
                mistakes.append(Mistake(
                    type="deletion", user_note=None, midi_note=expected_note
                ))

            last_i = i_val

        # trailing expected notes the path never reached
        for skipped_idx in range(last_i + 1, len(expected)):
            pairs.append((None, expected[skipped_idx]))
            mistakes.append(Mistake(
                type="deletion", user_note=None, midi_note=expected[skipped_idx]
            ))

        # user notes the path never attributed to any i-run (rare; e.g. notes during a
        # path stretch where score has ended) — surface as insertions
        for un in user_notes_objs:
            if id(un) not in consumed:
                pairs.append((un, None))
                mistakes.append(Mistake(type="insertion", user_note=un, midi_note=None))

        return pairs, mistakes
