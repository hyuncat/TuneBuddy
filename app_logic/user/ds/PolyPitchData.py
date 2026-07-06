import threading
from math import floor, ceil

import numpy as np

from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch


class PolyPitch(Pitch):
    """Detector-side pitch frame whose candidates are SIMULTANEOUS pitches
    (chord members, salience-ordered) rather than competing hypotheses for a
    single voice. Each candidate's prob is its share of the frame's harmonic
    salience (shares sum to 1)."""
    polyphonic = True


class _PolyFrame:
    """Write-through view of one PolyPitchData frame.

    Duck-types Pitch so existing consumers (GuitarHero, practice mode,
    Recording's t_origin lift / distance passes) can read AND mutate frames in
    place while the columnar arrays underneath stay the single source of truth.
    """

    polyphonic = True
    __slots__ = ("_pd", "_i")

    def __init__(self, pd: "PolyPitchData", i: int):
        self._pd = pd
        self._i = i

    @property
    def config(self) -> Config:
        return self._pd.config

    @property
    def time(self) -> float:
        return float(self._pd._times[self._i])

    @time.setter
    def time(self, v: float):
        self._pd._times[self._i] = v

    @property
    def volume(self) -> float:
        return float(self._pd._vol[self._i])

    @volume.setter
    def volume(self, v: float):
        self._pd._vol[self._i] = v

    @property
    def unvoiced_prob(self) -> float:
        return float(self._pd._unv[self._i])

    @unvoiced_prob.setter
    def unvoiced_prob(self, v: float):
        self._pd._unv[self._i] = v

    @property
    def value(self) -> float:
        """Strongest simultaneous pitch (-1 = unvoiced), mirroring Pitch.value."""
        return float(self._pd._midi[self._i, 0])

    @value.setter
    def value(self, v: float):
        self._pd._midi[self._i, 0] = v

    @property
    def candidate_pitches(self) -> list[tuple[float, float]]:
        midi = self._pd._midi[self._i]
        sal = self._pd._sal[self._i]
        return [(float(m), float(s)) for m, s in zip(midi, sal) if m != -1]

    @candidate_pitches.setter
    def candidate_pitches(self, cands):
        self._pd._store_candidates(self._i, list(cands or []))

    candidates = candidate_pitches  # legacy alias, like Pitch

    @property
    def live_distance(self):
        d = self._pd._live[self._i]
        return None if np.isnan(d) else float(d)

    @live_distance.setter
    def live_distance(self, v):
        self._pd._live[self._i] = np.nan if v is None else v

    distance = live_distance  # legacy alias, like Pitch

    @property
    def aligned_distance(self):
        d = self._pd._aligned[self._i]
        return None if np.isnan(d) else float(d)

    @aligned_distance.setter
    def aligned_distance(self, v):
        self._pd._aligned[self._i] = np.nan if v is None else v

    align_distance = aligned_distance  # legacy alias, like Pitch

    @property
    def is_transition(self):
        t = self._pd._trans[self._i]
        return None if t < 0 else bool(t)

    @is_transition.setter
    def is_transition(self, v):
        self._pd._trans[self._i] = -1 if v is None else int(bool(v))

    def set_transition(self, value: bool):
        self.is_transition = value

    def get_note_name(self) -> str:
        return self.config.get_note_name(self.value)

    def ensure_compatible(self, config: Config | None = None) -> "_PolyFrame":
        """Array-backed frames are always current-format; nothing to patch."""
        return self


class PolyPitchData:
    """PitchData counterpart for polyphonic frames.

    Same surface as PitchData (data / load / write / read / read_pitch /
    is_voiced_pitch / get_voiced_range / resize / t_origin / lock), but the
    core store is columnar numpy — (n_frames x max_voices) midi + salience
    planes plus per-frame volume / unvoiced / distance / transition arrays —
    so K simultaneous pitches read and write as one row instead of K objects,
    and whole-window reads vectorize (see read_arrays).

    `.data` is a property for compatibility with the object-per-frame
    consumers: the getter materializes write-through _PolyFrame views (None
    for unwritten frames, like the vanilla None-holes) and the setter
    bulk-ingests any Pitch-like sequence (detector output, cache loads, or
    slices of our own views, e.g. trim_end).
    """

    def __init__(self, config: Config):
        self.config = config

        # App-time (sec) that frame 0 represents; same contract as PitchData
        self.t_origin = 0.0
        self.time_to_index = lambda sec: floor((sec - self.t_origin)*(self.config.sr / self.config.h1))

        self.UNVOICED_THRESHOLD = config.unv_thresh
        self.max_voices = max(1, int(config.poly_max_voices))

        # RLock (not Lock): trim/save paths hold .lock around a `.data = ...`
        # reassignment and the setter locks again on the same thread
        self.lock = threading.RLock()

        DEFAULT_LENGTH = 60  # (sec)
        self._init_arrays(ceil(self.time_to_index(DEFAULT_LENGTH)))

    # --- columnar core ---
    def _init_arrays(self, n: int):
        K = self.max_voices
        self._times = np.zeros(n, dtype=np.float64)
        self._midi = np.full((n, K), -1.0, dtype=np.float32)
        self._sal = np.zeros((n, K), dtype=np.float32)
        self._vol = np.zeros(n, dtype=np.float32)
        self._unv = np.ones(n, dtype=np.float32)
        self._live = np.full(n, np.nan, dtype=np.float64)
        self._aligned = np.full(n, np.nan, dtype=np.float64)
        self._trans = np.full(n, -1, dtype=np.int8)  # -1 None / 0 False / 1 True
        self._written = np.zeros(n, dtype=bool)

    @property
    def capacity(self) -> int:
        return len(self._written)

    def _store_candidates(self, i: int, cands):
        K = self.max_voices
        self._midi[i] = -1.0
        self._sal[i] = 0.0
        for k, (m, s) in enumerate(cands[:K]):
            self._midi[i, k] = m
            self._sal[i, k] = s

    @staticmethod
    def _extract(p) -> tuple:
        """One frame's values off any Pitch-like object (detector output, mono
        Pitch, or one of our own views mid-reassignment)."""
        return (
            float(getattr(p, "time", 0.0)),
            list(getattr(p, "candidate_pitches", None) or []),
            float(getattr(p, "volume", 0.0)),
            float(getattr(p, "unvoiced_prob", 1.0)),
            getattr(p, "live_distance", None),
            getattr(p, "aligned_distance", None),
            getattr(p, "is_transition", None),
        )

    def _store_row(self, i: int, row: tuple):
        time, cands, vol, unv, live, aligned, trans = row
        self._written[i] = True
        self._times[i] = time
        self._store_candidates(i, cands)
        self._vol[i] = vol
        self._unv[i] = unv
        self._live[i] = np.nan if live is None else live
        self._aligned[i] = np.nan if aligned is None else aligned
        self._trans[i] = -1 if trans is None else int(bool(trans))

    # --- PitchData-compatible surface ---
    @property
    def data(self) -> list:
        return [
            _PolyFrame(self, i) if written else None
            for i, written in enumerate(self._written)
        ]

    @data.setter
    def data(self, pitches):
        # snapshot BEFORE swapping arrays so slices of our own views (which
        # read lazily from the old arrays) survive the reassignment
        rows = [None if p is None else self._extract(p) for p in pitches]
        with self.lock:
            self._init_arrays(len(rows))
            for i, row in enumerate(rows):
                if row is not None:
                    self._store_row(i, row)

    def load(self, pitches: list):
        """load in an entire pitch array"""
        self.data = pitches

    def resize(self, resize_factor=2):
        """increase the capacity of the current frame arrays"""
        with self.lock:
            extra = self.capacity * max(1, resize_factor - 1)

            def pad(arr, fill):
                block = np.full((extra,) + arr.shape[1:], fill, dtype=arr.dtype)
                return np.concatenate([arr, block])

            self._times = pad(self._times, 0.0)
            self._midi = pad(self._midi, -1.0)
            self._sal = pad(self._sal, 0.0)
            self._vol = pad(self._vol, 0.0)
            self._unv = pad(self._unv, 1.0)
            self._live = pad(self._live, np.nan)
            self._aligned = pad(self._aligned, np.nan)
            self._trans = pad(self._trans, -1)
            self._written = pad(self._written, False)

    def write(self, pitches, start_time: float = 0):
        """write the pitches to the arrays at the given time index"""
        if not isinstance(pitches, (list, tuple)):
            pitches = [pitches]
        if not pitches:
            return
        if not start_time:
            start_time = pitches[0].time

        i = self.time_to_index(start_time)
        j = i + len(pitches)

        if j > self.capacity * 0.8:  # if close enough to end
            self.resize()

        with self.lock:
            for k, p in enumerate(pitches):
                idx = i + k
                if p is None or idx < 0 or idx >= self.capacity:
                    continue
                self._store_row(idx, self._extract(p))

    def read(self, start_time: float = 0, end_time: float = 0, i: int = None, j: int = None,
             clean=False, include_transitions: bool = True) -> list:
        """returns the frames corresponding to start_time <--> end_time"""
        if not i and not j:
            i = max(0, self.time_to_index(start_time))
            j = min(self.time_to_index(end_time), self.capacity - 1)
        lo = max(0, i)
        hi = max(lo, min(j, self.capacity))

        if clean:
            mask = self._voiced_mask(include_transitions=include_transitions)
            return [_PolyFrame(self, k) for k in range(lo, hi) if mask[k]]

        return [
            _PolyFrame(self, k) if self._written[k] else None
            for k in range(lo, hi)
        ]

    def read_pitch(self, start_time: float = 0):
        """returns the closest frame to the start_time"""
        i = self.time_to_index(start_time)
        if i < 0 or i >= self.capacity or not self._written[i]:
            return None
        return _PolyFrame(self, i)

    def read_arrays(self, start_time: float = 0, end_time: float = 0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized window read: (times[N], midi[N,K], salience[N,K],
        voiced[N]). The fast path for plotting / notebook sweeps — no
        per-frame objects; unvoiced slots hold midi -1 / salience 0."""
        i = max(0, self.time_to_index(start_time))
        j = max(i, min(self.time_to_index(end_time), self.capacity))
        sl = slice(i, j)
        return (
            self._times[sl].copy(),
            self._midi[sl].copy(),
            self._sal[sl].copy(),
            self._voiced_mask()[sl],
        )

    def _voiced_mask(self, include_transitions: bool = True) -> np.ndarray:
        mask = (
            self._written
            & (self._midi[:, 0] != -1)
            & (self._unv < self.UNVOICED_THRESHOLD)
        )
        if not include_transitions:
            mask &= self._trans != 1
        return mask

    def is_voiced_pitch(self, pitch, include_transitions: bool = True) -> bool:
        if pitch is not None:
            pitch.ensure_compatible(self.config)
        return (
            pitch is not None
            and pitch.value != -1
            and pitch.unvoiced_prob < self.UNVOICED_THRESHOLD
            and (include_transitions or not getattr(pitch, "is_transition", False))
        )

    def get_voiced_range(self, include_transitions: bool = True) -> tuple[float, float]:
        """Return the app-time range covered by voiced pitch frames."""
        idx = np.flatnonzero(self._voiced_mask(include_transitions=include_transitions))
        if idx.size == 0:
            return None
        frame_dt = self.config.h1 / self.config.sr
        return float(self._times[idx[0]]), float(self._times[idx[-1]] + frame_dt)
