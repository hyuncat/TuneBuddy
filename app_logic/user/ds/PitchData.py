import threading
from math import floor, ceil
import numpy as np

from algorithms.Config import Config

class Pitch:
    def __init__(self, time: float, volume: float, unvoiced_prob: float,
                 live_distance: float, config: Config,
                 candidates: list[tuple[float, float]] | None=None, value: float=-1):
        """
        The quintessential pitch object for the app. Corresponds to a given [time] in the
        PitchData and stores all possible pitch [candidates] = [(midi_num, prob), ...]
        sorted from most --> least probable as well as the volume and a reference to the
        settings (config) with which it was computed
        """
        # --- essential --- #
        candidates = [
            (float(midi), float(prob))
            for midi, prob in (candidates or [])
        ]
        self.config = config  # tuning / fmin/fmax / note naming
        self.time = time      # center time of the frame when pitch was computed
        self.value = candidates[0][0] if value == -1 and candidates else value
        self.volume = volume  # mean |amplitude| of the frame

        # ---> pre-smoother characteristics
        self.unvoiced_prob = unvoiced_prob # how messy the signal was (no clean periodicity)
        self.candidate_pitches = candidates # [(midi_num, prob), ...]; descending probability
        # --- used in note detection --- #
        self.is_transition = None

        # --- alignment-based --- #
        self.live_distance = live_distance  # distance to target note, filled in live
        self.aligned_distance = None # post-analysis distance

    def get_note_name(self) -> str:
        return self.config.get_note_name(self.value)

    def set_transition(self, value: bool):
        self.is_transition = value

    @property
    def candidates(self) -> list[tuple[float, float]]:
        """Backward-compatible alias for pre-rename caches/code."""
        return self.__dict__.get(
            "candidate_pitches",
            self.__dict__.get("candidates", []),
        )

    @candidates.setter
    def candidates(self, value: list[tuple[float, float]] | None):
        self.__dict__["candidate_pitches"] = [
            (float(midi), float(prob))
            for midi, prob in (value or [])
        ]

    @property
    def distance(self):
        return self.__dict__.get("live_distance", self.__dict__.get("distance"))

    @distance.setter
    def distance(self, value):
        self.__dict__["live_distance"] = value

    @property
    def align_distance(self):
        return self.__dict__.get(
            "aligned_distance",
            self.__dict__.get("align_distance"),
        )

    @align_distance.setter
    def align_distance(self, value):
        self.__dict__["aligned_distance"] = value

    def ensure_compatible(self, config: Config | None = None) -> "Pitch":
        """Patch legacy pickled Pitch instances in-place.

        Older caches used `candidates`, `distance`, and `align_distance`; the
        current app uses `candidate_pitches`, `value`, `live_distance`, and
        `aligned_distance`.
        """
        if config is not None:
            self.config = config
        elif not hasattr(self, "config") or self.config is None:
            self.config = Config()

        if not hasattr(self, "candidate_pitches"):
            self.candidate_pitches = [
                (float(midi), float(prob))
                for midi, prob in (self.__dict__.get("candidates", []) or [])
            ]
        else:
            self.candidate_pitches = [
                (float(midi), float(prob))
                for midi, prob in (self.candidate_pitches or [])
            ]

        if not hasattr(self, "value"):
            self.value = self.candidate_pitches[0][0] if self.candidate_pitches else -1
        if not hasattr(self, "live_distance"):
            self.live_distance = self.__dict__.get("distance")
        if not hasattr(self, "aligned_distance"):
            self.aligned_distance = self.__dict__.get("align_distance")
        if not hasattr(self, "is_transition"):
            self.is_transition = None
        if not hasattr(self, "volume"):
            self.volume = 0.0
        if not hasattr(self, "unvoiced_prob"):
            self.unvoiced_prob = 1.0
        if not hasattr(self, "time"):
            self.time = 0.0
        return self

class PitchData:
    def __init__(self, config: Config):
        """
        an audio data-like pitch data: an array of pitches
	       - indexable using the (SR / hop size)
        """
        # reference to the global config (for sr + hop size used in pitch detection)
        self.config = config

        # App-time (sec) that frame 0 represents (mirrors AudioData.t_origin). A
        # Perform take's one-beat runway records from a NEGATIVE app-time; t_origin
        # keeps the frame array 0-indexed while time<->index stays in app-time.
        self.t_origin = 0.0

        # the essential time to index lambda
        self.time_to_index = lambda sec: floor((sec - self.t_origin)*(self.config.sr / self.config.h1))
        
        DEFAULT_LENGTH = 60 # (sec)
        self.data: list[Pitch] = [None] * ceil(self.time_to_index(DEFAULT_LENGTH))
        self.lock = threading.Lock()

        self.UNVOICED_THRESHOLD = config.unv_thresh # threshold above which a pitch is considered unvoiced

    def resize(self, resize_factor=2):
        """increase the capacity of the current pitch array"""
        with self.lock:
            new_data = [None] * (len(self.data) * resize_factor)
            self.data.extend(new_data)

    def load(self, pitches: list[Pitch]):
        """load in an entire pitch array"""
        self.data = [
            p.ensure_compatible(self.config) if p is not None else None
            for p in pitches
        ]

    def write(self, pitches: list[Pitch] | Pitch, start_time: float=0):
        """write the pitches to the data at the given time index"""
        if isinstance(pitches, Pitch):
            pitches = [pitches]
        pitches = [
            p.ensure_compatible(self.config) if p is not None else None
            for p in pitches
        ]
        if not start_time:
            start_time = pitches[0].time

        # get indices into data array
        i = self.time_to_index(start_time)
        j = i+len(pitches)

        if j > len(self.data)*0.8: # if close enough to end
            self.resize()

        with self.lock:
            self.data[i:j] = pitches

    def read(self, start_time: float=0, end_time: float=0, i: int=None, j: int=None, 
             clean=False, include_transitions: bool=True) -> list[Pitch]:
        """returns the array of pitches corresponding to start_time <--> end_time"""
        if not i and not j:
            i = max(0, self.time_to_index(start_time))
            j = min(self.time_to_index(end_time), len(self.data)-1)

        if clean:
            return [p for p in self.data[i:j] if self.is_voiced_pitch(p, include_transitions=include_transitions)]

        return self.data[i:j]
    
    def read_pitch(self, start_time: float=0) -> Pitch:
        """returns the closest pitch to the start_time"""
        i = self.time_to_index(start_time)
        if i < 0 or i >= len(self.data):
            return None
        return self.data[i]

    def is_voiced_pitch(self, pitch: Pitch, include_transitions: bool=True) -> bool:
        if pitch is not None:
            pitch.ensure_compatible(self.config) # patches legacy pitch caches in-place
        return (
            pitch is not None
            and pitch.value != -1
            and pitch.unvoiced_prob < self.UNVOICED_THRESHOLD
            and (include_transitions or not getattr(pitch, "is_transition", False))
        )

    def get_voiced_range(self, include_transitions: bool=True) -> tuple[float, float]:
        """Return the app-time range covered by voiced pitch frames."""
        first = None
        last = None
        for pitch in self.data:
            if self.is_voiced_pitch(
                pitch,
                include_transitions=include_transitions,
            ):
                first = pitch
                break
        for pitch in reversed(self.data):
            if self.is_voiced_pitch(
                pitch,
                include_transitions=include_transitions,
            ):
                last = pitch
                break
        if first is None or last is None:
            return None
        frame_dt = self.config.h1 / self.config.sr
        return first.time, last.time + frame_dt
