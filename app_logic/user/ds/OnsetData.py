import threading
import numpy as np
import librosa
from bisect import bisect_left, bisect_right

from app_logic.user.ds.Recording import Recording
from algorithms.Config import Config


class OnsetData:
    """
    Stores detected onsets as sorted arrays of times (sec) and strengths.
    Read by index or by time window via binary search.
    """
 
    def __init__(self, config: Config):
        self.config = config
        self.times: np.ndarray = np.array([], dtype=np.float64)
        self.strengths: np.ndarray = np.array([], dtype=np.float64)
        self.lock = threading.Lock()
 
    # ---------- Mutation ----------
 
    def load(self, times: np.ndarray, strengths: np.ndarray = None):
        """Replace contents with a fresh set of onsets (must arrive sorted by time)."""
        with self.lock:
            self.times = np.asarray(times, dtype=np.float64)
            self.strengths = (np.asarray(strengths, dtype=np.float64)
                              if strengths is not None
                              else np.zeros_like(self.times))
 
    # ---------- Read ----------
 
    def read(self, start_time: float = None, end_time: float = None,
             i: int = None, j: int = None) -> np.ndarray:
        """Return onset times either by [start_time, end_time] window or by [i, j] index slice."""
        if i is not None and j is not None:
            return self.times[i:j]
        if start_time is not None and end_time is not None:
            lo = bisect_left(self.times, start_time)
            hi = bisect_right(self.times, end_time)
            return self.times[lo:hi]
        return self.times.copy()
 
    def read_strengths(self, start_time: float = None, end_time: float = None,
                       i: int = None, j: int = None) -> np.ndarray:
        if i is not None and j is not None:
            return self.strengths[i:j]
        if start_time is not None and end_time is not None:
            lo = bisect_left(self.times, start_time)
            hi = bisect_right(self.times, end_time)
            return self.strengths[lo:hi]
        return self.strengths.copy()
 
    def __len__(self) -> int:
        return len(self.times)
 
 
class OnsetDetector:
    """
    Wraps librosa.onset.onset_detect for use with AudioData.
 
    Usage
    -----
        detector = OnsetDetector(config)
        onsets = detector.detect(audio_data)
    """
 
    # Reasonable defaults for a violin practice context. Tune as needed.
    HOP_LENGTH = 512
    BACKTRACK = True       # snap each onset to the nearest preceding energy minimum
    DELTA = 0.07           # threshold above local mean to count as a peak
    WAIT = 10              # min frames between onsets (~10 * hop / sr seconds)
 
    def __init__(self, recording: Recording):
        self.recording = recording
        self.config = recording.config
 
        # Inspection artifacts
        self.onset_env_ = None    # the onset strength envelope
        self.onset_frames_ = None # raw frame indices returned by librosa
 
    def detect(self) -> OnsetData:
        """Run onset detection on `audio_data` and return a populated OnsetData."""
        onset_data = OnsetData(self.config)
        if self.recording.audio_data.end_index  == 0:
            return onset_data
 
        # snapshot the recorded portion of the buffer under the lock
        y = self.recording.audio_data.read_data(start_time=0)
        sr = self.config.sr

        # Onset strength envelope (frame-rate signal of "how much is starting now")
        self.onset_env_ = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.HOP_LENGTH
        )
 
        # Peak-pick onsets from the envelope
        self.onset_frames_ = librosa.onset.onset_detect(
            onset_envelope=self.onset_env_,
            sr=sr,
            hop_length=self.HOP_LENGTH,
            backtrack=self.BACKTRACK,
            delta=self.DELTA,
            wait=self.WAIT,
            units='frames',
        )
 
        times = librosa.frames_to_time(
            self.onset_frames_, sr=sr, hop_length=self.HOP_LENGTH
        )
        # Pull the envelope value at each onset frame as a "strength" score
        strengths = self.onset_env_[self.onset_frames_] if len(self.onset_frames_) else np.array([])
 
        onset_data.load(times, strengths)
        return onset_data