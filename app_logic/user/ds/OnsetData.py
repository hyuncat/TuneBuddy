import threading
import numpy as np
from bisect import bisect_left, bisect_right
from scipy.ndimage import gaussian_filter1d, median_filter
from typing import TYPE_CHECKING

from algorithms.Config import Config

if TYPE_CHECKING:
    from app_logic.user.ds.Recording import Recording


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
    Explicit log-spectral-flux onset detector for use with AudioData.

    Candidate generation is intentionally transparent: positive changes in a
    log-magnitude STFT become a novelty curve, a rolling median/MAD supplies a
    local adaptive baseline, and greedy non-maximum suppression keeps only
    strong, separated peaks. These are candidates for score-guided correction,
    not instructions to split every detected note.
 
    Usage
    -----
        detector = OnsetDetector(recording)
        onsets = detector.detect()
    """
 
    FRAME_LENGTH = 2048
    HOP_LENGTH = 256
    LOG_GAIN = 20.0
    FFT_BLOCK_FRAMES = 512
    SMOOTH_SIGMA_FRAMES = 1.0
 
    def __init__(self, recording: "Recording"):
        self.recording = recording
        self.config = recording.config
 
        # Inspection artifacts
        self.onset_env_ = None     # raw positive log-spectral flux
        self.onset_z_ = None       # locally standardized novelty
        self.onset_frames_ = None  # selected frame indices

    def update_config(self, config: Config):
        self.config = config
 
    def detect(self) -> OnsetData:
        """Run onset detection on `audio_data` and return a populated OnsetData."""
        onset_data = OnsetData(self.config)
        if self.recording.audio_data.end_index == 0:
            return onset_data
 
        # snapshot the recorded portion of the buffer under the lock
        y = self.recording.audio_data.read_all()
        sr = int(getattr(self.recording.audio_data, "sr", self.config.sr))
        t_origin = float(getattr(self.recording.audio_data, "t_origin", 0.0))

        y = np.asarray(y, dtype=np.float64)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if len(y) < self.FRAME_LENGTH:
            return onset_data

        self.onset_env_ = self._spectral_flux(y)
        self.onset_env_ = gaussian_filter1d(
            self.onset_env_,
            self.SMOOTH_SIGMA_FRAMES,
            mode="nearest",
        )

        frame_rate = sr / self.HOP_LENGTH
        adaptive_frames = max(
            3,
            int(round(self.config.onset_adaptive_window_sec * frame_rate)) | 1,
        )
        baseline = median_filter(
            self.onset_env_,
            size=adaptive_frames,
            mode="nearest",
        )
        deviation = median_filter(
            np.abs(self.onset_env_ - baseline),
            size=adaptive_frames,
            mode="nearest",
        )
        nonzero_deviation = deviation[deviation > np.finfo(float).eps]
        deviation_floor = (
            0.05 * float(np.median(nonzero_deviation))
            if len(nonzero_deviation)
            else np.finfo(float).eps
        )
        self.onset_z_ = (self.onset_env_ - baseline) / np.maximum(
            deviation,
            max(deviation_floor, np.finfo(float).eps),
        )

        self.onset_frames_ = self._pick_peaks(self.onset_z_, frame_rate)
        # Spectral flux is a forward difference between STFT frames, so the
        # frame-start timestamp is the least-latent boundary estimate. It also
        # avoids the opaque energy backtracking that used to move candidates by
        # unrelated amounts.
        times = (
            self.onset_frames_ * self.HOP_LENGTH / sr
            + t_origin
        )
        strengths = self.onset_z_[self.onset_frames_]
 
        onset_data.load(times, strengths)
        return onset_data

    def _spectral_flux(self, y: np.ndarray) -> np.ndarray:
        """Positive log-spectral flux in bounded-memory FFT blocks."""
        n_frames = 1 + (len(y) - self.FRAME_LENGTH) // self.HOP_LENGTH
        flux = np.zeros(n_frames, dtype=np.float64)
        window = np.hanning(self.FRAME_LENGTH)
        previous = None
        for first in range(0, n_frames, self.FFT_BLOCK_FRAMES):
            count = min(self.FFT_BLOCK_FRAMES, n_frames - first)
            sample_start = first * self.HOP_LENGTH
            sample_end = (
                sample_start
                + (count - 1) * self.HOP_LENGTH
                + self.FRAME_LENGTH
            )
            frames = np.lib.stride_tricks.sliding_window_view(
                y[sample_start:sample_end],
                self.FRAME_LENGTH,
            )[::self.HOP_LENGTH][:count]
            magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
            spectrum = np.log1p(self.LOG_GAIN * magnitude)
            if previous is not None:
                flux[first] = np.mean(np.maximum(spectrum[0] - previous, 0.0))
            if count > 1:
                flux[first + 1:first + count] = np.mean(
                    np.maximum(np.diff(spectrum, axis=0), 0.0),
                    axis=1,
                )
            previous = spectrum[-1]
        return flux

    def _pick_peaks(self, novelty_z: np.ndarray, frame_rate: float) -> np.ndarray:
        """Local maxima above threshold, greedily separated by confidence."""
        if len(novelty_z) < 3:
            return np.array([], dtype=np.int64)
        candidates = np.flatnonzero(
            (novelty_z[1:-1] >= novelty_z[:-2])
            & (novelty_z[1:-1] > novelty_z[2:])
            & (novelty_z[1:-1] >= self.config.onset_z_threshold)
        ) + 1
        min_frames = max(
            1,
            int(round(self.config.onset_min_spacing_sec * frame_rate)),
        )
        accepted = []
        for candidate in candidates[np.argsort(novelty_z[candidates])[::-1]]:
            if all(abs(int(candidate) - other) >= min_frames for other in accepted):
                accepted.append(int(candidate))
        return np.asarray(sorted(accepted), dtype=np.int64)
