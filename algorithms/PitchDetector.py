import numpy as np
from scipy.signal import find_peaks, iirfilter, sosfilt
from scipy.stats import beta
import threading
import time
from PyQt6.QtCore import QObject, pyqtSignal
from tqdm import tqdm

from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording
from algorithms.Config import Config
from algorithms.CQT import CQT

class PitchDetector(QObject):

    pitch_detected = pyqtSignal(float)
    # offline (whole-file) detection: per-phase status text + a done signal
    status_changed = pyqtSignal(str)
    detection_finished = pyqtSignal()

    def __init__(self, recording: Recording=None, config: Config=None, parent: QObject|None=None):
        """
        Initialize the pitch detection parameters, like the tuning, frequency range, etc.
        Best to make it as specific as possible to your desired use case to improve accuracy of the detection.
        """
        super().__init__(parent)
        if not recording and not config:
            raise ValueError("Must provide either a recording or a config to initialize the PitchDetector.")
        self.recording = recording
        self.config = config if config else recording.config
        self.SR = self.config.sr # for sample-to-frequency conversion

        # --- pitch config variables ---
        # ensure max lag is big enough to detect lowest f0 (largest period)
        # defaults to score's min/max pitch range, can be overridden
        self.PADDING = 4.0 # in semitones
        padded_fmin = self.config.midi_to_freq(self.config.freq_to_midi(self.config.fmin) - self.PADDING)
        padded_fmax = self.config.midi_to_freq(self.config.freq_to_midi(self.config.fmax) + self.PADDING)
        self.tau_max = int(self.config.sr / padded_fmin)
        self.tau_min = int(self.config.sr / padded_fmax)

        # initialize beta distribution parameters
        self.UNVOICED_PROB = 0.01
        self.N_THRESHOLDS = 100
        self.beta_pdf, self.thresholds = self.threshold_prior(n_thresholds=self.N_THRESHOLDS)

        # rolling window variables (for detect_pitches)
        self.FRAME_SIZE = self.config.w1
        self.HOP_SIZE = self.config.h1

        # threading variables
        self.pda_thread: threading.Thread = None
        self.offline_thread: threading.Thread = None  # for detect_pitches_async
        self.stop_event = threading.Event()
        self._drain_on_stop = False
        self._stream_volume_peak = 0.0
        self._last_volume_gate_stats: dict[str, float | int] = {}
        self.cqt = CQT(self.config)
        self._cqt_frame_index = 0
        self._cqt_offline = False

        # block variable for stalling buffer
        self.block = False

        # Cached preprocessing SOS (see pitch_bandpass_filter): the Butterworth
        # design is constant per Config, but preprocess_audio filters every
        # frame.
        self._preprocess_sos = None
        self._preprocess_filter_key = None

    # might not be necessary anymore :)
    def load_config(self, config: Config):
        """re-initialize the tuning parameters"""
        self.config = config
        self.SR = config.sr
        padded_fmin = self.config.midi_to_freq(self.config.freq_to_midi(self.config.fmin) - self.PADDING)
        padded_fmax = self.config.midi_to_freq(self.config.freq_to_midi(self.config.fmax) + self.PADDING)
        self.tau_max = int(self.SR / padded_fmin)
        self.tau_min = int(self.SR / padded_fmax)

        # rolling window variables (for detect_pitches)
        self.FRAME_SIZE = config.w1
        self.HOP_SIZE = config.h1
        self._stream_volume_peak = 0.0
        self._last_volume_gate_stats = {}
        self.cqt = CQT(config)
        self._cqt_frame_index = 0
        self._cqt_offline = False

    def run(self, start_time: float=None):
        """keep trying to detect pitches while we can"""
        self.stop()
        self.stop_event.clear()
        self._drain_on_stop = False
        self._stream_volume_peak = 0.0
        self._cqt_frame_index = 0
        self.recording.a2p_queue.init_start_time(start_time)
        self.pda_thread = threading.Thread(
            target=self._run, daemon=True
        )
        self.pda_thread.start()
    
    def _run(self) -> None:
        while True:
            try:
                stopping = self.stop_event.is_set()
                if stopping and not self._drain_on_stop:
                    break
                # if self.block:
                    # self.recording.a2p_queue.stall(self.HOP_SIZE) # occurs when in practice mode and you fuck up
                x, t = self.recording.a2p_queue.pop(self.FRAME_SIZE, self.HOP_SIZE, stall=self.block)

                if x is None: # returns none if not enough to detect
                    if stopping:
                        break
                    self.stop_event.wait(0.002)
                    continue
                # x, t = x[0], x[1]
                pitch = self.detect_pitch(x, t)
                # Buffer addresses frames by their start so it can advance on
                # the h1 grid, but a pitch describes the center of its analysis
                # window (the convention already used by offline detection).
                pitch.time = t + 0.5 * self.FRAME_SIZE / self.SR
                self.recording.write_pitch_data([pitch], t)
                # print(f'detected pitch @ {pitch.time}, midi_num: {pitch.candidate_pitches[0][0]}, unvoiced_prob: {pitch.unvoiced_prob}')
                # Practice reads the just-written storage slot, so retain the
                # frame-start address on the signal even though Pitch.time is
                # now correctly center-stamped for plotting and analysis.
                self.pitch_detected.emit(t)

            except Exception as e:
                print(f"[PitchDetector] frame skipped due to error: {e}")
                continue

    def stop(self, drain: bool = False):
        """Stop live detection, optionally processing all complete queued frames."""
        if self.pda_thread and self.pda_thread.is_alive():
            self._drain_on_stop = drain
            self.stop_event.set()
            self.pda_thread.join() # pause the main thread until recording thread recognizes the stop event
        self._drain_on_stop = False

    # OFFLINE (whole-file) detection, run on a background thread so the Qt event
    # loop stays free (e.g. to animate a loading spinner while we wait).
    def detect_pitches_async(self):
        """Run the recording's full offline pitch pipeline (detect + smooth) on a
        background daemon thread. Emits `status_changed(text)` as each phase
        begins and `detection_finished` when the recording's pitch_data is ready.
        Both fire from the worker thread, so Qt queues the connected slots onto
        the main thread automatically."""
        if self.offline_thread and self.offline_thread.is_alive():
            return  # a detection is already in flight
        self.offline_thread = threading.Thread(target=self._detect_pitches_offline, daemon=True)
        self.offline_thread.start()

    def _detect_pitches_offline(self):
        """Worker body: detect + smooth the whole recording, then signal done.
        Phase changes are surfaced via `status_changed` (see detect_pitches)."""
        try:
            self.recording.detect_pitches(on_phase=self.status_changed.emit)
        except Exception as e:
            print(f"[PitchDetector] offline detection failed: {e}")
        finally:
            self.detection_finished.emit()


    # THE DETECTION ALGORITHM
    def detect_pitch(self, x: np.ndarray, start_time: float=None) -> Pitch:
        """a method to call pitch detection on a single frame
        requires an explicit reference to the start time

        Args:
            x: the array of audio to perform pitch detection on
            start_time: median time of frame (in sec)
        """
        # Timbre must see the RAW frame (including silence), before centering,
        # filtering, peak normalization, or the pitch detector's volume gate.
        self._write_timbre_frame(x, start_time)

        unvoiced_pitch = Pitch(
            time=start_time, candidates=[],
            volume=0.0, unvoiced_prob=1.0,
            live_distance=None, config=self.config
        )
        # if frame is empty, return unvoiced pitch
        if np.all(x == 0):
            return unvoiced_pitch

        # preprocess audio to center and get rid of low frequency noise
        x, volume = self.preprocess_audio(x)

        # VOLUME GATE: if frame quieter than min volume, return unvoiced
        self._stream_volume_peak = max(self._stream_volume_peak, float(volume))
        min_volume = self._stream_volume_peak * max(0.0, float(self.config.min_volume))
        if volume < min_volume or not np.any(x):
            unvoiced_pitch.volume = volume
            return unvoiced_pitch

        # compute autocorrelation and modify it to avoid 0-lag peak
        acf, _ = self.autocorrelation_fft(x)
        cdf = self.cmndf(x, acf)
        # cdf = self.clamped_diff_fct(x=x, acf=acf)

        # prominence picking + probability assignment to all freq estimates
        acf_peaks = self.find_acf_peaks(acf)
        pitch_probs, unvoiced_prob = self.pitch_probabilities(acf_peaks, cdf)

        # interpolate + compute final freq estimates
        freq_estimates = [self.SR/self.parabolic_interpolation(acf, t) for t in acf_peaks]
        midi_estimates = [self.config.freq_to_midi(f) for f in freq_estimates]

        # create + return the final pitch object
        candidates = list(zip(midi_estimates, pitch_probs))
        candidates.sort(key=lambda c: c[1], reverse=True) # sort from most to least probable
        score_note = self.recording.score_data.current_note() if self.recording and self.recording.score_data else None
        distance = score_note.midi_num[0] - candidates[0][0] if score_note and candidates else None
        
        pitch = Pitch(time=start_time, candidates=candidates,
                      volume=volume, unvoiced_prob=unvoiced_prob,
                      live_distance=distance, config=self.config)
        return pitch


    def detect_pitches(
        self,
        x: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "Detecting pitches",
        verbose: bool = False,
    ) -> list[Pitch]:
        """
        Computes multi-frame pitch detection on an arbitrary length array of audio data.
        Returns a nested list of pitches, each corresponding to the freq estimates (probabilistic)
        for each timestep
        """
        self._cqt_frame_index = 0
        self._cqt_offline = True
        if len(x) < self.FRAME_SIZE:
            self._cqt_offline = False
            return []

        # get memory efficient frames with np pointer c++ magic
        frames = np.lib.stride_tricks.sliding_window_view(x, self.FRAME_SIZE)[::self.HOP_SIZE]
        n_frames = 1 + (len(x) - self.FRAME_SIZE) // self.HOP_SIZE 
        volumes = self._frame_volumes(x, n_frames)
        # min_volume is the ratio (fraction of the reference below which a frame is
        # gated to unvoiced); max_volume is that reference's percentile, stored as a
        # fraction (0.95 -> the 95th percentile of frame RMS).
        gate_ratio = max(0.0, float(self.config.min_volume))
        gate_percentile = np.clip(
            float(self.config.max_volume) * 100.0,
            0.0,
            100.0,
        )
        volume_reference = (
            float(np.percentile(volumes, gate_percentile))
            if volumes.size and gate_ratio > 0
            else 0.0
        )
        min_volume = volume_reference * gate_ratio
        silent_frames = int(np.sum(volumes <= 0.0))
        gated_frames = int(np.sum((volumes > 0.0) & (volumes < min_volume))) if min_volume > 0 else 0
        self._last_volume_gate_stats = {
            "pitch_total_frames": int(n_frames),
            "pitch_silent_frame_count": silent_frames,
            "pitch_volume_gate_frame_count": gated_frames,
            "pitch_volume_gate_min_volume": float(min_volume),
            "pitch_volume_gate_reference": float(volume_reference),
            "pitch_volume_gate_ratio": float(gate_ratio),
            "pitch_volume_gate_percentile": float(gate_percentile),
        }

        pitches = []
        frames_iter = enumerate(frames)
        start = time.perf_counter()
        if verbose:
            print(f"[PitchDetector] detecting {n_frames} frame(s)", flush=True)
            if min_volume > 0:
                print(
                    f"[PitchDetector] volume gate: RMS < {min_volume:.6g} "
                    f"({gate_ratio:.3f} * p{gate_percentile:g} frame RMS "
                    f"{volume_reference:.6g}) -> unvoiced; "
                    f"gated {gated_frames}/{n_frames} non-silent frame(s), "
                    f"{silent_frames} already silent",
                    flush=True,
                )
        if show_progress:
            frames_iter = tqdm(
                frames_iter,
                total=n_frames,
                desc=progress_desc,
                leave=False,
                mininterval=0.25,
            )

        try:
            for i, frame in frames_iter:
                start_time = (i * self.HOP_SIZE + 0.5 * self.FRAME_SIZE) / self.SR
                pitch = self.detect_pitch(frame, start_time)
                # offline detection has no live playhead, so detect_pitch's per-frame
                # distance (to the score's *current* note at a fixed cursor) is
                # meaningless here. Leave it None so detected-but-unanalyzed pitches
                # render neutral grey until analyze() assigns alignment distances.
                pitch.live_distance = None
                pitches.append(pitch)
        finally:
            self._cqt_offline = False

        if verbose:
            print(
                f"[PitchDetector] done: {len(pitches)} pitch frame(s) in "
                f"{time.perf_counter() - start:.2f}s",
                flush=True,
            )
        return pitches

    def _write_timbre_frame(self, raw_frame: np.ndarray, start_time: float | None):
        """Write every configured stride-th raw frame to TimbreData."""
        local_frame_i = self._cqt_frame_index
        self._cqt_frame_index += 1
        stride = int(getattr(self.config, "cqt_stride", 0) or 0)
        if self.recording is None or stride <= 0:
            return
        # Offline frames are a dense array beginning at frame zero. Live
        # streams may begin at a nonzero clip time (and Practice may stall on
        # one time), so address their global PitchData grid from the emitted
        # app-time instead of pretending every run begins at column zero.
        if self._cqt_offline or start_time is None:
            frame_i = local_frame_i
        else:
            frame_i = max(0, self.recording.pitch_data.time_to_index(start_time))
        if frame_i % stride:
            return
        td = getattr(self.recording, "timbre_data", None)
        if td is None:
            return
        if td.computed_until == 0:
            td.t_origin = self.recording.audio_data.t_origin
        td.write(frame_i // stride, self.cqt.power_db(raw_frame))

    def _frame_volumes(self, audio: np.ndarray, n_frames: int) -> np.ndarray:
        if n_frames <= 0:
            return np.empty(0, dtype=float)

        audio = np.asarray(audio, dtype=np.float64)
        starts = np.arange(n_frames, dtype=np.int64) * self.HOP_SIZE
        ends = starts + self.FRAME_SIZE

        prefix = np.concatenate(([0.0], np.cumsum(audio, dtype=np.float64)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(audio * audio, dtype=np.float64)))

        sums = prefix[ends] - prefix[starts]
        sums_sq = prefix_sq[ends] - prefix_sq[starts]
        mean = sums / self.FRAME_SIZE
        variance = (sums_sq / self.FRAME_SIZE) - (mean * mean)
        return np.sqrt(np.maximum(variance, 0.0))

    # METHODS TO IMPLEMENT THE ALGORITHM
    # ---
    def re_init(self, config: Config=None):
        """re-initialize the tuning parameters"""
        if config:
            self.config = config
            self.SR = config.sr
            self.tau_max = int(self.SR / config.fmin) 
            self.tau_min = int(self.SR / config.fmax)

    # the probability distribution of thresholds
    def threshold_prior(self, n_thresholds: int=100, a: float=2, b: float=34/3) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a beta distribution modeling the pdf for YIN thresholds,
        represented as a numpy array of size N_THRESHOLDS corresponding to the pdf values.
        Also returns the array of thresholds.

        Possible a,b parameters from paper:
            - mean=0.1 beta(a=2, b=18)
            - mean=0.15 beta(a=2, b=11.33)
            - mean=0.2 beta(a=2, b=8)

        Not yet sure how these parameters are determined but I got them from the paper.
        """
        # all thresholds
        thresholds = np.linspace(0, 1, n_thresholds+1)
        thresholds = thresholds[1:] # remove the 0 threshold
        beta_pdf = beta.pdf(thresholds, a, b) / n_thresholds
        return beta_pdf, thresholds

    # --- frequency parsing functions ---
    # autocorrelation (base)
    def autocorrelation_fft(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast autocorrelation function implementation using Wiener-Khinchin theorem,
        which computes autocorrelation as the inverse FFT of the signal's power spectrum.

        Step 1 of Yin algorithm, corresponding to equation (1) in Cheveigne, Kawahara 2002.

        Args:
            x: The current frame of audio samples in Yin algorithm
            tau_max: Check for all time lags up to this value for in autocorrelation

        Returns:
            autocorrelation: The similarity curve.
            amplitudes: Amplitudes of the frame.
        """
        x = np.array(x, dtype=np.float64)
        w = x.size
        tau_max = min(self.tau_max, w)

        # zero-pad the audio signal array by the minimum power of 2 which
        # is larger than the window size + tau_max
        min_fft_size = w + tau_max  # (pad by >tau_max for frame end)

        p2 = (min_fft_size // 32).bit_length()
        nice_fft_sizes = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(size * (2 ** p2) for size in nice_fft_sizes if size * 2 ** p2 >= min_fft_size)

        # --- AUTOCORRELATION WITH WIENER-KHINCHIN ---
        # decompose the signal into its frequency components
        fft_x = np.fft.rfft(x, size_pad)  
        psd = fft_x * fft_x.conjugate() # power spectrum density
        autocorrelation = np.fft.irfft(psd)[:tau_max] 

        amplitudes = np.abs(fft_x)

        # only return valid overlapping values up to window_size-tau_max
        return autocorrelation, amplitudes

    def cmndf(self, x, acf) -> np.ndarray:
        """
        Cumulative mean normalized difference function from the original YIN
        algorithm.

        Args:
            x: The current analysis frame.
            acf: Full-frame autocorrelation of ``x``.
        """
        tau_limit = min(self.tau_max, len(acf), len(x) - 1)
        cmndf = np.ones(self.tau_max, dtype=np.float64)
        if tau_limit <= 1:
            return cmndf

        # YIN equation (2) compares x[j] with x[j + tau] over one fixed
        # window. Reserve tau_limit samples as look-ahead so every lag uses the
        # same number of terms instead of padding the shifted window with zero.
        window_size = len(x) - tau_limit
        reference_energy = float(np.dot(x[:window_size], x[:window_size]))

        # acf[tau] contains the desired fixed-window cross term plus the
        # autocorrelation of the reserved tail. Subtracting that small tail ACF
        # reuses the full-frame FFT rather than performing a second large FFT.
        tail = x[window_size:]
        tail_acf = np.correlate(tail, tail, mode="full")[
            len(tail) - 1 : len(tail) - 1 + tau_limit
        ]
        cross_terms = acf[:tau_limit] - tail_acf

        squared_prefix = np.concatenate(
            ([0.0], np.cumsum(x * x, dtype=np.float64))
        )
        starts = np.arange(tau_limit)
        shifted_energy = (
            squared_prefix[starts + window_size] - squared_prefix[starts]
        )
        diff_fct = (
            reference_energy + shifted_energy - 2.0 * cross_terms
        )
        # Roundoff can make the theoretically non-negative difference a few
        # ulps below zero.
        diff_fct = np.maximum(diff_fct, 0.0)
        diff_fct[0] = 0.0

        # YIN equation (3): d'(0) = 1 and, for tau > 0, divide by the
        # cumulative mean from lag 1 through tau. There is deliberately no
        # additive pseudocount in the cumulative sum.
        taus = np.arange(1, tau_limit)
        cumulative_diff = np.cumsum(diff_fct[1:tau_limit])
        cmndf[1:tau_limit] = np.divide(
            diff_fct[1:tau_limit] * taus,
            cumulative_diff,
            out=np.ones(tau_limit - 1, dtype=np.float64),
            where=cumulative_diff > 0.0,
        )
        return cmndf

    # modifying the difference function
    def clamped_diff_fct(self, x, acf) -> np.ndarray:
        """
        modifies the base autocorrelation by inverting + normalizing it, then
        clamping all values outside of desired tau_range to be 1

        Args:
            x: needed to compute energy for diff_fct inversion
            acf: the result of autocorrelation on x
        """
        # --- INVERT TO DIFFERENCE FUNCTION ---
        # compute the energy (r_t(0) and r_{t+\tau}(0)) for each lag
        r_0 = np.sum(x**2)
        energy = np.full(acf.shape, r_0)

        diff_fct = energy[0] + energy - 2*acf
        diff_fct[0] = 0
        diff_fct = np.abs(diff_fct)

        # --- NORMALIZE + CLAMP
        diff_fct = diff_fct / (np.max(diff_fct) - np.min(diff_fct))

        clamp_df = np.zeros(self.tau_max) 
        clamp_df[:self.tau_min] = 1 # make everything before min f_0 1
        total_diff = self.tau_min

        for tau in range(self.tau_min, self.tau_max):
            total_diff += diff_fct[tau]
            avg_diff = total_diff / tau 
            clamp_df[tau] = diff_fct[tau] / avg_diff

        return clamp_df
    
    # --- peak-picking ---
    # prominence-based initial peak-finding
    def find_acf_peaks(self, acf: np.ndarray):
        """prominence-based peak picking of the autocorrelation curve
        returns the indices of all possible tau (fundamental period) values
        """
        # initial prominence as 1/2 overall acf range
        prominence = abs((np.max(acf) - np.min(acf))/2)

        n = 5 # how many times to find peaks within the intial prominence range
        for i in range(0, n):
            # try the lowest prominence we can that still returns valid
            p = prominence - prominence*(i/n) 
            acf_peaks, _ = find_peaks(acf, prominence=p)
            if len(acf_peaks) > 0:
                break

        # fallback if still empty
        if acf_peaks.size == 0:
            # look for the global ACF max in [tau_min, tau_max)
            region = acf[self.tau_min : self.tau_max]
            best = np.argmax(region) + self.tau_min
            acf_peaks = np.array([best], dtype=int)

        return acf_peaks

    # find the pitch according to YIN thresholding
    def find_pitch(self, cdf: np.ndarray, acf_peaks: np.ndarray, threshold: float=0.1) -> tuple[int, bool]:
        """
        Finds the YIN pitch estimate with their absolute thresholding step by searching for the first cdf trough
        below a certain 'absolute threshold'. Runs 

        Args:
            cdf: the clamped difference function for y-values corresponding in the threshold range
            acf_peaks: indices of the prominent-peaks found from the ACF (to index into the CDF)
            threshold: take the first trough below this value of d'(tau)
        
        Returns:
            tau_0 (int): The fundamental period estimate. If possible, the first tau st. 
                         d'(tau) < threshold. Else, the x of the global minima
            tau_idx (int): index of the chosen peak in the acf_peaks array
            is_voiced (bool): False if we return the global min
        """
        for i, min in np.ndenumerate(acf_peaks):
            if cdf[min] <= threshold:
                return min, i[0], True
        
        # no min found below threshold, return the global minima
        i = np.argmin(cdf[acf_peaks])
        global_min = acf_peaks[i]
        return global_min, i, False
    
    # assign remaining peaks a probability with find_pitch
    def pitch_probabilities(self, acf_peaks: np.ndarray, cdf: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Given all prominent-enough peaks of the original ACF curve, 
        computes the probability of all possible period (tau) = 1/f_0 estimates.
        Based off the PYin method of probability assignment.

        Args:
            acf_peaks: indices of all prominent-enough peaks of the original ACF curve
            cdf: clamped difference function, to help for threshold-based peak-picking

        Returns:
            pitch_probs: array of same shape as acf_peaks, where corresponding
                         indices represent associated probabilities
            unvoiced_prob: 1 - sum(tau_probs), eg adding up all the times we had to 
                           take the global min because nothing was below the threshold
        """
        pitch_probs = np.zeros_like(acf_peaks, dtype=np.float64)

        for i, threshold in enumerate(self.thresholds):
            tau_0, j, is_voiced = self.find_pitch(cdf, acf_peaks, threshold)
            # if returned pitch for a threshold is not within pitch range, call it unvoiced
            # (trying to minimize harmonic errors)
            if is_voiced and tau_0 <= self.tau_max and tau_0 >= self.tau_min:
                pitch_probs[j] += self.beta_pdf[i]
            else:
                pitch_probs[j] += self.beta_pdf[i] * self.UNVOICED_PROB
            
        unvoiced_prob = 1 - np.sum(pitch_probs)
        return pitch_probs, unvoiced_prob

    # refine final peak estimates
    def parabolic_interpolation(self, acf: np.ndarray, acf_peak: int) -> float:
        """
        Refines the peak estimates by performing parabolic interpolation around the given index
        of the AACF. Fits a negative quadratic to the supplied minima.
        
        Args:
            acf: A 1D array of y-values (e.g., diff_fct values).
            acf_peak: The index of the maximum point in acf to interpolate around.

        Returns:
            The interpolated x-pos of the supplied acf_peak
        """
        x = acf_peak

        # don't interpolate at boundaries - need at least 3 points
        if x <= 0 or x >= len(acf) - 1:
            return float(x)

        y_1 = acf[x - 1]
        y_2 = acf[x]
        y_3 = acf[x + 1]

        denominator = 2 * (y_1 - 2*y_2 + y_3)
        if denominator == 0:
            return float(x)
        
        x_interpolated = x + (y_1 - y_3) / denominator
        return x_interpolated


    # AUDIO PREPROCESSING
    def pitch_bandpass_filter(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Retain the configured pitch range with asymmetric padding.

        Config.fmin/fmax constrain the YIN lag search and smoother state space;
        the filter extends two semitones below fmin and two octaves above fmax.
        The wider upper padding preserves four harmonics at the top of the
        expected range instead of repeating the old 1.2*fmax cutoff, which
        removed evidence needed to resolve high violin notes.
        """
        padded_fmin = self.config.midi_to_freq(
            self.config.freq_to_midi(self.config.fmin) - self.PADDING
        )
        padded_fmax = self.config.midi_to_freq(
            self.config.freq_to_midi(self.config.fmax) + 24.0
        )
        upper_cutoff = min(
            padded_fmax,
            0.5 * self.SR * (1.0 - 1e-6),
        )
        key = (padded_fmin, upper_cutoff, self.SR)
        if self._preprocess_filter_key != key:
            self._preprocess_sos = iirfilter(
                N=2,
                Wn=[padded_fmin, upper_cutoff],
                btype="bandpass",
                ftype="butter",
                output="sos",
                fs=self.SR,
            )
            self._preprocess_filter_key = key
        return sosfilt(self._preprocess_sos, x)

    def preprocess_audio(self, x: list) -> tuple[np.ndarray, float]:
        """
        centers the audio around mean, normalizes, 
        and applies high pass iir filter to prepare for pitch detection

        Args:
            x (list): The input audio signal as a list of samples.
            iir_cutoff_freq (float, optional): The cutoff frequency for the high-pass filter. Defaults to 150 Hz.

        Returns:
            tuple: A tuple containing the preprocessed audio signal (as a NumPy array) and the volume (as a float).
        """
        if len(x) == 0:
            return np.array([]), 0.0
        x = np.asarray(x, dtype=float)
        # x = x.astype(float)
        x = x - np.mean(x) # center
        volume = np.sqrt(np.mean(x ** 2))  # get volume as mean |amplitude| of the x (before normalizing)
        peak = np.max(np.abs(x))
        if peak == 0:  # constant/silent frame (digital silence, DC) -> no pitch
            return np.zeros_like(x), 0.0
        x = x/peak # normalize
        x = self.pitch_bandpass_filter(x)
        return x, volume
