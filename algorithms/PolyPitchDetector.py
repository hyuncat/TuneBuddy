"""Streaming polyphonic (multi-f0) pitch detection: supervised NMF against a
fixed harmonic dictionary.

Per frame (so it streams exactly like the mono detector): the magnitude
spectrum v is decomposed as v ~= W @ h, where W is a FIXED dictionary holding
one harmonic-comb template per dictionary pitch and h is the nonnegative
per-pitch activation ("salience") vector, solved by a few KL multiplicative
updates warm-started from the previous frame. Activation peaks above the
salience thresholds — that also pass a fundamental-presence check — are the
frame's simultaneous pitch candidates. Offline, the same updates run
chunk-wise over the whole spectrogram as matrix-matrix products.

This is the real-time architecture of Dessein, Cont & Lemaitre (ISMIR 2010),
"Real-time polyphonic music transcription with NMF and beta-divergence":
    https://archives.ismir.net/ismir2010/paper/000083.pdf
using the KL (beta=1) multiplicative updates of Lee & Seung (NIPS 2000):
    https://proceedings.neurips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf
NMF-for-transcription originates with Smaragdis & Brown (WASPAA 2003):
    https://www.merl.com/publications/docs/TR2003-139.pdf
The fixed harmonic-comb templates (instrument-agnostic: they encode
harmonicity, not timbre) follow the harmonic-constraint idea of Vincent,
Bertin & Badeau (IEEE TASLP 2010), "Adaptive harmonic spectral decomposition
for multiple pitch estimation":
    https://inria.hal.science/inria-00544094v1/document
Template-construction recipes / reference implementations:
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFSpecFac.html
    https://github.com/groupmm/libnmfd

Known failure modes of NMF multi-f0, and what fights them here:
  - sub-harmonic ghosts (a phantom A3 "explains" an A4+E5 fifth because its
    harmonic series covers both) -> the poly_f0_presence SUPPORT CONSTRAINT:
    pitches with no spectral energy in their own fundamental's region get
    their activation zeroed BEFORE the decode (multiplicative updates keep
    zeros at zero), so ghosts can't soak up mass that belongs to real
    pitches. Filtering only at the peak-picking stage is not enough: a
    rejected ghost still starves the real pitches below the salience
    thresholds.
  - octave ghosts (2*f0 shares every even partial with f0) -> harmonic decay
    + sparsity soften these, but we deliberately do NOT hard-suppress
    octaves: violin double-stop octaves are real notes, and a played f0 vs
    f0+octave is genuinely ambiguous from one frame's magnitudes.
"""
import numpy as np
from scipy import sparse
from scipy.signal import iirfilter, sosfilt
from tqdm import tqdm

from algorithms.Config import Config
from algorithms.PitchDetector import PitchDetector
from app_logic.user.ds.PolyPitchData import PolyPitch


class PolyPitchDetector(PitchDetector):
    """Drop-in A/B replacement for PitchDetector (same offline/online entry
    points, threading, and volume gate); only the per-frame algorithm differs,
    and candidates mean SIMULTANEOUS pitches (salience shares, not competing
    hypotheses)."""

    # per-frame activations are already final (no candidate distribution to
    # decode), so Recording.detect_pitches skips the monophonic HMM smoother
    requires_smoothing = False

    #: partial spread of each dictionary Gaussian, in semitones (tuning/vibrato slack)
    PARTIAL_SIGMA_SEMITONES = 0.25
    #: warm-start floor so activations zeroed by multiplicative updates can revive
    H_FLOOR = 1e-6
    #: relative-change early stop for the multiplicative updates
    MU_TOL = 1e-2
    #: iteration cap for warm-started (streaming) frames: tracking a slowly
    #: changing h doesn't need full convergence, just a bounded refresh
    WARM_ITERS = 8
    #: n_fft = ZERO_PAD * w1 (~2.7 Hz bins at sr=44100, w1=4096, so semitone
    #: templates stay separable down at the violin G string)
    ZERO_PAD = 4
    #: offline batch width (frames decoded per matrix-matrix NMF chunk)
    CHUNK_FRAMES = 4096

    def __init__(self, recording=None, config: Config = None, parent=None):
        super().__init__(recording=recording, config=config, parent=parent)
        self._h_prev = None
        self._sos_key = None
        self._build_dictionary()

    def load_config(self, config: Config):
        super().load_config(config)
        self._h_prev = None
        self._sos_key = None
        self._build_dictionary()

    def bandpass_filter(self, x: np.ndarray, fmin: float = 50, fmax: float = 4000) -> np.ndarray:
        """Same filter as the parent, but the IIR design is cached — the parent
        redesigns it every call, which is ms-scale and alone blows the
        per-frame budget at streaming hop rate."""
        key = (float(fmin), float(fmax), int(self.SR))
        if self._sos_key != key:
            self._sos_key = key
            self._sos = iirfilter(
                N=2, Wn=[fmin, fmax],
                btype='bandpass',
                ftype='butter',
                output='sos',
                fs=self.SR,
            )
        return sosfilt(self._sos, x)

    def _build_dictionary(self):
        """Precompute the harmonic template matrix W: one L1-normalized column
        per dictionary pitch, each a comb of Gaussian partials with geometric
        amplitude decay, on a semitone grid over [fmin, fmax] by default
        (poly_bins_per_semitone refines it). The spectrum rows are cropped to
        the band the templates live in, which shrinks every per-frame matvec.
        Also precomputes each pitch's partial windows for the f0-presence
        check."""
        cfg = self.config
        step = 1.0 / max(1, int(cfg.poly_bins_per_semitone))
        # pad the grid half a semitone past [fmin, fmax] so boundary notes keep
        # a template despite float rounding (fmin=196.0 vs G3=195.998 Hz!) —
        # the mono detector pads its lag range the same way (padded_fmin)
        lo = float(np.ceil(cfg.freq_to_midi(cfg.fmin) - 0.5))
        hi = float(np.floor(cfg.freq_to_midi(cfg.fmax) + 0.5))
        self.grid_step = step
        self.midi_grid = np.arange(lo, hi + step * 0.5, step)

        self.n_fft = self.ZERO_PAD * cfg.w1
        freqs_all = np.fft.rfftfreq(self.n_fft, 1.0 / cfg.sr)
        df = float(freqs_all[1])
        # preprocess_audio bandpasses to [0.8*fmin, 1.2*fmax]; partials past
        # that edge would only dilute the template's normalized mass
        f_cap = min(0.45 * cfg.sr, cfg.fmax * 1.2)
        spread = 2 ** (self.PARTIAL_SIGMA_SEMITONES / 12) - 1

        self._row_lo = max(0, int(np.floor(0.7 * cfg.fmin / df)))
        self._row_hi = min(
            freqs_all.size,
            int(np.ceil((f_cap + 6 * max(df, f_cap * spread)) / df)) + 1,
        )
        freqs = freqs_all[self._row_lo:self._row_hi]

        # partial windows for the f0-presence support constraint, flat and
        # grouped by pitch (win_starts[j] indexes pitch j's k=1 window) so the
        # per-frame mask computes via cumsum + reduceat, no python loop
        W = np.zeros((freqs.size, self.midi_grid.size), dtype=np.float64)
        win_lo, win_hi, win_starts = [], [], []
        for col, midi in enumerate(self.midi_grid):
            f0 = cfg.midi_to_freq(midi)
            win_starts.append(len(win_lo))
            for k in range(1, max(1, int(cfg.poly_n_harmonics)) + 1):
                fk = k * f0
                if fk > f_cap:
                    break
                sigma = max(df, fk * spread)
                W[:, col] += (cfg.poly_harmonic_decay ** (k - 1)) * np.exp(
                    -0.5 * ((freqs - fk) / sigma) ** 2
                )
                w_lo = int(np.searchsorted(freqs, fk - 2 * sigma))
                w_hi = max(w_lo + 1, int(np.searchsorted(freqs, fk + 2 * sigma)))
                win_lo.append(w_lo)
                win_hi.append(min(w_hi, freqs.size))
        self._win_lo = np.asarray(win_lo, dtype=np.intp)
        self._win_hi = np.asarray(win_hi, dtype=np.intp)
        self._win_starts = np.asarray(win_starts, dtype=np.intp)
        # L1-normalized columns make the KL update denominator (W^T @ 1)
        # exactly 1, and activations read as shares of the frame spectral mass
        W /= np.maximum(W.sum(axis=0, keepdims=True), 1e-12)
        # the Gaussian combs are ~90% zeros; sparse matvecs sidestep the BLAS
        # dispatch overhead that dominates dense gemv at this tiny size
        W[W < W.max(axis=0, keepdims=True) * 1e-4] = 0.0
        W /= np.maximum(W.sum(axis=0, keepdims=True), 1e-12)
        self.W = sparse.csr_matrix(W.astype(np.float32))
        self.WT = sparse.csr_matrix(W.T.astype(np.float32))
        self.window = np.hanning(cfg.w1).astype(np.float64)

    # THE DETECTION ALGORITHM (single frame: the streaming path)
    def detect_pitch(self, x: np.ndarray, start_time: float = None) -> PolyPitch:
        """NMF multi-pitch for a single frame. Same contract as the mono
        detect_pitch (called by the inherited online thread); candidates are
        simultaneous pitches, salience-ordered."""
        unvoiced_pitch = PolyPitch(
            time=start_time, candidates=[],
            volume=0.0, unvoiced_prob=1.0,
            live_distance=None, config=self.config,
        )
        if np.all(np.asarray(x) == 0):
            self._h_prev = None  # silence breaks the warm-start continuity
            return unvoiced_pitch

        x, volume = self.preprocess_audio(x)

        # VOLUME GATE: same streaming gate as the mono detector
        self._stream_volume_peak = max(self._stream_volume_peak, float(volume))
        min_volume = self._stream_volume_peak * max(0.0, float(self.config.min_volume))
        if volume < min_volume or not np.any(x):
            unvoiced_pitch.volume = volume
            self._h_prev = None
            return unvoiced_pitch

        n = self.window.size
        if x.size < n:
            x = np.pad(x, (0, n - x.size))
        spec = np.abs(np.fft.rfft(x[:n] * self.window, self.n_fft))
        v = spec[self._row_lo:self._row_hi].astype(np.float32)
        total = float(v.sum())
        if total <= 0:
            unvoiced_pitch.volume = volume
            return unvoiced_pitch
        v /= total  # unit spectral mass: saliences + sparsity are loudness-free

        h = self._activations(v)
        candidates = self._threshold_peaks(h)
        if not candidates:
            unvoiced_pitch.volume = volume
            return unvoiced_pitch

        score_note = self.recording.score_data.current_note() if self.recording and self.recording.score_data else None
        distance = score_note.midi_num[0] - candidates[0][0] if score_note else None

        return PolyPitch(
            time=start_time, candidates=candidates,
            volume=volume, unvoiced_prob=self._unvoiced_prob(v, h),
            live_distance=distance, config=self.config,
        )

    def _activations(self, v: np.ndarray) -> np.ndarray:
        """Nonnegative decode of v against the fixed dictionary: KL
        multiplicative updates (Lee & Seung 2000) with an L1 sparsity term,
        warm-started from the previous frame (Dessein et al. 2010) so the
        streaming case converges in a handful of iterations."""
        W, WT = self.W, self.WT
        eps = np.float32(1e-10)
        mask = self._presence_mask(v)  # support constraint: ghosts start (and stay) at 0
        max_iters = max(1, int(self.config.poly_nmf_iters))
        if self._h_prev is not None and self._h_prev.size == W.shape[1]:
            h = np.maximum(self._h_prev, self.H_FLOOR) * mask
            max_iters = min(max_iters, self.WARM_ITERS)
        else:
            h = (WT @ v + self.H_FLOOR) * mask  # projection init on cold start
        # W columns are L1-normalized, so the KL denominator W^T @ 1 == 1
        denom = np.float32(1.0 + max(0.0, float(self.config.poly_sparsity)))
        for _ in range(max_iters):
            recon = W @ h
            h_new = h * (WT @ (v / (recon + eps))) / denom
            delta = float(np.max(np.abs(h_new - h)))
            h = h_new
            if delta <= self.MU_TOL * max(float(h.max()), 1e-12):
                break
        self._h_prev = h.copy()
        return h

    def _threshold_peaks(self, h: np.ndarray) -> list[tuple[float, float]]:
        """Turn the activation vector into (midi, salience share) candidates,
        strongest first: salience thresholds (as shares of the frame's unit
        spectral mass), de-duplication of dictionary neighbors, and sub-grid
        parabolic refinement. (Ghost filtering already happened upstream via
        the presence mask in the decode.)"""
        cfg = self.config
        order = np.argsort(h)[::-1]
        h_max = float(h[order[0]])
        floor_ = float(cfg.poly_min_salience)
        rel_floor = float(cfg.poly_salience_thresh) * h_max
        neighbor = max(1, int(cfg.poly_bins_per_semitone))  # ±1 semitone of a kept pitch

        keep: list[int] = []
        for idx in order:
            if len(keep) >= max(1, int(cfg.poly_max_voices)):
                break
            sal = float(h[idx])
            if sal < floor_ or sal < rel_floor:
                break  # order is descending; nothing further can pass
            if any(abs(int(idx) - k) <= neighbor for k in keep):
                continue  # spill-over into a kept pitch's neighboring template
            keep.append(int(idx))
        if not keep:
            return []

        # sub-grid refinement: parabola through the activation peak (inherited
        # helper); roughly ±20 cents on the default semitone grid
        candidates = []
        for idx in keep:
            refined = self.parabolic_interpolation(h, idx)
            refined = min(max(refined, idx - 1.0), idx + 1.0)
            midi = float(self.midi_grid[0] + refined * self.grid_step)
            candidates.append((midi, float(h[idx])))
        total = sum(s for _, s in candidates)
        return [(m, s / total) for m, s in candidates]  # salience SHARES

    def _presence_mask(self, v: np.ndarray) -> np.ndarray:
        """f0-presence support constraint (float 0/1 per dictionary pitch): a
        real pitch has spectral energy in its OWN f0 region; a ghost an octave
        (or a twelfth) below covers the played partials but predicts a
        fundamental the spectrum doesn't contain."""
        cs = np.concatenate(([0.0], np.cumsum(v, dtype=np.float64)))
        energies = cs[self._win_hi] - cs[self._win_lo]  # every partial window
        strongest = np.maximum.reduceat(energies, self._win_starts)
        e_f0 = energies[self._win_starts]  # each pitch's k=1 window
        mask = e_f0 >= float(self.config.poly_f0_presence) * strongest
        return mask.astype(np.float32)

    def _presence_mask_batch(self, V: np.ndarray) -> np.ndarray:
        """Column-wise _presence_mask for a spectrogram chunk (rows, m)."""
        cs = np.vstack((
            np.zeros((1, V.shape[1]), dtype=np.float32),
            np.cumsum(V, axis=0, dtype=np.float32),
        ))
        energies = cs[self._win_hi] - cs[self._win_lo]
        strongest = np.maximum.reduceat(energies, self._win_starts, axis=0)
        e_f0 = energies[self._win_starts]
        mask = e_f0 >= np.float32(self.config.poly_f0_presence) * strongest
        return mask.astype(np.float32)

    def _unvoiced_prob(self, v: np.ndarray, h: np.ndarray) -> float:
        """Soft voicing: 1 - the cosine fit of the harmonic reconstruction, so
        inharmonic frames the volume gate lets through still read as noisy."""
        recon = self.W @ h
        norm = float(np.linalg.norm(v) * np.linalg.norm(recon))
        fit = float(v @ recon) / norm if norm > 0 else 0.0
        return float(np.clip(1.0 - fit, 0.0, 1.0))

    # OFFLINE (whole-file) detection
    def detect_pitches(
        self,
        x: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "Detecting pitches",
        verbose: bool = False,
    ) -> list[PolyPitch]:
        """Batched offline pass: same frame grid, volume gate, and outputs as
        the inherited per-frame loop, but the bandpass runs once over the whole
        take and the NMF decodes CHUNK_FRAMES-wide spectrogram chunks with
        matrix-matrix updates (orders of magnitude faster than frame-wise)."""
        if len(x) < self.FRAME_SIZE:
            return []
        raw = np.asarray(x, dtype=np.float64)
        n_frames = 1 + (len(raw) - self.FRAME_SIZE) // self.HOP_SIZE
        volumes = self._frame_volumes(raw, n_frames)

        # the streaming running-peak gate, vectorized (volume < peak_so_far * ratio)
        gate_ratio = max(0.0, float(self.config.min_volume))
        peaks = np.maximum.accumulate(volumes) if volumes.size else volumes
        voiced = (volumes > 0.0) & (volumes >= peaks * gate_ratio)
        self._last_volume_gate_stats = {
            "pitch_total_frames": int(n_frames),
            "pitch_silent_frame_count": int(np.sum(volumes <= 0.0)),
            "pitch_volume_gate_frame_count": int(np.sum((volumes > 0.0) & ~voiced)),
            "pitch_volume_gate_min_volume": float(peaks[-1] * gate_ratio) if volumes.size else 0.0,
            "pitch_volume_gate_reference": float(peaks[-1]) if volumes.size else 0.0,
            "pitch_volume_gate_ratio": float(gate_ratio),
            "pitch_volume_gate_percentile": 100.0,  # running peak == the max so far
        }

        frame_dt = self.HOP_SIZE / self.SR
        t_off = 0.5 * self.FRAME_SIZE / self.SR
        pitches = [
            PolyPitch(
                time=i * frame_dt + t_off, candidates=[],
                volume=float(volumes[i]), unvoiced_prob=1.0,
                live_distance=None, config=self.config,
            )
            for i in range(n_frames)
        ]

        filtered = self.bandpass_filter(
            raw - np.mean(raw),
            fmin=self.config.fmin * 0.8,
            fmax=self.config.fmax * 1.2,
        )
        frames = np.lib.stride_tricks.sliding_window_view(filtered, self.FRAME_SIZE)[::self.HOP_SIZE]

        voiced_idx = np.flatnonzero(voiced)
        progress = tqdm(
            total=int(voiced_idx.size), desc=progress_desc,
            leave=False, mininterval=0.25,
        ) if show_progress else None

        eps = np.float32(1e-10)
        denom = np.float32(1.0 + max(0.0, float(self.config.poly_sparsity)))
        for c0 in range(0, voiced_idx.size, self.CHUNK_FRAMES):
            sel = voiced_idx[c0:c0 + self.CHUNK_FRAMES]
            spec = np.abs(np.fft.rfft(frames[sel] * self.window, self.n_fft, axis=1))
            V = np.ascontiguousarray(
                spec[:, self._row_lo:self._row_hi].T, dtype=np.float32,
            )  # (rows, m)
            mass = V.sum(axis=0)
            ok = mass > 0
            V /= np.maximum(mass, np.float32(1e-20))

            # projection init under the presence support constraint;
            # no warm start offline (columns decode independently)
            H = (self.WT @ V + self.H_FLOOR) * self._presence_mask_batch(V)
            for _ in range(max(1, int(self.config.poly_nmf_iters))):
                R = self.W @ H
                H_new = H * (self.WT @ (V / (R + eps))) / denom
                delta = float(np.max(np.abs(H_new - H)))
                H = H_new
                if delta <= self.MU_TOL * max(float(H.max()), 1e-12):
                    break
            R = self.W @ H
            fits = (V * R).sum(axis=0) / np.maximum(
                np.linalg.norm(V, axis=0) * np.linalg.norm(R, axis=0),
                np.float32(1e-20),
            )

            for local, i in enumerate(sel):
                if not ok[local]:
                    continue
                candidates = self._threshold_peaks(H[:, local])
                if not candidates:
                    continue
                p = pitches[i]
                p.candidate_pitches = candidates
                p.value = candidates[0][0]
                p.unvoiced_prob = float(np.clip(1.0 - float(fits[local]), 0.0, 1.0))
            if progress is not None:
                progress.update(len(sel))

        if progress is not None:
            progress.close()
        if verbose:
            print(f"[PolyPitchDetector] {n_frames} frame(s), "
                  f"{int(voiced_idx.size)} past the volume gate", flush=True)
        return pitches
