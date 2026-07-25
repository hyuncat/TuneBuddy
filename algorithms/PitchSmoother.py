"""
PitchSmoother — Stage 2 of the pYIN algorithm (HMM-based pitch tracking).

This module implements the second stage of probabilistic YIN (Mauch & Dixon,
"pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold
Distributions", ICASSP 2014). Stage 1 (the probabilistic YIN frontend that
emits, per frame, a set of pitch candidates each with a probability plus an
unvoiced probability) already lives in `algorithms/PitchDetector.py` and
produces `app_logic.user.ds.PitchData.Pitch` objects.

Here we take that per-frame candidate distribution and decode a single smooth
pitch track through it with a hidden Markov model, exactly as described in
section 2.2 of the paper and realised in the reference implementation
(MonoPitch / MonoPitchHMM in https://github.com/aguai/pyin):

  * Pitch space: the four-octave (configurable) range is quantised into bins of
    10 cents (0.1 semitone). Following [9] in the paper, every pitch bin is
    split into a voiced and an unvoiced state -> 2 * M states total.

  * Observation probabilities (paper eq. 6 / MonoPitchHMM::calculateObsProb):
        voiced bin m : yin_trust * p*_m      (p*_m = candidate prob in bin m)
        unvoiced     : (1 - yin_trust * sum_k p*_k) / M    (uniform over bins)

  * Transition probabilities:
        - voicing: Attune defaults to 0.9975 to stay, 0.0025 to switch;
                   the paper uses 0.99 / 0.01 (eq. 7)
        - pitch:   triangular kernel, Attune default max jump 9 bins =
                   90 cents/frame, peak at 0, normalised to sum to 1 (eq. 8)
        - assumed independent, so the full transition is their product.

  * Initial distribution: uniform over the unvoiced states.

The decoding itself is delegated to hmmlearn's C-accelerated Viterbi. hmmlearn
expects parametric emissions, but pYIN's emissions are precomputed per frame,
so we subclass `hmmlearn.base.BaseHMM` and override `_compute_log_likelihood`
to hand back our own log-observation matrix. scipy.sparse is used to assemble
the band-diagonal transition matrix compactly before densifying it for
hmmlearn.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.sparse import diags
from hmmlearn.base import BaseHMM
from tqdm import tqdm

from algorithms.Config import Config
from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording


class _PrecomputedHMM(BaseHMM):
    """A thin hmmlearn HMM whose 'emissions' are supplied directly.

    We never fit this model. We set `startprob_` and `transmat_` by hand and
    feed the per-frame log-observation matrix straight through, so that
    hmmlearn's Viterbi implementation does the heavy lifting.
    """

    def __init__(self, n_components: int, startprob: np.ndarray, transmat: np.ndarray):
        # params="" / init_params="" => don't try to learn or initialise anything
        super().__init__(
            n_components=n_components,
            params="",
            init_params="",
            implementation="log",
        )
        self.startprob_ = startprob
        self.transmat_ = transmat
        # hmmlearn validates against this when checking the input width
        self.n_features = n_components

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        # X is already the (n_frames, n_states) log-observation matrix.
        return X

    # The base class is abstract about sampling; we never sample, but provide
    # stubs so instantiation doesn't complain.
    def _init(self, X, lengths=None):
        pass

    def _generate_sample_from_state(self, state, random_state=None):
        raise NotImplementedError("PitchSmoother HMM is decode-only.")


class PitchSmoother:
    """HMM smoothing of a pYIN candidate pitch track.

    Typical use:
        smoother = PitchSmoother(config)
        smoothed = smoother.smooth(recording.pitch_data.data)   # list[Pitch]

    The output is a new list of `Pitch` objects (one per input frame) in which
    each frame holds a single decoded candidate (voiced) or no candidate with
    `unvoiced_prob = 1.0` (unvoiced). Time / volume / distance are preserved.
    """

    def __init__(self, recording: Recording=None, config: Config=None):
        """
        Args:
            recording: optional Recording to pull config from (if not supplied
                directly). If both are supplied, the explicit config takes precedence.
            config: app Config (used for tuning + the freq<->midi conversion,
                and as the default pitch range via fmin/fmax).
        """
        self.recording = recording
        self.config = config if config else recording.config
        if self.config:
            self.update_config(self.config)

    def update_config(self, config: Config):
        """update the config and all relevant parameters. importantly, this sets
        the following attributes on self to be used in the algorithm
            fmin / fmax: pitch-range bounds in Hz. Default to config.fmin/fmax
            resolution_cents: bin width in cents; paper uses 10 (= 0.1 semitone)
            max_jump_cents: maximum frame-to-frame pitch jump. Attune uses 90
                cents (9 bins of 10 cents); the paper uses 250 cents.
            switch_prob: probability of switching voiced<->unvoiced (eq. 7).
            yin_trust: weight on the YIN candidate mass when voiced. Attune
                uses 0.65; eq. 6 / MonoPitchHMM uses 0.5.
        """
        self.config = config
        RESOLUTION_CENTS = 10.0
        self.resolution = RESOLUTION_CENTS / 100.0  # in semitones
        self.switch_prob = 0.0025
        self.yin_trust = 0.65
        MAX_JUMP_CENTS = 90.0

        # --- build the pitch grid (bin index <-> midi number) ---
        fmin = config.fmin
        fmax = config.fmax
        midi_lo = config.freq_to_midi(fmin)
        midi_hi = config.freq_to_midi(fmax)
        # snap to the resolution grid so bin centres land on nice values
        self.midi_min = np.floor(midi_lo / self.resolution) * self.resolution
        midi_max = np.ceil(midi_hi / self.resolution) * self.resolution
        self.n_bins = int(round((midi_max - self.midi_min) / self.resolution)) + 1
        self.bin_midis = self.midi_min + self.resolution * np.arange(self.n_bins)

        # max jump in bins (each side of the diagonal)
        self.max_jump = int(round(MAX_JUMP_CENTS / RESOLUTION_CENTS))

        self.n_states = 2 * self.n_bins  # voiced bins, then unvoiced bins

        # precompute the (expensive, input-independent) model pieces
        self._startprob = self._build_startprob()
        self._transmat = self._build_transition()

    # ------------------------------------------------------------------ #
    # model construction
    # ------------------------------------------------------------------ #
    def _build_startprob(self) -> np.ndarray:
        """Uniform over the unvoiced states (paper: "uniformly distributed
        over the unvoiced states")."""
        p = np.zeros(self.n_states, dtype=np.float64)
        p[self.n_bins:] = 1.0 / self.n_bins
        return p

    def _triangular_pitch_block(self) -> np.ndarray:
        """The M x M pitch-transition block: a band-diagonal triangular kernel,
        each row renormalised to sum to 1 (handles truncation at the edges).
        Built with scipy.sparse.diags, then densified."""
        M, K = self.n_bins, self.max_jump
        offsets = np.arange(-K, K + 1)
        # triangular weights: peak (K+1) at offset 0, falling to 1 at +/-K
        weights = (K + 1) - np.abs(offsets)
        diagonals = [
            np.full(M - abs(o), w, dtype=np.float64)
            for o, w in zip(offsets, weights)
        ]
        block = diags(diagonals, offsets, shape=(M, M)).toarray()
        # row-normalise so each source bin's outgoing pitch distribution sums to 1
        block /= block.sum(axis=1, keepdims=True)
        return block

    def _build_transition(self) -> np.ndarray:
        """Full (2M x 2M) transition matrix = pitch transition (x) voicing
        transition, assuming independence (paper section 2.2)."""
        P = self._triangular_pitch_block()
        stay = 1.0 - self.switch_prob
        switch = self.switch_prob
        # rows/cols ordered [voiced bins ... , unvoiced bins ...]
        transmat = np.block([
            [stay * P,   switch * P],
            [switch * P, stay * P],
        ])
        return transmat

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def _midi_to_bin(self, midi: float) -> int | None:
        """Nearest pitch bin for a midi value, or None if out of range."""
        idx = int(round((midi - self.midi_min) / self.resolution))
        if idx < 0 or idx >= self.n_bins:
            return None
        return idx

    def _observation_logprobs(self, pitches: list[Pitch]) -> np.ndarray:
        """Build the (n_frames, 2M) log-observation matrix from the per-frame
        candidate distributions (paper eq. 6 / MonoPitchHMM::calculateObsProb).
        """
        T = len(pitches)
        obs = np.zeros((T, self.n_states), dtype=np.float64)

        for t, pitch in enumerate(pitches):
            voiced = np.zeros(self.n_bins, dtype=np.float64)
            if pitch is not None:
                for midi, prob in pitch.candidate_pitches:
                    b = self._midi_to_bin(midi)
                    if b is not None:
                        voiced[b] += prob

            voiced_mass = voiced.sum()                  # sum_k p*_k
            prob_pitched = self.yin_trust * voiced_mass  # really-voiced mass
            if voiced_mass > 0:
                # scale candidate distribution down to prob_pitched total
                voiced *= prob_pitched / voiced_mass
            # unvoiced mass spread uniformly across the M unvoiced states
            unvoiced_val = (1.0 - prob_pitched) / self.n_bins

            obs[t, : self.n_bins] = voiced
            obs[t, self.n_bins:] = unvoiced_val

        # log with a floor to keep -inf out of the Viterbi recursion
        np.maximum(obs, 1e-12, out=obs)
        return np.log(obs)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def decode(self, pitches: list[Pitch]) -> np.ndarray:
        """Run Viterbi and return the decoded state index per frame.

        State s < n_bins  -> voiced,   pitch bin = s
        State s >= n_bins -> unvoiced, pitch bin = s - n_bins
        """
        if not pitches:
            return np.empty(0, dtype=int)



        log_obs = self._observation_logprobs(pitches)
        hmm = _PrecomputedHMM(self.n_states, self._startprob, self._transmat)
        _logprob, states = hmm.decode(log_obs, algorithm="viterbi")

        return states

    def smooth_to_arrays(self, pitches: list[Pitch]):
        """Decode and return plain arrays, handy for plotting / debugging.

        Returns:
            times   (np.ndarray): frame times in seconds (NaN where unknown).
            midi    (np.ndarray): decoded midi number per frame (NaN if unvoiced).
            voiced  (np.ndarray): bool mask, True where the frame is voiced.
        """
        states = self.decode(pitches)
        T = len(pitches)
        times = np.full(T, np.nan)
        midi = np.full(T, np.nan)
        voiced = np.zeros(T, dtype=bool)

        for t, (p, s) in enumerate(zip(pitches, states)):
            if p is not None:
                times[t] = p.time
            is_voiced = s < self.n_bins
            voiced[t] = is_voiced
            if is_voiced:
                midi[t] = self.bin_midis[s]
        return times, midi, voiced

    def smooth(self, pitches: list[Pitch], show_progress: bool=False, verbose: bool = False) -> list[Pitch]:
        """Decode the track and return a new list of `Pitch` objects.

        Each voiced frame keeps a single candidate (the decoded midi at 10-cent
        resolution) with probability 1.0 and unvoiced_prob 0.0; each unvoiced
        frame has no candidates and unvoiced_prob 1.0. Frame time, volume and
        distance are copied from the corresponding input pitch.
        """
        if verbose:
            print("Starting pitch smoothing... ", end="", flush=True)
        start = time.time()

        states = self.decode(pitches)
        out: list[Pitch] = []
        frame_states = zip(pitches, states)
        if show_progress:
            frame_states = tqdm(
                frame_states,
                total=len(pitches),
                desc="Converting smoothed pitches",
                leave=False,
                mininterval=0.25,
            )

        for p, s in frame_states:
            is_voiced = s < self.n_bins
            if p is None:
                # keep the time grid intact even if the frame was empty
                out.append(None)
                continue

            if is_voiced:
                midi = float(self.bin_midis[s])
                candidates = [(midi, 1.0)]
                unvoiced_prob = 0.0
            else:
                candidates = []
                unvoiced_prob = 1.0

            out.append(Pitch(
                time=p.time,
                value=midi if is_voiced else -1,
                candidates=candidates,
                volume=p.volume,
                unvoiced_prob=unvoiced_prob,
                live_distance=p.live_distance,
                config=p.config,
            ))
        
        if verbose:
            print(f"Done! Took {time.time() - start:.2f} sec.")
        return out

    # convenience: operate directly on a PitchData container
    def smooth_pitch_data(self, pitch_data):
        """Smooth a `PitchData` in place-ish: returns a new PitchData with the
        same time indexing but the smoothed pitch track."""
        from app_logic.user.ds.PitchData import PitchData
        smoothed = self.smooth(list(pitch_data.data))
        out = PitchData(config=pitch_data.config)
        out.load(smoothed)
        return out
