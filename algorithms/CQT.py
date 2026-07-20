import numpy as np
from scipy.sparse import csr_matrix

from algorithms.Config import Config


class CQT:
    """Lightweight semitone filterbank over one raw pitch-analysis FFT frame.

    This is intentionally a pseudo-CQT for display: triangular, L1-normalized
    filters centered at tuning-aware MIDI frequencies. Low bins are floored to
    1.5 FFT-bin half-widths, trading frequency resolution for a stable heatmap
    without the cost of a variable-window transform.
    """

    FLOOR_DB = -120.0

    def __init__(self, config: Config):
        self.config = config
        self.sr = int(config.sr)
        self.n_fft = int(config.w1)
        self.midi_min = int(config.cqt_midi_min)
        self.midi_max = int(config.cqt_midi_max)
        self.midis = np.arange(self.midi_min, self.midi_max + 1, dtype=float)
        self.window = np.hanning(self.n_fft).astype(np.float64)
        self.scale = 2.0 / max(float(self.window.sum()), np.finfo(float).eps)
        self.filterbank = self._build_filterbank()

    def _build_filterbank(self) -> csr_matrix:
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
        centers = np.asarray([self.config.midi_to_freq(m) for m in self.midis])
        fft_bin_hz = self.sr / self.n_fft
        rows, cols, values = [], [], []
        for row, center in enumerate(centers):
            # Neighbor midpoints form the natural semitone cell. Guarantee a
            # usable footprint where the FFT is coarser than that cell.
            lower_neighbor = self.config.midi_to_freq(self.midis[row] - 1)
            upper_neighbor = self.config.midi_to_freq(self.midis[row] + 1)
            half_width = max(
                center - 0.5 * (lower_neighbor + center),
                0.5 * (center + upper_neighbor) - center,
                1.5 * fft_bin_hz,
            )
            weights = np.maximum(1.0 - np.abs(freqs - center) / half_width, 0.0)
            nz = np.flatnonzero(weights > 0)
            if nz.size == 0:
                nz = np.asarray([int(np.argmin(np.abs(freqs - center)))])
                weights[nz] = 1.0
            w = weights[nz]
            w /= w.sum()
            rows.extend([row] * len(nz))
            cols.extend(nz.tolist())
            values.extend(w.tolist())
        return csr_matrix(
            (values, (rows, cols)),
            shape=(len(centers), len(freqs)),
            dtype=np.float64,
        )

    def power_db(self, x: np.ndarray) -> np.ndarray:
        """One raw centered frame -> semitone-bin power in clipped dBFS."""
        frame = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(frame) < self.n_fft:
            frame = np.pad(frame, (0, self.n_fft - len(frame)))
        elif len(frame) > self.n_fft:
            frame = frame[:self.n_fft]
        spectrum = np.abs(np.fft.rfft(frame * self.window)) * self.scale
        power = np.asarray(self.filterbank @ (spectrum * spectrum)).reshape(-1)
        db = 10.0 * np.log10(np.maximum(power, 10.0 ** (self.FLOOR_DB / 10.0)))
        return np.clip(db, self.FLOOR_DB, 0.0).astype(np.float32)
