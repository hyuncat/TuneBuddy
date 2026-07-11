// JS port of algorithms/PitchDetector.py — the per-frame real-time hot path only.
//
// Scope: this file ports `detect_pitch` and everything it calls (preprocessing,
// FFT autocorrelation, CMNDF, peak-picking, pYIN-style probability assignment,
// parabolic interpolation). It intentionally leaves out:
//   - Qt signals / threading (`run`, `_run`, `stop`) — that's the AudioWorklet's
//     job on this side, wired up separately later.
//   - The offline whole-file `detect_pitches` batch method, which uses a
//     percentile-based volume gate over the entire recording; the streaming
//     path here uses the running-peak gate instead, matching `detect_pitch`.
//
// Two known deviations from the Python reference, worth validating empirically
// rather than assuming away:
//   1. `autocorrelationFFT`'s zero-padded FFT size is rounded up to the next
//      power of two, instead of Python's "nice" mixed-radix composite sizes
//      (scipy/numpy pick those for raw speed). This only changes performance,
//      not correctness: the padded size only needs to be >= frameSize + tauMax
//      to avoid circular-wraparound contaminating the autocorrelation lags we
//      actually read out ([0, tauMax)); that invariant is preserved here.
//   2. `findAcfPeaks`' local-maxima scan is a strict neighbor comparison and
//      doesn't special-case flat plateaus the way scipy.signal.find_peaks does.
//      Unlikely to matter for a continuous autocorrelation curve, but flagged.

// ---------------------------------------------------------------------------
// Config — mirrors algorithms/Config.py's pitch-detection-relevant fields.
// ---------------------------------------------------------------------------
export class Config {
  constructor({
    sr = 44100,
    fmin = 196.0,
    fmax = 3000.0,
    tuning = 440.0,
    minVolume = 0.05,
    maxVolume = 0.95,
    w1 = 1024 * 4,
    h1 = 128,
  } = {}) {
    this.sr = sr;
    this.fmin = fmin;
    this.fmax = fmax;
    this.tuning = tuning;
    this.minVolume = minVolume;
    this.maxVolume = maxVolume;
    this.w1 = w1;
    this.h1 = h1;
  }

  freqToMidi(freq) {
    if (freq <= 0) return -1;
    return 69 + 12 * Math.log2(freq / this.tuning);
  }

  midiToFreq(midiNum) {
    return this.tuning * Math.pow(2, (midiNum - 69) / 12);
  }
}

// ---------------------------------------------------------------------------
// Gamma / beta-pdf — needed for the pYIN threshold prior (scipy.stats.beta.pdf
// has no built-in JS equivalent). Standard Lanczos approximation (g=7, n=9).
// ---------------------------------------------------------------------------
const LANCZOS_G = 7;
const LANCZOS_COEF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function gamma(z) {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }
  z -= 1;
  let x = LANCZOS_COEF[0];
  for (let i = 1; i < LANCZOS_G + 2; i++) {
    x += LANCZOS_COEF[i] / (z + i);
  }
  const t = z + LANCZOS_G + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

export function betaPdfValue(x, a, b) {
  const B = (gamma(a) * gamma(b)) / gamma(a + b);
  return (Math.pow(x, a - 1) * Math.pow(1 - x, b - 1)) / B;
}

// Mirrors Config.py's threshold_prior(): thresholds = linspace(0,1,n+1)[1:],
// beta_pdf = beta.pdf(thresholds, a, b) / n_thresholds.
export function thresholdPrior(nThresholds = 100, a = 2, b = 34 / 3) {
  const thresholds = new Float64Array(nThresholds);
  const betaPdf = new Float64Array(nThresholds);
  for (let i = 1; i <= nThresholds; i++) {
    const t = i / nThresholds;
    thresholds[i - 1] = t;
    betaPdf[i - 1] = betaPdfValue(t, a, b) / nThresholds;
  }
  return { betaPdf, thresholds };
}

// ---------------------------------------------------------------------------
// FFT — iterative radix-2 Cooley-Tukey, plus real-input rfft/irfft wrappers
// (numpy.fft.rfft/irfft equivalents, used for the Wiener-Khinchin autocorrelation).
// ---------------------------------------------------------------------------
function nextPow2(n) {
  return 1 << Math.ceil(Math.log2(n));
}

// In-place FFT on parallel re/im arrays (length must be a power of two).
// inverse=true computes the IFFT, normalized by 1/n (matches numpy's default
// "backward" normalization: unnormalized forward, 1/n on the inverse).
function fftInPlace(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len / 2;
    const ang = (sign * 2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

// numpy.fft.rfft(x, n) equivalent: zero-pads x to length n, forward-FFTs, and
// keeps only the first n/2+1 bins (real input -> conjugate-symmetric spectrum).
function rfft(x, n) {
  const re = new Float64Array(n); // zero-initialized -> the zero-padding
  re.set(x);
  const im = new Float64Array(n);
  fftInPlace(re, im, false);
  const half = n / 2 + 1;
  return { re: re.slice(0, half), im: im.slice(0, half), n };
}

// numpy.fft.irfft(spec, n) equivalent: rebuilds the full conjugate-symmetric
// spectrum from the half-spectrum, inverse-FFTs, and returns the real part.
function irfft(specRe, specIm, n) {
  const fullRe = new Float64Array(n);
  const fullIm = new Float64Array(n);
  const half = specRe.length; // n/2 + 1
  for (let k = 0; k < half; k++) {
    fullRe[k] = specRe[k];
    fullIm[k] = specIm[k];
  }
  for (let k = 1; k < n / 2; k++) {
    fullRe[n - k] = specRe[k];
    fullIm[n - k] = -specIm[k];
  }
  fftInPlace(fullRe, fullIm, true);
  return fullRe; // imaginary part is ~0 by construction (conjugate symmetry)
}

// ---------------------------------------------------------------------------
// Complex-number helpers (plain [re, im] pairs) — only needed for the
// Butterworth bandpass filter design below.
// ---------------------------------------------------------------------------
const cAdd = (a, b) => [a[0] + b[0], a[1] + b[1]];
const cSub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const cMul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cScale = (a, s) => [a[0] * s, a[1] * s];
function cDiv(a, b) {
  const denom = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / denom, (a[1] * b[0] - a[0] * b[1]) / denom];
}
function cSqrt(a) {
  const r = Math.hypot(a[0], a[1]);
  const re = Math.sqrt((r + a[0]) / 2);
  let im = Math.sqrt(Math.max(r - a[0], 0) / 2);
  if (a[1] < 0) im = -im;
  return [re, im];
}

// ---------------------------------------------------------------------------
// Butterworth bandpass filter design (order N=2 prototype -> 2 SOS sections),
// matching scipy.signal.iirfilter(N=2, Wn=[fmin,fmax], btype='bandpass',
// ftype='butter', output='sos', fs=fs). Steps: (1) analog Butterworth lowpass
// prototype poles, (2) lowpass->bandpass transform, (3) bilinear transform
// with frequency prewarping, (4) pair into second-order sections.
//
// This is the part most worth validating numerically against the Python
// reference — it's standard DSP, but there's a lot of arithmetic to get
// exactly right by hand.
// ---------------------------------------------------------------------------
export function designButterworthBandpassSOS(fmin, fmax, fs) {
  const N = 2;

  // (1) analog Butterworth lowpass prototype: N poles on the unit circle in
  // the left half-plane, no zeros, unit gain.
  const protoPoles = [];
  for (let k = 0; k < N; k++) {
    const theta = (Math.PI * (2 * k + N + 1)) / (2 * N);
    protoPoles.push([Math.cos(theta), Math.sin(theta)]);
  }

  // (2) prewarp the band edges (bilinear-transform correction), then apply
  // the lowpass -> bandpass frequency transform (scipy's lp2bp_zpk).
  const omega1 = 2 * fs * Math.tan((Math.PI * fmin) / fs);
  const omega2 = 2 * fs * Math.tan((Math.PI * fmax) / fs);
  const w0 = Math.sqrt(omega1 * omega2);
  const bw = omega2 - omega1;

  const bpPoles = [];
  for (const p of protoPoles) {
    const pLp = cScale(p, bw / 2);
    const disc = cSqrt(cSub(cMul(pLp, pLp), [w0 * w0, 0]));
    bpPoles.push(cAdd(pLp, disc));
    bpPoles.push(cSub(pLp, disc));
  }
  const bpZeros = new Array(N).fill(0).map(() => [0, 0]); // N zeros at s=0
  const bpGain = Math.pow(bw, N); // prototype gain was 1

  // (3) bilinear transform: s -> z = (2fs + s) / (2fs - s). Zeros introduced
  // by the pole/zero degree difference map to z = -1.
  const fs2 = 2 * fs;
  const zZeros = bpZeros.map((z) => cDiv(cAdd([fs2, 0], z), cSub([fs2, 0], z)));
  const zPoles = bpPoles.map((p) => cDiv(cAdd([fs2, 0], p), cSub([fs2, 0], p)));
  while (zZeros.length < zPoles.length) zZeros.push([-1, 0]);

  let numProd = [1, 0];
  let denProd = [1, 0];
  for (const z of bpZeros) numProd = cMul(numProd, cSub([fs2, 0], z));
  for (const p of bpPoles) denProd = cMul(denProd, cSub([fs2, 0], p));
  const zGain = bpGain * cDiv(numProd, denProd)[0];

  // (4) pair each conjugate pole pair (and its corresponding zero pair) into
  // one second-order section. zPoles/zZeros were built two-at-a-time above,
  // so consecutive pairs already line up.
  const sections = [];
  for (let i = 0; i < zPoles.length; i += 2) {
    const p = zPoles[i]; // paired with zPoles[i+1] = conj(p)
    const z1 = zZeros[i];
    const z2 = zZeros[i + 1];
    const a1 = -2 * p[0];
    const a2 = p[0] * p[0] + p[1] * p[1];
    const zSum = cAdd(z1, z2)[0]; // zeros are real (+-1) here
    const zProd = cMul(z1, z2)[0];
    sections.push({ b0: 1, b1: -zSum, b2: zProd, a1, a2 });
  }
  // apply the overall gain into the first section (equivalent mathematically
  // to splitting it across sections; only affects rounding, not the result)
  sections[0].b0 *= zGain;
  sections[0].b1 *= zGain;
  sections[0].b2 *= zGain;

  return sections;
}

// Direct Form II Transposed, cascaded section-by-section, zero initial state
// (matches scipy.signal.sosfilt's default zi=None).
export function sosFilter(sections, x) {
  let y = Float64Array.from(x);
  for (const { b0, b1, b2, a1, a2 } of sections) {
    const out = new Float64Array(y.length);
    let z1 = 0;
    let z2 = 0;
    for (let i = 0; i < y.length; i++) {
      const input = y[i];
      const o = b0 * input + z1;
      z1 = b1 * input - a1 * o + z2;
      z2 = b2 * input - a2 * o;
      out[i] = o;
    }
    y = out;
  }
  return y;
}

// ---------------------------------------------------------------------------
// PitchDetector — the per-frame YIN/pYIN core.
// ---------------------------------------------------------------------------
export class PitchDetector {
  constructor(config) {
    this.config = config;
    this.SR = config.sr;

    const paddedFmin = config.midiToFreq(config.freqToMidi(config.fmin) - 0.5);
    const paddedFmax = config.midiToFreq(config.freqToMidi(config.fmax) + 0.5);
    this.tauMax = Math.floor(config.sr / paddedFmin);
    this.tauMin = Math.floor(config.sr / paddedFmax);

    this.UNVOICED_PROB = 0.01;
    this.N_THRESHOLDS = 100;
    const { betaPdf, thresholds } = thresholdPrior(this.N_THRESHOLDS);
    this.betaPdf = betaPdf;
    this.thresholds = thresholds;

    this.FRAME_SIZE = config.w1;
    this.HOP_SIZE = config.h1;

    // running peak volume, used for the streaming (not percentile) volume gate
    this.streamVolumePeak = 0;

    // designed once at init, applied every frame in preprocessAudio — same
    // effective band as PitchDetector.py's bandpass_filter call site
    // (fmin*0.8 .. fmax*1.2).
    this.bandpassSections = designButterworthBandpassSOS(
      config.fmin * 0.8,
      config.fmax * 1.2,
      config.sr
    );
  }

  // --- AUDIO PREPROCESSING ---
  preprocessAudio(x) {
    const n = x.length;
    if (n === 0) return { x: new Float64Array(0), volume: 0 };

    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;

    const centered = new Float64Array(n);
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      centered[i] = x[i] - mean;
      sumSq += centered[i] * centered[i];
    }
    const volume = Math.sqrt(sumSq / n);

    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(centered[i]));
    if (peak === 0) return { x: new Float64Array(n), volume: 0 };

    const normalized = new Float64Array(n);
    for (let i = 0; i < n; i++) normalized[i] = centered[i] / peak;

    const filtered = sosFilter(this.bandpassSections, normalized);
    return { x: filtered, volume };
  }

  // --- FREQUENCY-DOMAIN AUTOCORRELATION (Wiener-Khinchin) ---
  autocorrelationFFT(x) {
    const w = x.length;
    const tauMaxLocal = Math.min(this.tauMax, w);
    const minFftSize = w + tauMaxLocal;
    const n = nextPow2(minFftSize); // see file-header note on this simplification

    const spec = rfft(x, n);
    const half = spec.re.length;

    const amplitudes = new Float64Array(half);
    for (let i = 0; i < half; i++) amplitudes[i] = Math.hypot(spec.re[i], spec.im[i]);

    // psd = fft_x * conj(fft_x) = |fft_x|^2 (real-valued)
    const psdRe = new Float64Array(half);
    for (let i = 0; i < half; i++) {
      psdRe[i] = spec.re[i] * spec.re[i] + spec.im[i] * spec.im[i];
    }
    const psdIm = new Float64Array(half); // all zero

    const full = irfft(psdRe, psdIm, n);
    const autocorrelation = full.slice(0, tauMaxLocal);
    return { autocorrelation, amplitudes };
  }

  // --- CUMULATIVE MEAN NORMALIZED DIFFERENCE FUNCTION ---
  cmndf(x, acf) {
    let r0 = 0;
    for (let i = 0; i < x.length; i++) r0 += x[i] * x[i];

    const diffFct = new Float64Array(acf.length);
    for (let t = 0; t < acf.length; t++) diffFct[t] = 2 * r0 - 2 * acf[t];
    diffFct[0] = 0;
    for (let t = 0; t < diffFct.length; t++) diffFct[t] = Math.abs(diffFct[t]);

    let maxD = -Infinity;
    let minD = Infinity;
    for (let t = 0; t < diffFct.length; t++) {
      if (diffFct[t] > maxD) maxD = diffFct[t];
      if (diffFct[t] < minD) minD = diffFct[t];
    }
    const range = maxD - minD;
    for (let t = 0; t < diffFct.length; t++) diffFct[t] /= range;

    const cmndf = new Float64Array(this.tauMax);
    cmndf[0] = 1;
    let totalDiff = 1;
    for (let tau = 1; tau < this.tauMax; tau++) {
      totalDiff += diffFct[tau];
      const avgDiff = totalDiff / tau;
      cmndf[tau] = diffFct[tau] / avgDiff;
    }
    return cmndf;
  }

  // --- PEAK-PICKING (prominence-based, mirrors scipy.signal.find_peaks) ---
  findAcfPeaks(acf) {
    let maxV = -Infinity;
    let minV = Infinity;
    for (let i = 0; i < acf.length; i++) {
      if (acf[i] > maxV) maxV = acf[i];
      if (acf[i] < minV) minV = acf[i];
    }
    const baseProminence = Math.abs((maxV - minV) / 2);

    const candidates = [];
    for (let i = 1; i < acf.length - 1; i++) {
      if (acf[i - 1] < acf[i] && acf[i] > acf[i + 1]) candidates.push(i);
    }

    const prominenceAt = (idx) => {
      const peakVal = acf[idx];
      let leftMin = peakVal;
      for (let j = idx - 1; j >= 0; j--) {
        if (acf[j] > peakVal) break;
        if (acf[j] < leftMin) leftMin = acf[j];
      }
      let rightMin = peakVal;
      for (let j = idx + 1; j < acf.length; j++) {
        if (acf[j] > peakVal) break;
        if (acf[j] < rightMin) rightMin = acf[j];
      }
      return peakVal - Math.max(leftMin, rightMin);
    };

    const n = 5;
    let peaks = [];
    for (let i = 0; i < n; i++) {
      const p = baseProminence - baseProminence * (i / n);
      peaks = candidates.filter((idx) => prominenceAt(idx) >= p);
      if (peaks.length > 0) break;
    }

    if (peaks.length === 0) {
      // fallback: global max within [tauMin, tauMax)
      let best = this.tauMin;
      let bestVal = acf[this.tauMin];
      for (let j = this.tauMin; j < this.tauMax; j++) {
        if (acf[j] > bestVal) {
          bestVal = acf[j];
          best = j;
        }
      }
      peaks = [best];
    }
    return peaks;
  }

  // --- YIN ABSOLUTE-THRESHOLD SEARCH ---
  findPitch(cdf, acfPeaks, threshold) {
    for (let i = 0; i < acfPeaks.length; i++) {
      const idx = acfPeaks[i];
      if (cdf[idx] <= threshold) {
        return { tau0: idx, peakIndex: i, isVoiced: true };
      }
    }
    let bestI = 0;
    let bestVal = cdf[acfPeaks[0]];
    for (let i = 1; i < acfPeaks.length; i++) {
      if (cdf[acfPeaks[i]] < bestVal) {
        bestVal = cdf[acfPeaks[i]];
        bestI = i;
      }
    }
    return { tau0: acfPeaks[bestI], peakIndex: bestI, isVoiced: false };
  }

  // --- pYIN PROBABILITY ASSIGNMENT ACROSS ALL THRESHOLDS ---
  pitchProbabilities(acfPeaks, cdf) {
    const pitchProbs = new Float64Array(acfPeaks.length);
    for (let i = 0; i < this.thresholds.length; i++) {
      const threshold = this.thresholds[i];
      const { tau0, peakIndex, isVoiced } = this.findPitch(cdf, acfPeaks, threshold);
      if (isVoiced && tau0 <= this.tauMax && tau0 >= this.tauMin) {
        pitchProbs[peakIndex] += this.betaPdf[i];
      } else {
        pitchProbs[peakIndex] += this.betaPdf[i] * this.UNVOICED_PROB;
      }
    }
    let sum = 0;
    for (let i = 0; i < pitchProbs.length; i++) sum += pitchProbs[i];
    return { pitchProbs, unvoicedProb: 1 - sum };
  }

  // --- PEAK REFINEMENT ---
  parabolicInterpolation(acf, acfPeak) {
    const x = acfPeak;
    if (x <= 0 || x >= acf.length - 1) return x;
    const y1 = acf[x - 1];
    const y2 = acf[x];
    const y3 = acf[x + 1];
    const denom = 2 * (y1 - 2 * y2 + y3);
    if (denom === 0) return x;
    return x + (y1 - y3) / denom;
  }

  // --- THE MAIN ENTRY POINT: one frame in, one Pitch estimate out ---
  detectPitch(x, startTime) {
    const unvoiced = { time: startTime, candidates: [], volume: 0, unvoicedProb: 1.0 };

    let allZero = true;
    for (let i = 0; i < x.length; i++) {
      if (x[i] !== 0) {
        allZero = false;
        break;
      }
    }
    if (allZero) return unvoiced;

    const { x: proc, volume } = this.preprocessAudio(x);

    this.streamVolumePeak = Math.max(this.streamVolumePeak, volume);
    const minVolume = this.streamVolumePeak * Math.max(0, this.config.minVolume);

    let anyNonZero = false;
    for (let i = 0; i < proc.length; i++) {
      if (proc[i] !== 0) {
        anyNonZero = true;
        break;
      }
    }
    if (volume < minVolume || !anyNonZero) {
      return { ...unvoiced, volume };
    }

    const { autocorrelation: acf } = this.autocorrelationFFT(proc);
    const cdf = this.cmndf(proc, acf);
    const acfPeaks = this.findAcfPeaks(acf);
    const { pitchProbs, unvoicedProb } = this.pitchProbabilities(acfPeaks, cdf);

    const candidates = acfPeaks.map((tau, i) => {
      const freq = this.config.sr / this.parabolicInterpolation(acf, tau);
      const midi = this.config.freqToMidi(freq);
      return [midi, pitchProbs[i]];
    });
    candidates.sort((a, b) => b[1] - a[1]);

    return { time: startTime, candidates, volume, unvoicedProb };
  }
}
