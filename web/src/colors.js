// Client-side port of the volume-coloring half of ui/Colors.py + the
// PitchData volume methods it consumes (app_logic/user/ds/PitchData.py).
// Mistake-role colors (deletion/substitution/timing) already live as a
// static SCORE_THEME constant in ScoreViewer.svelte - this file is only the
// part that needs real computation: the take's own loudness range and the
// viridis ramp built from it.

// --- ui.Colors.VIRIDIS_ANCHORS / SCORE_DIM / VOL_LIVE_FLOOR_DB ---
const VIRIDIS_ANCHORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
const SCORE_DIM = 0.9;
// Only reachable here when a take's own volume range can't be computed (no
// voiced frames) - review mode (this app) always has one once there's any
// voiced audio, so this is a rare fallback, not the common path.
const VOL_LIVE_FLOOR_DB = -42.0;

function dim(rgb, factor = SCORE_DIM) {
  return rgb.map((c) => Math.round(c * factor));
}

export function cssRgb(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

// Linear interpolation along an anchor list for a 0..1 fraction (ui.Colors.ramp).
function ramp(anchors, frac, dimmed = false) {
  const t = Math.max(0, Math.min(1, frac)) * (anchors.length - 1);
  const i = Math.min(Math.floor(t), anchors.length - 2);
  const localT = t - i;
  const [c0, c1] = [anchors[i], anchors[i + 1]];
  const rgb = c0.map((a, k) => Math.round(a + (c1[k] - a) * localT));
  return dimmed ? dim(rgb) : rgb;
}

export function viridis(frac, dimmed = false) {
  return ramp(VIRIDIS_ANCHORS, frac, dimmed);
}

// A frame's volume as a 0..1 fraction (quietest..loudest) - ui.Colors.volume_frac.
export function volumeFrac(volume, vminDb, vmaxDb) {
  if (!volume || volume <= 0) return 0;
  const db = 20 * Math.log10(volume);
  let frac;
  if (vminDb == null || vmaxDb == null || vmaxDb <= vminDb) {
    frac = (db - VOL_LIVE_FLOOR_DB) / (0 - VOL_LIVE_FLOOR_DB);
  } else {
    frac = (db - vminDb) / (vmaxDb - vminDb);
  }
  return Math.max(0, Math.min(1, frac));
}

// candidateFrames: the app's raw pitch_data.pitches payload
// ([time, candidates, volume, unvoicedProb, ..., isTransition, value]) -
// filters to voiced frames itself, mirroring PitchData.read(clean=True).
const UNVOICED_THRESHOLD = 0.9;
function voicedVolumes(pitchFrames, startTime = -Infinity, endTime = Infinity) {
  const vols = [];
  for (const frame of pitchFrames ?? []) {
    if (!frame) continue;
    const [time, , volume, unvoicedProb, , , , value] = frame;
    if (time < startTime || time > endTime) continue;
    if (value === -1 || unvoicedProb >= UNVOICED_THRESHOLD) continue;
    if (volume > 0) vols.push(volume);
  }
  return vols;
}

// (minDb, maxDb) over the whole take's voiced frames - PitchData.volume_range_db.
export function volumeRangeDb(pitchFrames) {
  const vols = voicedVolumes(pitchFrames);
  if (!vols.length) return [null, null];
  return [20 * Math.log10(Math.min(...vols)), 20 * Math.log10(Math.max(...vols))];
}

// Mean voiced-frame volume in [startTime, endTime] - PitchData.mean_volume.
export function meanVolume(pitchFrames, startTime, endTime) {
  const vols = voicedVolumes(pitchFrames, startTime, endTime);
  if (!vols.length) return 0;
  return vols.reduce((a, b) => a + b, 0) / vols.length;
}
