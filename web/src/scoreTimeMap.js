// Port of ui/time/ScoreTimeMap.py: a piecewise-linear correspondence between
// the app's note/MIDI timeline and the Verovio score-viewer's own timeline,
// anchored at measure barlines.
//
// The app's MIDI/NoteData onsets are the single source of timing truth: the
// audio player and the pitch overlay both run straight off them. Verovio,
// however, builds its OWN timemap by re-integrating the *notated* durations
// of the MusicXML that music21 exports from the MIDI - a lossy round-trip
// (quantization, tied/collapsed notes) whose timeline drifts from the MIDI
// and *accumulates* over the piece. This map pins the two timelines together
// at every barline - the one landmark that's unambiguously 1:1 between them
// - and interpolates linearly within each bar. Both axes are in the score's
// ORIGINAL-tempo timeframe (bpmOg); viewerTime/appTime handle the current-
// tempo conversion, so the anchors stay valid across tempo changes and only
// need rebuilding when the score is re-laid-out.
//
// Until anchors are installed, both directions pass time through unchanged.
export class ScoreTimeMap {
  constructor() {
    this._app = [];
    this._vero = [];
  }

  // Installs paired barline onsets, already index-aligned (appTimes[k] and
  // veroTimes[k] are the same measure). Keeps only points strictly
  // increasing on BOTH axes, so the map stays single-valued and invertible
  // even if a degenerate/repeated bar slips in.
  setAnchors(appTimes, veroTimes) {
    const app = [];
    const vero = [];
    const n = Math.min(appTimes?.length ?? 0, veroTimes?.length ?? 0);
    for (let i = 0; i < n; i++) {
      const a = Number(appTimes[i]);
      const v = Number(veroTimes[i]);
      if (app.length && !(a > app[app.length - 1] + 1e-9 && v > vero[vero.length - 1] + 1e-9)) {
        continue; // skip non-monotone points (repeats / degenerate bars)
      }
      app.push(a);
      vero.push(v);
    }
    this._app = app;
    this._vero = vero;
  }

  clear() {
    this._app = [];
    this._vero = [];
  }

  get ready() {
    return this._app.length >= 2;
  }

  // app (original-tempo) time -> Verovio time. Identity until anchored.
  toViewer(appT) {
    return interp(this._app, this._vero, appT);
  }

  // Verovio time -> app (original-tempo) time, the inverse of toViewer.
  fromViewer(veroT) {
    return interp(this._vero, this._app, veroT);
  }

  // Wall-clock app time (current tempo) -> Verovio cursor time: undo the
  // transpose offset (a clip-resize shifts the score), then the tempo
  // change (-> the original-tempo timeframe the anchors live in), then the
  // barline map. Falls back to the plain scalar until anchored.
  // score: {bpm, bpmOg, transposeOffset}
  viewerTime(t, score) {
    const bpmOg = score?.bpmOg || score?.bpm;
    if (!bpmOg) return t;
    const ogT = (t - (score?.transposeOffset ?? 0)) * score.bpm / bpmOg;
    return this.toViewer(ogT);
  }

  // Inverse of viewerTime: a Verovio-timeline time back onto the app's
  // wall-clock timeline (barline map, then redo tempo + transpose offset).
  appTime(viewerT, score) {
    const bpmOg = score?.bpmOg || score?.bpm;
    if (!bpmOg || !score?.bpm) return viewerT;
    const ogT = this.fromViewer(viewerT);
    return (ogT * bpmOg) / score.bpm + (score?.transposeOffset ?? 0);
  }
}

function interp(xs, ys, x) {
  if (xs.length < 2) return x; // not anchored yet - pass through unchanged
  let k = bisectRight(xs, x) - 1;
  k = Math.max(0, Math.min(k, xs.length - 2)); // clamp; extrapolate on the end segments
  const x0 = xs[k];
  const x1 = xs[k + 1];
  if (x1 === x0) return ys[k];
  return ys[k] + ((x - x0) * (ys[k + 1] - ys[k])) / (x1 - x0);
}

function bisectRight(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < arr[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}
