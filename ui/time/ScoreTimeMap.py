import bisect


class ScoreTimeMap:
    """Piecewise-linear correspondence between the app's note/MIDI timeline and
    the Verovio score-viewer timeline, anchored at measure barlines.

    The app's MIDI / NoteData onsets are the single source of timing truth: the
    audio player and the GuitarHero piano-roll both run straight off them.
    Verovio, however, builds its OWN timemap by re-integrating the *notated*
    durations of the MusicXML that music21 exports from the MIDI — a lossy
    round-trip (quantization, tied / collapsed notes) whose timeline drifts from
    the MIDI and *accumulates* over the piece. A single global tempo scalar
    (bpm/bpm_og) can't undo that nonlinear drift, so on long files the score
    cursor falls progressively out of sync with what's actually sounding.

    This map pins the two timelines together at every barline — the one landmark
    that is unambiguously 1:1 between them — and interpolates linearly within
    each bar, so the cursor tracks whatever note is sounding instead of
    re-deriving its own position from the drifting timemap. Both axes are in the
    score's ORIGINAL-tempo timeframe (bpm_og); the caller handles current-tempo
    conversion with the bpm/bpm_og scalar, so these anchors stay valid across
    tempo changes / resize and only need rebuilding when the score is re-laid-out.

    Until anchors are installed (the Verovio onsets are pulled asynchronously),
    both directions pass time through unchanged, so the cursor simply falls back
    to the plain scalar behaviour.
    """

    def __init__(self):
        self._app: list[float] = []   # barline onsets in app (original-tempo) time
        self._vero: list[float] = []  # the same barlines in Verovio's timeframe

    def set_anchors(self, app_times, vero_times) -> None:
        """Install paired barline onsets, already index-aligned (app_times[k] and
        vero_times[k] are the same measure). Keeps only the leading run that is
        strictly increasing on BOTH axes, so the map stays single-valued and
        invertible even if a degenerate / repeated bar slips in."""
        app: list[float] = []
        vero: list[float] = []
        for a, v in zip(app_times, vero_times):
            a, v = float(a), float(v)
            if app and not (a > app[-1] + 1e-9 and v > vero[-1] + 1e-9):
                continue  # skip non-monotone points (repeats / degenerate bars)
            app.append(a)
            vero.append(v)
        self._app, self._vero = app, vero

    def clear(self) -> None:
        self._app, self._vero = [], []

    @property
    def ready(self) -> bool:
        return len(self._app) >= 2

    def to_viewer(self, app_t: float) -> float:
        """app (original-tempo) time -> Verovio time. Identity until anchored."""
        return self._interp(self._app, self._vero, app_t)

    def from_viewer(self, vero_t: float) -> float:
        """Verovio time -> app (original-tempo) time, the inverse of to_viewer.
        Identity until anchored."""
        return self._interp(self._vero, self._app, vero_t)

    def viewer_time(self, t: float, score_data) -> float:
        """Wall-clock app time (current tempo) -> Verovio cursor time: undo the
        transpose offset (a clip-resize shifts the score), then the tempo change
        (-> the original-tempo timeframe the anchors live in), then the barline
        map. Falls back to the plain scalar until anchored."""
        bpm_og = score_data.bpm_og or score_data.bpm
        if not bpm_og:
            return t
        og_t = (t - score_data.transpose_offset) * score_data.bpm / bpm_og
        return self.to_viewer(og_t)

    def app_time(self, viewer_t: float, score_data) -> float:
        """Inverse of viewer_time: a Verovio-timeline time back onto the app's
        wall-clock timeline (barline map, then redo tempo + transpose offset)."""
        bpm_og = score_data.bpm_og or score_data.bpm
        if not bpm_og or not score_data.bpm:
            return viewer_t
        og_t = self.from_viewer(viewer_t)
        return og_t * bpm_og / score_data.bpm + score_data.transpose_offset

    @staticmethod
    def _interp(xs: list[float], ys: list[float], x: float) -> float:
        if len(xs) < 2:
            return x  # not anchored yet -> pass through unchanged
        k = bisect.bisect_right(xs, x) - 1
        k = max(0, min(k, len(xs) - 2))  # clamp; extrapolate on the end segments
        x0, x1 = xs[k], xs[k + 1]
        if x1 == x0:
            return ys[k]
        return ys[k] + (x - x0) * (ys[k + 1] - ys[k]) / (x1 - x0)
