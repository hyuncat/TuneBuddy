<script>
  // Static note-level piano-roll matching ui/GuitarHero.py's color system as
  // closely as a note-level (not frame-level) view can:
  //   - the letter-based MIDI "piano key" rainbow background
  //     (GuitarHero.MidiBackground.LETTER_RGB, sharps darkened 70%)
  //   - insertion = GREEN, deletion = RED (GuitarHero.init_colors - the
  //     opposite of the "errors are red" scheme this file used before)
  //   - a substitution's color is a continuous green->red HSV ramp based on
  //     how far its pitch distance is beyond tolerance, not a flat color
  //     (GuitarHero._build_align_brushes / get_align_distance_brush,
  //     ALIGN_MAX_MULT = 4.0)
  //   - dashed white lines connecting matched/substituted pairs
  //     (GuitarHero.match_lines)
  //   - a solid green playhead line at the current playback time
  //     (GuitarHero.timeline, pg.InfiniteLine, colors['timeline'] =
  //     mkPen(0, 255, 0, 255))
  import { noteFromArray } from "./mistakes.js";

  let {
    scoreNotes,
    userNotes,
    pairs,
    pitchMistakes,
    pitchTolerance,
    pitchFrames = null,
    currentTime = null,
  } = $props();

  // --- GuitarHero.MidiBackground, ported exactly ---
  const LETTER_RGB = {
    A: [230, 60, 60],
    B: [255, 150, 40],
    C: [245, 220, 70],
    D: [70, 200, 90],
    E: [70, 140, 240],
    F: [100, 90, 210],
    G: [170, 90, 210],
  };
  const PC_TO_LETTER = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"];
  const isSharp = (m) => [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
  function midiToRgba(m, alpha = 0.2) {
    const letter = PC_TO_LETTER[((m % 12) + 12) % 12];
    let [r, g, b] = LETTER_RGB[letter];
    if (isSharp(m)) {
      r *= 0.7;
      g *= 0.7;
      b *= 0.7;
    }
    return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;
  }

  // --- GuitarHero._build_align_brushes / get_align_distance_brush, ported ---
  // QColor.setHsv(hue, 255, 255) (full saturation+value) is exactly
  // hsl(hue, 100%, 50%) - same color space, no conversion loss.
  const ALIGN_MAX_MULT = 4.0;
  function alignHue(distance, tolerance) {
    const greenThresh = Math.max(tolerance, 0);
    const maxDist = Math.max(ALIGN_MAX_MULT * greenThresh, greenThresh + 0.05);
    const d = Math.min(Math.abs(distance), maxDist);
    if (d <= greenThresh) return 120;
    const frac = Math.max(0, Math.min(1, (d - greenThresh) / (maxDist - greenThresh)));
    return 120 * (1 - frac);
  }
  const hsl = (hue, alpha = 1) => `hsla(${hue}, 100%, 50%, ${alpha})`;

  const SCORE_NOTE_COLOR = "rgba(255, 255, 255, 0.78)";
  const DELETION_COLOR = "rgba(255, 0, 0, 0.78)";
  const INSERTION_COLOR = "rgba(0, 200, 0, 0.78)";
  const MATCH_LINE_COLOR = "rgba(255, 255, 255, 0.55)";

  let mistakeByUserIdx = $derived(
    new Map(pitchMistakes.filter((m) => m.userIdx != null).map((m) => [m.userIdx, m]))
  );
  let deletedScoreIndices = $derived(
    new Set(pitchMistakes.filter((m) => m.type === "deletion").map((m) => m.scoreIdx))
  );

  let scoreNotesParsed = $derived(scoreNotes.map(noteFromArray));
  let userNotesParsed = $derived(userNotes.map(noteFromArray));

  // --- GuitarHero.update_user_items' pitch scatter, ported ---
  // Payload shape (JsonHandler._pitch_to_payload):
  //   [time, candidate_pitches[[midi, prob], ...], volume, unvoiced_prob,
  //    live_distance, aligned_distance, is_transition, value]
  // Only the first (most probable) candidate is plotted per frame, matching
  // GuitarHero's `break` after the first candidate_pitches entry. Voicing
  // filter mirrors PitchData.is_voiced_pitch (value != -1 and
  // unvoiced_prob < UNVOICED_THRESHOLD, default 0.9 - see algorithms/Config.py).
  const UNVOICED_THRESHOLD = 0.9;
  const REST_COLOR = "rgb(140, 140, 140)";
  let pitchPoints = $derived.by(() => {
    if (!pitchFrames) return [];
    const points = [];
    for (const frame of pitchFrames) {
      if (!frame) continue;
      const [time, candidates, , unvoicedProb, , alignedDistance, isTransition, value] = frame;
      if (value === -1 || unvoicedProb >= UNVOICED_THRESHOLD) continue;
      if (!candidates || candidates.length === 0) continue;
      const midi = candidates[0][0];
      const color = isTransition
        ? REST_COLOR
        : alignedDistance != null
          ? hsl(alignHue(alignedDistance, pitchTolerance))
          : REST_COLOR;
      points.push({ time, midi, color });
    }
    return points;
  });

  function topPitch(note) {
    return Math.max(...note.midiNum.filter((m) => m >= 0));
  }

  // color for a user note bar: green/red for insertion/deletion-adjacent
  // cases don't apply to user notes directly (insertions are user notes with
  // no score match), substitutions ramp by distance, clean matches are solid
  // green (matches GuitarHero: d <= tolerance is always hue 120, no partial
  // ramp inside the tolerance zone itself).
  function userNoteColor(i) {
    const mistake = mistakeByUserIdx.get(i);
    if (!mistake) return hsl(120); // clean match
    if (mistake.type === "insertion") return INSERTION_COLOR;
    if (mistake.type === "substitution") return hsl(alignHue(mistake.distance, pitchTolerance));
    return hsl(120); // deletion has no user note to color
  }

  let allTimes = $derived([
    ...scoreNotesParsed.flatMap((n) => [n.startTime, n.endTime]),
    ...userNotesParsed.flatMap((n) => [n.startTime, n.endTime]),
    ...pitchPoints.map((p) => p.time),
  ]);
  let allPitches = $derived(
    [
      ...[...scoreNotesParsed, ...userNotesParsed].map(topPitch).filter((p) => Number.isFinite(p)),
      ...pitchPoints.map((p) => p.midi),
    ]
  );

  const WIDTH = 800;
  const HEIGHT = 280;
  const BAR_HEIGHT = 6;
  const PITCH_DOT_RADIUS = 2.5;
  const PITCH_PADDING = 3;

  let minTime = $derived(allTimes.length ? Math.min(...allTimes) : 0);
  let maxTime = $derived(allTimes.length ? Math.max(...allTimes) : 1);
  let minPitch = $derived((allPitches.length ? Math.min(...allPitches) : 60) - PITCH_PADDING);
  let maxPitch = $derived((allPitches.length ? Math.max(...allPitches) : 72) + PITCH_PADDING);

  function xPos(t) {
    const span = maxTime - minTime || 1;
    return ((t - minTime) / span) * WIDTH;
  }
  function yPos(midi) {
    const span = maxPitch - minPitch || 1;
    return HEIGHT - ((midi - minPitch) / span) * HEIGHT;
  }

  // one thin horizontal band per semitone in view, colored by the MIDI
  // rainbow key scheme - the piano-roll background GuitarHero always shows.
  let midiBands = $derived.by(() => {
    const bands = [];
    const lo = Math.floor(minPitch);
    const hi = Math.ceil(maxPitch);
    for (let m = lo; m <= hi; m++) {
      bands.push({ midi: m, y: yPos(m + 0.5), color: midiToRgba(m) });
    }
    return bands;
  });

  const semitoneHeight = $derived(HEIGHT / (maxPitch - minPitch || 1));

  // dashed lines between matched/substituted pairs (both sides present) -
  // mirrors GuitarHero's match_lines (goods + subs, not insertions/deletions).
  let matchLines = $derived.by(() => {
    if (!pairs) return [];
    const lines = [];
    for (const [userIdx, scoreIdx] of pairs) {
      if (userIdx == null || scoreIdx == null) continue;
      const u = userNotesParsed[userIdx];
      const s = scoreNotesParsed[scoreIdx];
      if (!u || !s) continue;
      lines.push({
        x1: xPos((u.startTime + u.endTime) / 2),
        y1: yPos(topPitch(u)),
        x2: xPos((s.startTime + s.endTime) / 2),
        y2: yPos(topPitch(s)),
      });
    }
    return lines;
  });
</script>

<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="overlay" role="img" aria-label="Pitch overlay">
  {#each midiBands as band}
    <rect x="0" y={band.y - semitoneHeight / 2} width={WIDTH} height={semitoneHeight} fill={band.color} />
  {/each}

  {#each matchLines as line}
    <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} class="match-line" />
  {/each}

  {#each scoreNotesParsed as note, i}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - BAR_HEIGHT / 2}
      width={Math.max(1.5, xPos(note.endTime) - xPos(note.startTime))}
      height={BAR_HEIGHT}
      fill={deletedScoreIndices.has(i) ? DELETION_COLOR : SCORE_NOTE_COLOR}
    />
  {/each}

  {#each userNotesParsed as note, i}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - BAR_HEIGHT / 2}
      width={Math.max(1.5, xPos(note.endTime) - xPos(note.startTime))}
      height={BAR_HEIGHT}
      fill={userNoteColor(i)}
      class="user-note"
    />
  {/each}

  {#each pitchPoints as p}
    <circle cx={xPos(p.time)} cy={yPos(p.midi)} r={PITCH_DOT_RADIUS} fill={p.color} />
  {/each}

  {#if currentTime != null && currentTime >= minTime && currentTime <= maxTime}
    <line x1={xPos(currentTime)} y1="0" x2={xPos(currentTime)} y2={HEIGHT} class="playhead" />
  {/if}
</svg>

<p class="legend">
  <span class="legend-item"><span class="swatch" style="background:{hsl(120)}"></span>correct</span>
  <span class="legend-item"><span class="swatch" style="background:{hsl(60)}"></span>off-pitch</span>
  <span class="legend-item"><span class="swatch" style="background:{hsl(0)}"></span>way off</span>
  <span class="legend-item"><span class="swatch" style="background:{INSERTION_COLOR}"></span>extra note</span>
  <span class="legend-item"><span class="swatch" style="background:{DELETION_COLOR}"></span>missed note</span>
</p>

<style>
  .overlay {
    width: 100%;
    height: 280px;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .match-line {
    stroke: rgba(255, 255, 255, 0.55);
    stroke-width: 1.5;
    stroke-dasharray: 4, 3;
  }
  /* GuitarHero.timeline: solid green, above everything else in the plot */
  .playhead {
    stroke: rgb(0, 255, 0);
    stroke-width: 1.5;
  }
  .user-note {
    opacity: 0.95;
  }
  .legend {
    font-size: 0.8rem;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    row-gap: 0.3rem;
    column-gap: 0.7rem;
    margin-top: 0.5rem;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.2rem;
  }
</style>
