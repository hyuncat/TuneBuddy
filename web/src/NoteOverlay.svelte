<script>
  // Static note-level piano-roll: score notes as reference bars, user notes
  // colored by mistake classification and positioned at their OWN actual
  // pitch - a wrong note is visually offset from its target rather than
  // needing a separate marker, and a correct note visually stacks on top of
  // it. Not a live/animated view: there's no MIDI playback yet (task #3) to
  // drive a moving cursor, and this renders a finished result once, so plain
  // reactive SVG is simpler here than a Canvas redraw loop would be.
  import { noteFromArray, noteName } from "./mistakes.js";

  let { scoreNotes, userNotes, pitchMistakes } = $props();

  const MISTAKE_COLOR = {
    substitution: "#e67e22", // orange - paired but wrong pitch
    insertion: "#c0392b", // red - extra note, no score match
  };
  const MATCH_COLOR = "#27ae60"; // green
  const DELETION_STROKE = "#c0392b";

  let mistakeByUserIdx = $derived(
    new Map(pitchMistakes.filter((m) => m.userIdx != null).map((m) => [m.userIdx, m]))
  );
  let deletedScoreIndices = $derived(
    new Set(pitchMistakes.filter((m) => m.type === "deletion").map((m) => m.scoreIdx))
  );

  let scoreNotesParsed = $derived(scoreNotes.map(noteFromArray));
  let userNotesParsed = $derived(userNotes.map(noteFromArray));

  // representative pitch for a (possibly chord) note: highest member
  function topPitch(note) {
    return Math.max(...note.midiNum.filter((m) => m >= 0));
  }

  let allTimes = $derived([
    ...scoreNotesParsed.flatMap((n) => [n.startTime, n.endTime]),
    ...userNotesParsed.flatMap((n) => [n.startTime, n.endTime]),
  ]);
  let allPitches = $derived(
    [...scoreNotesParsed, ...userNotesParsed]
      .map(topPitch)
      .filter((p) => Number.isFinite(p))
  );

  const WIDTH = 800;
  const HEIGHT = 260;
  const BAR_HEIGHT = 6;
  const PITCH_PADDING = 3; // semitones of headroom above/below the note range

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

  // A few evenly-spaced pitch gridlines with note-name labels, for orientation.
  let pitchGridlines = $derived.by(() => {
    const lines = [];
    const step = Math.max(1, Math.round((maxPitch - minPitch) / 6));
    for (let p = Math.ceil(minPitch / step) * step; p <= maxPitch; p += step) {
      lines.push(p);
    }
    return lines;
  });
</script>

<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="overlay" role="img" aria-label="Pitch overlay">
  {#each pitchGridlines as midi}
    <line x1="0" x2={WIDTH} y1={yPos(midi)} y2={yPos(midi)} class="gridline" />
    <text x="2" y={yPos(midi) - 2} class="gridline-label">{noteName(midi)}</text>
  {/each}

  {#each scoreNotesParsed as note, i}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - BAR_HEIGHT / 2}
      width={Math.max(1.5, xPos(note.endTime) - xPos(note.startTime))}
      height={BAR_HEIGHT}
      class="score-note"
      class:deleted={deletedScoreIndices.has(i)}
    />
  {/each}

  {#each userNotesParsed as note, i}
    {@const mistake = mistakeByUserIdx.get(i)}
    <rect
      x={xPos(note.startTime)}
      y={yPos(topPitch(note)) - BAR_HEIGHT / 2}
      width={Math.max(1.5, xPos(note.endTime) - xPos(note.startTime))}
      height={BAR_HEIGHT}
      fill={mistake ? (MISTAKE_COLOR[mistake.type] ?? MATCH_COLOR) : MATCH_COLOR}
      class="user-note"
    />
  {/each}
</svg>

<p class="legend">
  <span class="swatch" style="background:{MATCH_COLOR}"></span> correct
  <span class="swatch" style="background:{MISTAKE_COLOR.substitution}"></span> wrong pitch
  <span class="swatch" style="background:{MISTAKE_COLOR.insertion}"></span> extra note
  <span class="swatch outline"></span> missed note
</p>

<style>
  .overlay {
    width: 100%;
    height: 260px;
    background: #fafafa;
    border: 1px solid #ddd;
  }
  .gridline {
    stroke: #e0e0e0;
    stroke-width: 1;
  }
  .gridline-label {
    font-size: 8px;
    fill: #999;
    font-family: system-ui, sans-serif;
  }
  .score-note {
    fill: #bdc3c7;
    opacity: 0.5;
  }
  .score-note.deleted {
    fill: none;
    stroke: #c0392b;
    stroke-dasharray: 3, 2;
    opacity: 0.9;
  }
  .user-note {
    opacity: 0.9;
  }
  .legend {
    font-size: 0.8rem;
    font-family: system-ui, sans-serif;
    color: #555;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.5rem;
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.2rem;
  }
  .swatch.outline {
    background: none;
    border: 1px dashed #c0392b;
  }
</style>
