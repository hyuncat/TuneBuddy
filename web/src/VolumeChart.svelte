<script>
  // Ports ui/note/VolumeWidget.py: the cursor-note's loudness curve in dBFS,
  // with the pitch contour underlaid for context, windowed to the note's own
  // [startTime, endTime] span. NotePanel.svelte owns picking which note/tab
  // is active; this only draws.
  import { volumeCurveDb, pitchContour, contourToBand, paddedRange } from "./noteCurve.js";

  let { note, pitchFrames, volumeRange, currentTime } = $props();

  const WIDTH = 380;
  const HEIGHT = 140;
  const VOL_LIVE_FLOOR_DB = -42.0; // ui.Colors.VOL_LIVE_FLOOR_DB - review fallback only
  const Y_PADDING = 0.15;

  // set_default_y_range, ported: take's own [quietest, loudest] dBFS padded
  // 15% each side; falls back to the live window only when the whole take
  // has no measurable volume at all (this app is always "review" mode).
  let yDomain = $derived.by(() => {
    let [y0, y1] = volumeRange ?? [null, null];
    if (y0 == null || y1 == null) [y0, y1] = [VOL_LIVE_FLOOR_DB, 0];
    return paddedRange(y0, y1, Y_PADDING);
  });

  let window = $derived(note ? [note.startTime, note.endTime] : null);

  let contourPoints = $derived.by(() => {
    if (!window) return [];
    return contourToBand(pitchContour(pitchFrames, window[0], window[1]), yDomain[0], yDomain[1]);
  });

  let curvePoints = $derived.by(() => {
    if (!window) return [];
    return volumeCurveDb(pitchFrames, window[0], window[1], yDomain[0]);
  });

  function xPos(t) {
    if (!window) return 0;
    const span = window[1] - window[0] || 1;
    return ((t - window[0]) / span) * WIDTH;
  }
  function yPos(v) {
    const [y0, y1] = yDomain;
    const span = y1 - y0 || 1;
    return HEIGHT - ((v - y0) / span) * HEIGHT;
  }

  // Builds an SVG path, breaking into a new subpath (M) after every null -
  // mirrors PlotDataItem(connect="finite")'s gap behavior.
  function toPath(points, xKey, yKey) {
    let d = "";
    let started = false;
    for (const p of points) {
      const v = p[yKey];
      if (v == null || !Number.isFinite(v)) {
        started = false;
        continue;
      }
      const cmd = started ? "L" : "M";
      d += `${cmd}${xPos(p[xKey]).toFixed(1)},${yPos(v).toFixed(1)} `;
      started = true;
    }
    return d.trim();
  }

  let contourPath = $derived(toPath(contourPoints, "time", "y"));
  let curvePath = $derived(toPath(curvePoints, "time", "db"));
  let timelineX = $derived(window ? xPos(Math.min(Math.max(currentTime, window[0]), window[1])) : null);
</script>

<div class="chart-wrap">
  {#if !note}
    <div class="chart-blank">No note under the cursor</div>
  {:else}
    <svg viewBox="0 0 {WIDTH} {HEIGHT}" class="chart" role="img" aria-label="Volume over the current note">
      {#if contourPath}
        <path d={contourPath} class="contour-line" />
      {/if}
      {#if curvePath}
        <path d={curvePath} class="volume-line" />
      {/if}
      {#if timelineX != null}
        <line x1={timelineX} y1="0" x2={timelineX} y2={HEIGHT} class="timeline" />
      {/if}
      <text x="4" y="12" class="axis-label">dBFS</text>
      <text x={WIDTH - 4} y={HEIGHT - 4} class="axis-label" text-anchor="end">Time (s)</text>
    </svg>
  {/if}
</div>

<style>
  .chart-wrap {
    width: 100%;
    height: 140px;
  }
  .chart {
    width: 100%;
    height: 100%;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .chart-blank {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgb(20, 20, 25);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 0.8rem;
  }
  /* ui.Colors.NOTE_CONTOUR_RGB['volume'] = (95,100,110) */
  .contour-line {
    fill: none;
    stroke: rgb(95, 100, 110);
    stroke-width: 1.5;
  }
  /* ui.Colors.NOTE_VOLUME_RGB = (94,201,98) */
  .volume-line {
    fill: none;
    stroke: rgb(94, 201, 98);
    stroke-width: 2.5;
  }
  /* ui.Colors.NOTE_TIMELINE_RGB = (0,255,0), mirrors GuitarHero's own cursor */
  .timeline {
    stroke: rgb(0, 255, 0);
    stroke-width: 1.5;
  }
  .axis-label {
    fill: rgb(230, 230, 235);
    font-size: 9px;
  }
</style>
