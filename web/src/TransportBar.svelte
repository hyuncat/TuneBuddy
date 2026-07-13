<script>
  // Ports app.py's init_slider_layout transport row exactly: play button,
  // record button, time label, slider, Analyze button - a persistent
  // full-width row below the main splitter, not inside a tab.
  //
  // Play/record/slider are stubs: they drive live MIDI/audio playback
  // (task #3, not built yet), so there's nothing to wire them to. Analyze
  // is real - moved here from the old standalone upload panel, since on the
  // desktop app it's the last control in this same row, not a separate form.
  import { session } from "./sessionState.svelte.js";

  const ICONS = "/icons";

  let canAnalyze = $derived(
    !!session.scoreFile && !!session.audioFile && session.analyzeStatus !== "loading"
  );
</script>

<div class="transport-bar">
  <button class="icon-btn" disabled title="Playback isn't wired up yet">
    <img src="{ICONS}/play.png" alt="Play" />
  </button>
  <button class="icon-btn" disabled title="Recording isn't wired up yet">
    <img src="{ICONS}/record.png" alt="Record" />
  </button>
  <span class="time-label">00:00.0 / 00:00.0</span>
  <input class="transport-slider" type="range" min="0" max="100" value="0" disabled />
  <button class="analyze-btn" onclick={() => session.runAnalyze()} disabled={!canAnalyze}>
    {session.analyzeStatus === "loading" ? "Analyzing..." : "Analyze"}
  </button>
</div>

{#if session.analyzeStatus === "error"}
  <p class="error">{session.analyzeError}</p>
{/if}

<style>
  .transport-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-surface-raised);
    border-top: 1px solid var(--border);
  }
  .icon-btn {
    width: 26px;
    height: 26px;
    padding: 4px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: default;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .icon-btn img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    opacity: 0.5;
  }
  .time-label {
    min-width: 100px;
    color: var(--text-secondary);
    font-family: monospace;
    font-size: 0.85rem;
  }
  .transport-slider {
    flex: 1;
    accent-color: var(--accent);
  }
  .analyze-btn {
    background: var(--bg-surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 5px 14px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .analyze-btn:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .analyze-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .error {
    color: var(--danger);
    font-size: 0.85rem;
    margin: 4px 10px;
  }
</style>
