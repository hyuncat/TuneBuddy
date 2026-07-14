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
  /* self.transport_widget is a plain QWidget (not a QToolBar/QStatusBar), so
     it takes the base QWidget background - same color as the rest of the
     window, not a distinct "raised" panel. The hairline top border is a
     minimal web-only affordance since the real app gets its visual
     separation for free from the splitter/tab edge above this row. */
  .transport-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-window);
    border-top: 1px solid var(--border);
  }
  /* play/record are real QPushButtons (icon-only, fixed 26x26): transparent
     at rest, accent-tinted hover/pressed, bordered in the shared gray. */
  .icon-btn {
    width: 26px;
    height: 26px;
    padding: 4px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .icon-btn:hover:not(:disabled) {
    background: var(--accent-hover-bg);
  }
  .icon-btn:active:not(:disabled) {
    background: var(--accent-pressed-bg);
  }
  .icon-btn img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    opacity: 0.5;
  }
  .time-label {
    min-width: 100px;
    color: var(--text);
    font-family: monospace;
    font-size: 0.85rem;
  }
  /* QSlider: 4px groove, filled portion in accent, unfilled in a dark blue
     (not neutral gray) - accent-color alone can't express the two-tone
     track, so the groove/thumb are hand-styled instead of relying on it. */
  .transport-slider {
    flex: 1;
    -webkit-appearance: none;
    appearance: none;
    height: 4px;
    border-radius: 2px;
    background: var(--slider-track-unfilled);
  }
  .transport-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 8px;
    border-radius: 8px;
    background: var(--accent);
  }
  .transport-slider:disabled {
    opacity: 0.5;
  }
  /* QPushButton: border in the shared gray, accent-colored TEXT (not
     white), transparent at rest, accent-tinted hover/pressed backgrounds. */
  .analyze-btn {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    flex-shrink: 0;
  }
  .analyze-btn:hover:not(:disabled) {
    background: var(--accent-hover-bg);
  }
  .analyze-btn:active:not(:disabled) {
    background: var(--accent-pressed-bg);
  }
  .analyze-btn:disabled {
    color: var(--text-disabled);
    cursor: default;
  }
  .error {
    color: var(--danger);
    font-size: 0.85rem;
    margin: 4px 10px;
  }
</style>
