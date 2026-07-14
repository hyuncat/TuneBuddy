<script>
  // Ports ui/info/Toolbar.py's structure and order exactly: Upload (menu) ->
  // Settings -> Save -> Clip (menu) -> separator -> Playback/instrument
  // select (menu) -> spacer -> Tempo spinbox -> Metronome label + switch.
  // Text-only buttons, no icons - the desktop toolbar itself uses none
  // (confirmed: no QIcon/.svg/.png references in Toolbar.py).
  //
  // Settings/Save/Clip/Playback/Tempo/Metronome have no backend behind them
  // yet (no persistence layer, no clip selection, no MIDI playback - task
  // #3) so they're rendered disabled with a tooltip explaining why, rather
  // than either hiding them (breaking the structural match) or faking
  // functionality that doesn't exist.
  import { session } from "./sessionState.svelte.js";

  const SCORE_ACCEPT = ".mid,.midi,.mxl,.musicxml,.xml,.mei";
  const AUDIO_ACCEPT = ".wav,.wave,.flac,.ogg,.aif,.aiff,.m4a";

  let uploadMenuOpen = $state(false);
  let clipMenuOpen = $state(false);
  let playbackMenuOpen = $state(false);
  let scoreInputEl;
  let audioInputEl;

  function closeMenus() {
    uploadMenuOpen = false;
    clipMenuOpen = false;
    playbackMenuOpen = false;
  }

  function handleScoreFile(e) {
    const file = e.target.files?.[0] ?? null;
    if (file) session.pickScore(file);
    e.target.value = "";
    closeMenus();
  }

  function handleAudioFile(e) {
    const file = e.target.files?.[0] ?? null;
    if (file) session.pickAudio(file);
    e.target.value = "";
    closeMenus();
  }

  let tempoDisplay = $derived(session.noteData?.bpm ?? 120);
</script>

<div class="toolbar">
  <div class="tb-item">
    <button class="tb-button" onclick={() => (uploadMenuOpen = !uploadMenuOpen)}>
      Upload
    </button>
    {#if uploadMenuOpen}
      <div class="tb-backdrop" onclick={closeMenus}></div>
      <div class="tb-menu">
        <button onclick={() => scoreInputEl.click()}>Score</button>
        <button onclick={() => audioInputEl.click()}>Recording</button>
        <button disabled title="Folder-based recording libraries aren't supported in the web version">Folder</button>
      </div>
    {/if}
  </div>

  <button class="tb-button" disabled title="Not available in the web version">Settings</button>
  <button class="tb-button" disabled title="No save/persistence in the web version">Save</button>

  <div class="tb-item">
    <button class="tb-button" onclick={() => (clipMenuOpen = !clipMenuOpen)} disabled title="Clip selection isn't implemented yet">
      Clip
    </button>
  </div>

  <div class="tb-separator"></div>

  <div class="tb-item">
    <button
      class="tb-button"
      onclick={() => (playbackMenuOpen = !playbackMenuOpen)}
      disabled={!session.noteData}
      title="Playback selection requires MIDI playback, which isn't wired up yet"
    >
      Playback
    </button>
  </div>

  <div class="tb-spacer"></div>

  <input
    class="tempo-spinbox"
    type="number"
    min="20"
    max="400"
    value={tempoDisplay}
    disabled
    title="Tempo control requires playback, which isn't wired up yet"
    readonly
  />
  <span class="tb-label">BPM</span>
  <span class="tb-label">Metronome</span>
  <label class="toggle-switch" title="Metronome requires playback, which isn't wired up yet">
    <input type="checkbox" checked disabled />
    <span class="toggle-track"></span>
  </label>
</div>

<input bind:this={scoreInputEl} type="file" accept={SCORE_ACCEPT} onchange={handleScoreFile} hidden />
<input bind:this={audioInputEl} type="file" accept={AUDIO_ACCEPT} onchange={handleAudioFile} hidden />

{#if session.noteDataError}
  <p class="upload-error">Couldn't load score note data: {session.noteDataError}</p>
{/if}

<style>
  /* QToolBar: bg #333333 (lighter than the window), 1px padding, bold text,
     2px spacing between items - all real values, not approximated. */
  .toolbar {
    display: flex;
    align-items: center;
    gap: 2px;
    background: var(--bg-toolbar);
    padding: 1px;
    font-size: 0.85rem;
    font-weight: 700;
  }
  .tb-item {
    position: relative;
  }
  /* QToolBar > QToolButton: transparent, 3px padding, 4px radius */
  .tb-button {
    background: none;
    border: none;
    color: var(--text);
    border-radius: 4px;
    padding: 4px 10px;
    cursor: pointer;
  }
  .tb-button:hover:not(:disabled) {
    background: var(--bg-hover);
  }
  .tb-button:active:not(:disabled) {
    background: var(--bg-pressed);
  }
  .tb-button:disabled {
    color: var(--text-disabled);
    cursor: default;
  }
  .tb-backdrop {
    position: fixed;
    inset: 0;
    z-index: 10;
  }
  /* QMenu: bg #292a2d, 8px vertical padding, 1px border in the shared
     border-gray, items get 4px/28px padding and highlight to the same gray
     on hover/selection. */
  .tb-menu {
    position: absolute;
    top: 100%;
    left: 0;
    z-index: 11;
    display: flex;
    flex-direction: column;
    background: var(--bg-menu);
    border: 1px solid var(--border);
    border-radius: 4px;
    min-width: 140px;
    padding: 8px 0;
    font-weight: 400;
  }
  .tb-menu button {
    background: none;
    border: none;
    color: var(--text);
    text-align: left;
    padding: 4px 28px 4px 12px;
    cursor: pointer;
  }
  .tb-menu button:hover:not(:disabled) {
    background: var(--border);
  }
  .tb-menu button:disabled {
    color: var(--text-disabled);
    cursor: default;
  }
  .tb-separator {
    width: 2px;
    align-self: stretch;
    margin: 6px;
    background: var(--border);
  }
  .tb-spacer {
    flex: 1;
  }
  /* QLineEdit/QAbstractSpinBox: bg #3f4042 (the shared surface/border gray),
     4px radius, focus ring in accent. */
  .tempo-spinbox {
    width: 52px;
    background: var(--bg-surface);
    color: var(--text-disabled);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 4px;
    text-align: right;
    font-weight: 400;
  }
  .tb-label {
    color: var(--text);
    white-space: nowrap;
    padding: 0 4px;
  }
  /* ui/info/ToggleSwitch.py, ported exactly: a QCheckBox custom-painted at a
     fixed 46x26, 2px margin to the track, off-track #777777 (not the theme's
     neutral gray), a solid white knob at any state, on-track "#3daee9". */
  .toggle-switch input {
    display: none;
  }
  .toggle-switch {
    display: inline-block;
    position: relative;
    width: 46px;
    height: 26px;
    vertical-align: middle;
    opacity: 0.7;
  }
  .toggle-track {
    position: absolute;
    inset: 2px;
    border-radius: 11px;
    background: #777777;
  }
  .toggle-track::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
  }
  .toggle-switch input:checked + .toggle-track {
    background: var(--toggle-on);
  }
  .toggle-switch input:checked + .toggle-track::after {
    left: 22px;
  }
  .upload-error {
    color: var(--danger);
    font-size: 0.85rem;
    margin: 4px 8px;
  }
</style>
