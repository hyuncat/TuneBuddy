<script>
  // Right column: MistakeWidget + ToleranceWidget (ui/info/MistakeWidget.py,
  // ui/info/ToleranceWidget.py). NoteOverlay (GuitarHero) now lives in the
  // center column next to ScoreViewer, matching the desktop's Perform tab
  // layout - this component only owns the mistake table + tolerance
  // control, reading shared state from sessionState.svelte.js instead of
  // taking analysisResult/noteData/pitchTolerance as props.
  import { noteName } from "./mistakes.js";
  import { session } from "./sessionState.svelte.js";

  const ICONS = "/icons";

  const TIMING_LABELS = { long: "Too long", short: "Too short", early: "Early", late: "Late" };

  function typeIcon(m) {
    if (m.type === "insertion") return { src: `${ICONS}/plus.svg`, tip: "Insertion (extra note played)" };
    if (m.type === "deletion") return { src: `${ICONS}/minus.svg`, tip: "Deletion (note missed)" };
    const user = m.userNote?.midiNum?.[0] ?? 0;
    const target = m.scoreNote?.midiNum?.[0] ?? 0;
    return user < target
      ? { src: `${ICONS}/flatsign.svg`, tip: "Substitution (played flat)" }
      : { src: `${ICONS}/sharpsign.svg`, tip: "Substitution (played sharp)" };
  }

  function formatTime(seconds) {
    if (seconds < 60) return seconds.toFixed(2);
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${String(secs).padStart(2, "0")}`;
  }

  function mistakeTime(m) {
    return m.scoreNote?.startTime ?? m.userNote?.startTime ?? null;
  }

  function handlePitchToleranceChange(e) {
    session.setPitchTolerance(parseFloat(e.target.value));
  }

  function handleTimingToleranceChange(e) {
    session.setTimingTolerance(parseFloat(e.target.value));
  }
</script>

<div class="results-column">
  {#if session.analysisResult}
    {#if session.realigning}
      <p class="status">Realigning...</p>
    {:else if session.realignError}
      <p class="error">{session.realignError}</p>
    {/if}

    <div class="mistake-widget">
      <div class="mistake-header">
        <span>Mistakes:</span>
        <select bind:value={session.mode}>
          <option value="pitch">Pitch</option>
          <option value="timing">Timing</option>
        </select>
      </div>
      <table class="mistake-table">
        <thead>
          {#if session.mode === "pitch"}
            <tr><th>#</th><th>Time</th><th>Type</th><th>Intended</th><th>Actual</th><th><img src="{ICONS}/pencil.svg" alt="Override" class="header-icon" title="Override the user mistake" /></th></tr>
          {:else}
            <tr><th>#</th><th>Time</th><th>Type</th><th>Note</th><th>Amount</th><th><img src="{ICONS}/pencil.svg" alt="Override" class="header-icon" title="Override the user mistake" /></th></tr>
          {/if}
        </thead>
        <tbody>
          {#if session.visibleMistakes.length === 0}
            <tr><td colspan="6" class="clean">No mistakes at the current tolerance.</td></tr>
          {/if}
          {#each session.visibleMistakes as m, idx}
            {@const isOverridden = session.overridden.has(session.overrideKey(m))}
            {@const isSelected = session.selectedMistakeKey === session.overrideKey(m)}
            <tr
              class:overridden={isOverridden}
              class:selected={isSelected}
              onclick={() => session.selectMistake(m)}
            >
              <td>{idx}</td>
              <td>{mistakeTime(m) != null ? formatTime(mistakeTime(m)) : "—"}</td>
              {#if session.mode === "pitch"}
                {@const icon = typeIcon(m)}
                <td class="icon-cell"><img src={icon.src} alt={m.type} title={icon.tip} class="type-icon" /></td>
                <td>{m.type === "insertion" ? "—" : noteName(m.scoreNote?.midiNum?.[0])}</td>
                <td>{m.type === "deletion" ? "—" : noteName(m.userNote?.midiNum?.[0])}</td>
              {:else}
                <td>{TIMING_LABELS[m.type] ?? m.type}</td>
                <td>{noteName(m.scoreNote?.midiNum?.[0])}</td>
                <td>{m.info}</td>
              {/if}
              <td class="icon-cell">
                <button class="override-btn" onclick={() => session.toggleOverride(m)} title={isOverridden ? "Undo override" : "Override (dismiss this mistake)"}>
                  <img src="{ICONS}/{isOverridden ? 'undo-2' : 'trash-2'}.svg" alt={isOverridden ? "Undo" : "Dismiss"} class="type-icon" />
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  {#if session.noteData}
    {#if session.mode === "timing"}
      <div class="tolerance-widget">
        <img src="{ICONS}/circle-help.svg" alt="" class="help-icon" title="How far off +/- the user's note can vary from the score in timing." />
        <span class="label">Tolerance (s):</span>
        <input
          type="range"
          min="0.02"
          max="1"
          step="0.02"
          value={session.timingTolerance}
          oninput={handleTimingToleranceChange}
          class="tolerance-slider"
        />
        <span class="value-box">{session.timingTolerance.toFixed(2)}</span>
      </div>
    {:else}
      <div class="tolerance-widget">
        <img src="{ICONS}/circle-help.svg" alt="" class="help-icon" title="How close to the intended note (in semitones) the user can play to be counted correct." />
        <span class="label">Tolerance:</span>
        <input
          type="range"
          min="0.5"
          max="5"
          step="0.5"
          value={session.pitchTolerance}
          oninput={handlePitchToleranceChange}
          class="tolerance-slider"
        />
        <span class="value-box">{session.pitchTolerance.toFixed(2)}</span>
      </div>
    {/if}
  {/if}
</div>

<style>
  .results-column {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 8px;
    overflow-y: auto;
  }
  .tolerance-widget {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
  }
  .help-icon {
    width: 14px;
    height: 14px;
    opacity: 0.7;
  }
  .label {
    color: var(--text);
    white-space: nowrap;
  }
  /* QSlider: 4px groove, dark-blue unfilled track, pill-shaped accent
     handle - accent-color can't express the two-tone track, so this is
     hand-styled rather than relying on the browser default. */
  .tolerance-slider {
    flex: 1;
    max-width: 320px;
    -webkit-appearance: none;
    appearance: none;
    height: 4px;
    border-radius: 2px;
    background: var(--slider-track-unfilled);
  }
  .tolerance-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 8px;
    border-radius: 8px;
    background: var(--accent);
    cursor: pointer;
  }
  /* QLineEdit */
  .value-box {
    display: inline-block;
    min-width: 40px;
    padding: 3px 6px;
    text-align: center;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: monospace;
  }
  .status {
    color: var(--text);
  }
  .error {
    color: var(--danger);
  }
  .mistake-widget {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  .mistake-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
  }
  /* QComboBox */
  .mistake-header select {
    background: var(--bg-surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 6px;
  }
  .mistake-header select:focus {
    outline: none;
    border-color: var(--accent);
  }
  /* QTableView: genuinely black background + a distinct gridline color -
     not the general surface gray. */
  .mistake-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: var(--bg-table);
    border: 1px solid var(--border);
  }
  /* QHeaderView::section: bg #3f4042, left-aligned, not dimmed */
  .mistake-table th {
    background: var(--bg-header);
    color: var(--text);
    text-align: center;
    padding: 4px 6px;
    border-bottom: 1px solid var(--table-gridline);
    font-weight: 600;
  }
  .mistake-table tbody tr {
    cursor: pointer;
  }
  .mistake-table tbody tr:hover td {
    background: var(--bg-item-hover);
  }
  .mistake-table td {
    text-align: center;
    padding: 4px 6px;
    border-bottom: 1px solid var(--table-gridline);
  }
  .mistake-table td.clean {
    color: var(--success);
  }
  .icon-cell img {
    width: 18px;
    height: 18px;
    vertical-align: middle;
  }
  .header-icon {
    width: 14px;
    height: 14px;
    vertical-align: middle;
  }
  .override-btn {
    background: none;
    border: none;
    border-radius: 3px;
    padding: 2px;
    cursor: pointer;
    line-height: 0;
  }
  .override-btn:hover {
    background: var(--bg-hover);
  }
  /* MistakeWidget._OVERRIDE_BG / _OVERRIDE_FG, exact */
  tr.overridden td {
    background: rgba(0, 0, 0, 0.45);
    color: var(--text-disabled);
  }
  /* QAbstractItemView::item:selected - matches RecordingTree's selection
     color. Comes after .overridden so a selected+overridden row still
     reads as selected. */
  tr.selected td {
    background: var(--bg-item-selected);
    color: var(--text);
  }
</style>
