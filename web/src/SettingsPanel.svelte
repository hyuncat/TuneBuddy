<script>
  // Ports ui/info/SettingsWidget.py's row structure exactly: Instrument,
  // Range, Tuning, Transpose, each a label + control(s) + checkmark Apply
  // button. Only Instrument is wired to something real (score_data has
  // multiple channels, and /analyze already accepts active_instrument -
  // see analyze_api.py). Range/Tuning/Transpose map to Config.fmin/fmax/
  // tuning and ScoreData.transpose_offset server-side, but nothing in this
  // app sends them yet, so they're present for structural parity and
  // disabled rather than half-wired.
  import { session } from "./sessionState.svelte.js";

  const ICONS = "/icons";

  let pendingInstrument = $state(null);
  $effect(() => {
    // reset the pending selection to whatever's actually active whenever a
    // new score loads, mirroring SettingsWidget syncing on score/recording
    // change rather than carrying over a stale pending value.
    pendingInstrument = session.activeInstrument;
  });

  function applyInstrument() {
    if (pendingInstrument != null) {
      session.setSelectedInstrument(pendingInstrument);
    }
  }
</script>

<div class="settings-panel">
  <div class="row">
    <label for="instrument-select">Instrument:</label>
    <select
      id="instrument-select"
      bind:value={pendingInstrument}
      disabled={!session.noteData}
    >
      {#if session.noteData}
        {#each session.noteData.instruments as ch}
          <option value={ch}>Channel {ch}</option>
        {/each}
      {/if}
    </select>
    <button class="apply-btn" onclick={applyInstrument} disabled={!session.noteData} title="Apply instrument">
      <img src="{ICONS}/check.svg" alt="Apply" />
    </button>
  </div>

  <div class="row">
    <label for="range-low">Range:</label>
    <input id="range-low" type="text" placeholder="G3" disabled title="Not yet wired to the backend" />
    <span class="sep">—</span>
    <input type="text" placeholder="E7" disabled title="Not yet wired to the backend" />
    <button class="apply-btn" disabled title="Apply frequency range">
      <img src="{ICONS}/check.svg" alt="Apply" />
    </button>
  </div>

  <div class="row">
    <label for="tuning-input">Tuning:</label>
    <input id="tuning-input" type="text" value="440" disabled title="Not yet wired to the backend" />
    <span class="sep">Hz</span>
    <button class="apply-btn" disabled title="Apply tuning">
      <img src="{ICONS}/check.svg" alt="Apply" />
    </button>
  </div>

  <div class="row">
    <img src="{ICONS}/circle-help.svg" alt="" class="help-icon" title="Shift the score's displayed/matched pitch by a fixed interval." />
    <label for="transpose-input">Transpose:</label>
    <input id="transpose-input" type="text" placeholder="C4" disabled title="Not yet wired to the backend" />
    <button class="apply-btn" disabled title="Apply transpose">
      <img src="{ICONS}/check.svg" alt="Apply" />
    </button>
  </div>
</div>

<style>
  .settings-panel {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    font-size: 0.8rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .row label {
    color: var(--text-secondary);
    width: 62px;
    flex-shrink: 0;
  }
  .row input,
  .row select {
    flex: 1;
    min-width: 0;
    background: var(--bg-window);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 4px;
  }
  .row input:disabled,
  .row select:disabled {
    color: var(--text-disabled);
  }
  .sep {
    color: var(--text-secondary);
  }
  .apply-btn {
    background: none;
    border: none;
    padding: 2px;
    cursor: pointer;
    line-height: 0;
    flex-shrink: 0;
  }
  .apply-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .apply-btn img {
    width: 16px;
    height: 16px;
  }
  .help-icon {
    width: 14px;
    height: 14px;
    opacity: 0.7;
    flex-shrink: 0;
  }
</style>
