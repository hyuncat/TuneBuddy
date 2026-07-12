<script>
  import { realign, debounce } from "./realign.js";
  import { classifyPitchMistakes, classifyTimingMistakes, noteName } from "./mistakes.js";
  import NoteOverlay from "./NoteOverlay.svelte";

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const ICONS = "/icons"; // synced from resources/icons via npm run sync-icons

  let {
    analysisResult = null,
    noteData = null,
    pitchTolerance = $bindable(0.5),
  } = $props();

  let timingTolerance = $state(0.25);
  // ui/info/MistakeWidget.py's mode dropdown: one tree, swapped between the
  // pitch and timing mistake lists rather than one combined chronological
  // list.
  let mode = $state("pitch"); // "pitch" | "timing"

  let realignedPairs = $state(null);
  let realignError = $state("");
  let realigning = $state(false);

  let lastAnalysisResult = null;
  $effect(() => {
    if (analysisResult !== lastAnalysisResult) {
      lastAnalysisResult = analysisResult;
      realignedPairs = null;
    }
  });

  const debouncedRealign = debounce(async (tolerance) => {
    if (!analysisResult || !noteData) return;
    const activeInstrument = String(analysisResult.recording.active_instrument);
    const scoreNotes = noteData.note_data[activeInstrument];
    realigning = true;
    try {
      const result = await realign(analysisResult.note_data, scoreNotes, tolerance, API_BASE_URL);
      realignedPairs = result.pairs;
      realignError = "";
    } catch (err) {
      realignError = err instanceof Error ? err.message : String(err);
    } finally {
      realigning = false;
    }
  }, 250);

  function handlePitchToleranceChange(e) {
    pitchTolerance = parseFloat(e.target.value);
    debouncedRealign(pitchTolerance); // no-op pre-analysis; guarded above
  }

  function handleTimingToleranceChange(e) {
    timingTolerance = parseFloat(e.target.value);
    // No /realign: timing-mistake reclassification is a pure client-side
    // threshold check over the existing pairs (see project notes).
  }

  let currentPairs = $derived(realignedPairs ?? analysisResult?.alignment?.pairs ?? null);
  let scoreNotesForActiveInstrument = $derived(
    noteData && analysisResult
      ? noteData.note_data[String(analysisResult.recording.active_instrument)]
      : null
  );

  let pitchMistakes = $derived.by(() => {
    if (!currentPairs || !analysisResult || !scoreNotesForActiveInstrument) return [];
    return classifyPitchMistakes(
      currentPairs,
      analysisResult.note_data,
      scoreNotesForActiveInstrument,
      pitchTolerance
    );
  });

  let timingMistakes = $derived.by(() => {
    if (!currentPairs || !analysisResult || !scoreNotesForActiveInstrument) return [];
    return classifyTimingMistakes(
      currentPairs,
      analysisResult.note_data,
      scoreNotesForActiveInstrument,
      timingTolerance
    );
  });

  let visibleMistakes = $derived(
    (mode === "timing" ? timingMistakes : pitchMistakes).slice().sort((a, b) => {
      const ta = a.scoreNote?.startTime ?? a.userNote?.startTime ?? 0;
      const tb = b.scoreNote?.startTime ?? b.userNote?.startTime ?? 0;
      return ta - tb;
    })
  );

  // Override ("dismiss this mistake"): client-side only, since there's no
  // persistence layer for the web app yet. Keyed by mode+pairIndex+type since
  // a pair can appear in both an onset AND duration timing mistake.
  let overridden = $state(new Set());
  function overrideKey(m) {
    return `${mode}:${m.pairIndex}:${m.type}`;
  }
  function toggleOverride(m) {
    const key = overrideKey(m);
    const next = new Set(overridden);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    overridden = next;
  }

  const TIMING_LABELS = { long: "Too long", short: "Too short", early: "Early", late: "Late" };

  // Type-column icon + tooltip for a pitch mistake - mirrors
  // MistakeWidget._type_icon_and_tip exactly, including the flat/sharp rule.
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
    // mirrors MistakeWidget._format_time
    if (seconds < 60) return seconds.toFixed(2);
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${String(secs).padStart(2, "0")}`;
  }

  function mistakeTime(m) {
    return m.scoreNote?.startTime ?? m.userNote?.startTime ?? null;
  }
</script>

{#if analysisResult}
  {#if realigning}
    <p class="status">Realigning...</p>
  {:else if realignError}
    <p class="error">{realignError}</p>
  {/if}

  {#if scoreNotesForActiveInstrument}
    <NoteOverlay
      scoreNotes={scoreNotesForActiveInstrument}
      userNotes={analysisResult.note_data}
      pairs={currentPairs}
      pitchMistakes={pitchMistakes}
      pitchFrames={analysisResult.pitch_data?.pitches}
      {pitchTolerance}
    />
  {/if}

  <div class="mistake-widget">
    <div class="mistake-header">
      <span>Mistakes:</span>
      <select bind:value={mode}>
        <option value="pitch">Pitch</option>
        <option value="timing">Timing</option>
      </select>
    </div>
    <table class="mistake-table">
      <thead>
        {#if mode === "pitch"}
          <tr><th>#</th><th>Time</th><th>Type</th><th>Intended</th><th>Actual</th><th><img src="{ICONS}/pencil.svg" alt="Override" class="header-icon" title="Override the user mistake" /></th></tr>
        {:else}
          <tr><th>#</th><th>Time</th><th>Type</th><th>Note</th><th>Amount</th><th><img src="{ICONS}/pencil.svg" alt="Override" class="header-icon" title="Override the user mistake" /></th></tr>
        {/if}
      </thead>
      <tbody>
        {#if visibleMistakes.length === 0}
          <tr><td colspan="6" class="clean">No mistakes at the current tolerance.</td></tr>
        {/if}
        {#each visibleMistakes as m, idx}
          {@const isOverridden = overridden.has(overrideKey(m))}
          <tr class:overridden={isOverridden}>
            <td>{idx}</td>
            <td>{mistakeTime(m) != null ? formatTime(mistakeTime(m)) : "—"}</td>
            {#if mode === "pitch"}
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
              <button class="override-btn" onclick={() => toggleOverride(m)} title={isOverridden ? "Undo override" : "Override (dismiss this mistake)"}>
                <img src="{ICONS}/{isOverridden ? 'undo-2' : 'trash-2'}.svg" alt={isOverridden ? "Undo" : "Dismiss"} class="type-icon" />
              </button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}

{#if noteData}
  {#if mode === "timing"}
    <div class="tolerance-widget">
      <img src="{ICONS}/circle-help.svg" alt="" class="help-icon" title="How far off +/- the user's note can vary from the score in timing." />
      <span class="label">Tolerance (s):</span>
      <input
        type="range"
        min="0.02"
        max="1"
        step="0.02"
        value={timingTolerance}
        oninput={handleTimingToleranceChange}
        class="tolerance-slider"
      />
      <span class="value-box">{timingTolerance.toFixed(2)}</span>
    </div>
  {:else}
    <div class="tolerance-widget">
      <img src="{ICONS}/circle-help.svg" alt="" class="help-icon" title="How close to the intended note (in semitones) the user can play to be counted correct." />
      <span class="label">Tolerance:</span>
      <input
        type="range"
        min="0.05"
        max="5"
        step="0.05"
        value={pitchTolerance}
        oninput={handlePitchToleranceChange}
        class="tolerance-slider"
      />
      <span class="value-box">{pitchTolerance.toFixed(2)}</span>
    </div>
  {/if}
{/if}

<style>
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
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .tolerance-slider {
    flex: 1;
    max-width: 320px;
    accent-color: var(--accent);
  }
  .value-box {
    display: inline-block;
    min-width: 40px;
    padding: 2px 6px;
    text-align: center;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: monospace;
  }
  .status {
    color: var(--text-secondary);
  }
  .error {
    color: var(--danger);
  }
  .mistake-widget {
    margin-top: 1rem;
    max-width: 520px;
  }
  .mistake-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
  }
  .mistake-header select {
    background: var(--bg-surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 6px;
  }
  .mistake-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: var(--bg-surface);
    border: 1px solid var(--border);
  }
  .mistake-table th {
    background: var(--bg-surface-raised);
    color: var(--text-secondary);
    text-align: center;
    padding: 4px 6px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  .mistake-table td {
    text-align: center;
    padding: 4px 6px;
    border-bottom: 1px solid var(--border);
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
    padding: 2px;
    cursor: pointer;
    line-height: 0;
  }
  /* mirrors MistakeWidget._OVERRIDE_BG/_OVERRIDE_FG: a translucent dark tint
     over the row plus dimmed text, so a dismissed mistake reads as "set
     aside" rather than disappearing. */
  tr.overridden td {
    background: rgba(0, 0, 0, 0.45);
    color: var(--text-disabled);
  }
</style>
