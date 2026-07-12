<script>
  import { realign, debounce } from "./realign.js";
  import { classifyPitchMistakes, classifyTimingMistakes, noteName } from "./mistakes.js";
  import NoteOverlay from "./NoteOverlay.svelte";

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // analysisResult/noteData: owned by the parent (App.svelte), which needs
  // noteData for the score viewer too. pitchTolerance is $bindable since
  // UploadForm needs its current value at Analyze-time (see project notes on
  // why /analyze accepts an optional tolerance); timingTolerance has no such
  // upstream consumer so it's local-only state below.
  let {
    analysisResult = null,
    noteData = null,
    pitchTolerance = $bindable(0.5),
  } = $props();

  let timingTolerance = $state(0.25);

  let realignedPairs = $state(null);
  let realignError = $state("");
  let realigning = $state(false);

  // Stale against a new recording until re-realigned - reset whenever a
  // fresh analysisResult comes in.
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
    // No /realign here: timing-mistake reclassification is a pure
    // client-side threshold check over the existing pairs, no re-alignment
    // involved (unlike pitch tolerance, which is baked into the alignment's
    // own DP cost matrix - see project notes).
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

  // Combined, time-sorted list for display. A pair can appear once for pitch
  // and up to twice for timing (onset + duration) - each is its own row.
  let allMistakes = $derived(
    [...pitchMistakes, ...timingMistakes].sort((a, b) => {
      const ta = a.scoreNote?.startTime ?? a.userNote?.startTime ?? 0;
      const tb = b.scoreNote?.startTime ?? b.userNote?.startTime ?? 0;
      return ta - tb;
    })
  );

  const MISTAKE_LABEL = {
    deletion: "Missing note",
    insertion: "Extra note",
    substitution: "Wrong pitch",
    early: "Early",
    late: "Late",
    short: "Too short",
    long: "Too long",
  };

  function mistakeTime(m) {
    return m.scoreNote?.startTime ?? m.userNote?.startTime ?? null;
  }

  function mistakeDetail(m) {
    if (m.type === "substitution") {
      return `played ${noteName(m.userNote.midiNum[0])}, expected ${noteName(m.scoreNote.midiNum[0])}`;
    }
    if (m.type === "deletion") {
      return `expected ${noteName(m.scoreNote.midiNum[0])}`;
    }
    if (m.type === "insertion") {
      return `played ${noteName(m.userNote.midiNum[0])}`;
    }
    return m.info ?? "";
  }
</script>

{#if noteData}
  <div class="tolerance-controls">
    <label class="tolerance-control">
      Pitch tolerance (semitones): {pitchTolerance.toFixed(2)}
      <input
        type="range"
        min="0.05"
        max="5"
        step="0.05"
        value={pitchTolerance}
        oninput={handlePitchToleranceChange}
      />
    </label>
    <label class="tolerance-control">
      Timing tolerance (seconds): {timingTolerance.toFixed(2)}
      <input
        type="range"
        min="0.02"
        max="1"
        step="0.02"
        value={timingTolerance}
        oninput={handleTimingToleranceChange}
      />
    </label>
  </div>
{/if}

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
      pitchMistakes={pitchMistakes}
    />
  {/if}

  <ul class="mistake-list">
    {#if allMistakes.length === 0}
      <li class="mistake-item clean">No mistakes at the current tolerance.</li>
    {/if}
    {#each allMistakes as m}
      <li class="mistake-item {m.type}">
        <span class="mistake-type">{MISTAKE_LABEL[m.type] ?? m.type}</span>
        <span class="mistake-time">{mistakeTime(m)?.toFixed(2) ?? "?"}s</span>
        <span class="mistake-detail">{mistakeDetail(m)}</span>
      </li>
    {/each}
  </ul>
{/if}

<style>
  .tolerance-controls {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .tolerance-control {
    display: block;
    font-family: system-ui, sans-serif;
    font-size: 0.9rem;
  }
  .tolerance-control input {
    display: block;
    width: 100%;
    max-width: 400px;
  }
  .status {
    font-family: system-ui, sans-serif;
    color: #555;
  }
  .error {
    color: #c0392b;
    font-family: system-ui, sans-serif;
  }
  .mistake-list {
    list-style: none;
    margin: 0.75rem 0 0;
    padding: 0;
    max-width: 500px;
    font-family: system-ui, sans-serif;
    font-size: 0.9rem;
  }
  .mistake-item {
    display: flex;
    gap: 0.6rem;
    padding: 0.35rem 0.5rem;
    border-bottom: 1px solid #eee;
  }
  .mistake-item.clean {
    color: #27ae60;
  }
  .mistake-type {
    font-weight: 600;
    min-width: 100px;
  }
  .mistake-time {
    color: #888;
    min-width: 45px;
  }
  .mistake-item.substitution .mistake-type,
  .mistake-item.insertion .mistake-type {
    color: #e67e22;
  }
  .mistake-item.deletion .mistake-type {
    color: #c0392b;
  }
  .mistake-item.early .mistake-type,
  .mistake-item.late .mistake-type,
  .mistake-item.short .mistake-type,
  .mistake-item.long .mistake-type {
    color: #8e44ad;
  }
</style>
