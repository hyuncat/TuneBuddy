<script>
  import ScoreViewer from "./ScoreViewer.svelte";
  import UploadForm from "./UploadForm.svelte";
  import { realign, debounce } from "./realign.js";

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  let scoreViewer;
  let analysisResult = $state(null);
  let noteData = $state(null);

  let pitchTolerance = $state(0.5);
  let realignedPairs = $state(null);
  let realignError = $state("");
  let realigning = $state(false);

  function handleAnalysisResult(data) {
    analysisResult = data;
    realignedPairs = null; // stale against the new recording until re-realigned
  }

  function handleNoteData(data) {
    noteData = data;
  }

  // 5A verification only: proves /realign genuinely re-aligns as pitch
  // tolerance changes, not just that the request succeeds. The actual
  // mistake list/tolerance UI is a later step - this just wires the data
  // layer and shows it's live.
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

  function handleToleranceChange(e) {
    pitchTolerance = parseFloat(e.target.value);
    debouncedRealign(pitchTolerance);
  }

  function unmatchedCount(pairs) {
    return pairs.filter(([u, s]) => u === null || s === null).length;
  }

  // Minimal valid MusicXML (a single whole note, C4) - just enough to prove
  // ScoreViewer's iframe wiring end-to-end. Not from the real pipeline (that's
  // ScoreData.to_musicxml_bytes() server-side, not built yet) - this is a
  // throwaway verification fixture, not app content.
  const TEST_MUSICXML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>`;

  function loadTestScore() {
    const bytes = new TextEncoder().encode(TEST_MUSICXML);
    scoreViewer.loadScore(bytes);
  }
</script>

<main>
  <h1>Attune</h1>

  <section>
    <h2>Analyze a recording</h2>
    <UploadForm onResult={handleAnalysisResult} onNoteData={handleNoteData} />
    {#if noteData}
      <!-- placeholder pending the real results view (task #5) -->
      <p class="result-summary">
        Score note data: "{noteData.title}", {noteData.instruments.length} instrument(s),
        {Object.values(noteData.note_data).reduce((n, notes) => n + notes.length, 0)} score notes.
      </p>
    {/if}
    {#if analysisResult}
      <!-- mistake classification now happens client-side (task #5); this just
           proves the raw alignment pairs came back, not pre-filtered mistakes -->
      <p class="result-summary">
        Got {analysisResult.pitch_data.pitches.length} pitch frames,
        {analysisResult.note_data.length} user notes,
        {analysisResult.alignment.pairs.length} aligned pairs
        ({unmatchedCount(analysisResult.alignment.pairs)} unmatched at the
        server's default tolerance).
      </p>
      <label class="tolerance-control">
        Pitch tolerance (semitones): {pitchTolerance.toFixed(2)}
        <input
          type="range"
          min="0.05"
          max="5"
          step="0.05"
          value={pitchTolerance}
          oninput={handleToleranceChange}
        />
      </label>
      {#if realigning}
        <p class="result-summary">Realigning...</p>
      {:else if realignError}
        <p class="error">{realignError}</p>
      {:else if realignedPairs}
        <p class="result-summary">
          Realigned: {realignedPairs.length} pairs,
          {unmatchedCount(realignedPairs)} unmatched at tolerance {pitchTolerance.toFixed(2)}.
        </p>
      {/if}
    {/if}
  </section>

  <section>
    <h2>ScoreViewer verification</h2>
    <button onclick={loadTestScore}>Load test score (ScoreViewer verification)</button>
    <div class="viewer-frame">
      <ScoreViewer bind:this={scoreViewer} />
    </div>
  </section>
</main>

<style>
  main {
    font-family: system-ui, sans-serif;
    padding: 2rem;
  }
  .viewer-frame {
    margin-top: 1rem;
    height: 400px;
    border: 1px solid #ccc;
  }
  section {
    margin-bottom: 2rem;
  }
  .result-summary {
    margin-top: 0.75rem;
    font-family: system-ui, sans-serif;
  }
  .tolerance-control {
    display: block;
    margin-top: 0.75rem;
    font-family: system-ui, sans-serif;
    font-size: 0.9rem;
  }
  .tolerance-control input {
    display: block;
    width: 100%;
    max-width: 400px;
  }
  .error {
    color: #c0392b;
  }
</style>
