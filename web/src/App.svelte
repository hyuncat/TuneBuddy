<script>
  import ScoreViewer from "./ScoreViewer.svelte";
  import UploadForm from "./UploadForm.svelte";
  import ResultsView from "./ResultsView.svelte";

  let scoreViewer;
  let analysisResult = $state(null);
  let noteData = $state(null);
  let pitchTolerance = $state(0.5);

  function handleAnalysisResult(data) {
    analysisResult = data;
  }

  function handleNoteData(data) {
    noteData = data;
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

<div class="toolbar">
  <span class="app-title">Attune</span>
</div>

<main>
  <section class="panel">
    <h2>Analyze a recording</h2>
    <UploadForm onResult={handleAnalysisResult} onNoteData={handleNoteData} {pitchTolerance} />
    {#if noteData}
      <p class="result-summary">
        Score note data: "{noteData.title}", {noteData.instruments.length} instrument(s),
        {Object.values(noteData.note_data).reduce((n, notes) => n + notes.length, 0)} score notes.
      </p>
    {/if}
    <ResultsView {analysisResult} {noteData} bind:pitchTolerance />
  </section>

  <section class="panel">
    <h2>ScoreViewer verification</h2>
    <button class="btn" onclick={loadTestScore}>Load test score (ScoreViewer verification)</button>
    <div class="viewer-frame">
      <ScoreViewer bind:this={scoreViewer} />
    </div>
  </section>
</main>

<style>
  .toolbar {
    background: var(--bg-surface-raised);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 1.5rem;
  }
  .app-title {
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
  }
  main {
    padding: 1.5rem;
    max-width: 900px;
    margin: 0 auto;
  }
  .panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
  }
  .panel h2 {
    margin-top: 0;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
  }
  .viewer-frame {
    margin-top: 1rem;
    height: 400px;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .result-summary {
    margin-top: 0.75rem;
    color: var(--text-secondary);
    font-size: 0.9rem;
  }
  .btn {
    background: var(--bg-surface-raised);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.4rem 0.8rem;
    cursor: pointer;
  }
  .btn:hover {
    border-color: var(--accent);
  }
</style>
