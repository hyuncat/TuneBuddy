<script>
  import ScoreViewer from "./ScoreViewer.svelte";

  let scoreViewer;

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
  <button onclick={loadTestScore}>Load test score (ScoreViewer verification)</button>
  <div class="viewer-frame">
    <ScoreViewer bind:this={scoreViewer} />
  </div>
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
</style>
