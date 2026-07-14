<script>
  // Ports app.py's QMainWindow layout: Toolbar (top) -> horizontal splitter
  // [left: RecordingTree + SettingsWidget | center: ScoreViewer + GuitarHero
  // (Perform tab's vertical splitter) | right: MistakeWidget + ToleranceWidget]
  // -> transport row -> StatusBar. Practice tab (live pitch-matching) is out
  // of scope (upload-only, no live recording), so only Perform's layout is
  // ported - there's nothing to put in a second tab.
  import ScoreViewer from "./ScoreViewer.svelte";
  import NoteOverlay from "./NoteOverlay.svelte";
  import ResultsView from "./ResultsView.svelte";
  import Toolbar from "./Toolbar.svelte";
  import RecordingTree from "./RecordingTree.svelte";
  import SettingsPanel from "./SettingsPanel.svelte";
  import TransportBar from "./TransportBar.svelte";
  import StatusBar from "./StatusBar.svelte";
  import { session } from "./sessionState.svelte.js";

  let scoreViewer;

  function base64ToBytes(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // Loads the real uploaded score into Verovio the moment /notedata returns
  // it (musicxml_b64 - see analyze_api.py), independent of Analyze ever
  // running, mirroring the desktop app's score-independent score loading.
  let lastLoadedMusicXml = null;
  $effect(() => {
    const b64 = session.noteData?.musicxml_b64;
    if (b64 && b64 !== lastLoadedMusicXml && scoreViewer?.isReady()) {
      lastLoadedMusicXml = b64;
      scoreViewer.loadScore(base64ToBytes(b64));
    }
  });
</script>

<div class="app-shell">
  <Toolbar />

  <div class="main-splitter">
    <aside class="left-column">
      <RecordingTree />
      <SettingsPanel />
    </aside>

    <section class="center-column">
      <div class="score-pane">
        <ScoreViewer bind:this={scoreViewer} />
      </div>
      <div class="overlay-pane">
        {#if session.analysisResult && session.scoreNotesActive}
          <NoteOverlay
            scoreNotes={session.scoreNotesActive}
            userNotes={session.analysisResult.note_data}
            pairs={session.currentPairs}
            pitchMistakes={session.pitchMistakes}
            pitchFrames={session.analysisResult.pitch_data?.pitches}
            pitchTolerance={session.pitchTolerance}
          />
        {:else}
          <div class="overlay-placeholder">
            Upload a score and a recording, then click Analyze to see the
            pitch overlay here.
          </div>
        {/if}
      </div>
    </section>

    <aside class="right-column">
      <ResultsView />
    </aside>
  </div>

  <TransportBar />
  <StatusBar />
</div>

<style>
  .app-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  .main-splitter {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  /* QMainWindow::separator is 4px, in the shared border-gray (a static
     approximation here - no drag interaction, just the real handle width
     and color instead of a 1px pane border). Panes themselves are plain
     QWidgets, so they take the base window background, not the
     input-family surface gray. */
  .left-column {
    width: 240px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border-right: 4px solid var(--border);
    background: var(--bg-window);
    min-width: 180px;
    max-width: 320px;
  }
  .center-column {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .score-pane {
    flex: 1;
    min-height: 200px;
    border-bottom: 4px solid var(--border);
  }
  .overlay-pane {
    flex: 1;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    padding: 8px;
    overflow: auto;
  }
  .overlay-placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
    background: var(--overlay-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .right-column {
    width: 296px;
    flex-shrink: 0;
    border-left: 4px solid var(--border);
    background: var(--bg-window);
    min-width: 240px;
    max-width: 420px;
  }
</style>
