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
  import NotePanel from "./NotePanel.svelte";
  import Toolbar from "./Toolbar.svelte";
  import RecordingTree from "./RecordingTree.svelte";
  import SettingsPanel from "./SettingsPanel.svelte";
  import TransportBar from "./TransportBar.svelte";
  import StatusBar from "./StatusBar.svelte";
  import { session } from "./sessionState.svelte.js";
  import { playback } from "./playback.svelte.js";
  import { buildAnnotations } from "./annotations.js";
  import { ScoreTimeMap } from "./scoreTimeMap.js";

  let scoreViewer;
  // Bumped by ScoreViewer's onReady callback the instant its iframe finishes
  // loading - read (not just scoreViewer?.ready) inside effects below so
  // they have a guaranteed local $state dependency to re-run on, rather
  // than relying on reactively noticing a property read through the
  // cross-component scoreViewer reference, which proved unreliable.
  let scoreReadyTick = $state(0);

  // Owned here (not a shared singleton, not inside ScoreViewer.svelte) -
  // mirrors perform.py owning _time_map rather than ScoreViewer.py: the
  // widget stays a dumb Verovio wrapper with no score-policy knowledge
  // (tempo/transpose), and App.svelte plays the same "host tab" role
  // perform.py does. Only this file's two call sites below need it today.
  let timeMap = new ScoreTimeMap();

  function scoreInfo() {
    return {
      bpm: session.noteData?.bpm,
      bpmOg: session.noteData?.bpm_og,
      transposeOffset: session.noteData?.transpose_offset,
    };
  }

  // Mirrors perform.py's _refresh_score_mistakes: push score-note-indexed
  // mistake markers into the viewer whenever the underlying mistakes,
  // overrides, active instrument, or color mode change. Both mistake
  // categories always go in together - viewer.js itself filters which one is
  // VISIBLE per the active color mode (see annotations.js's header).
  $effect(() => {
    scoreReadyTick;
    if (!scoreViewer?.ready) return;
    const annotations = buildAnnotations({
      scoreNotesActive: session.scoreNotesActive,
      userNotesActive: session.analysisResult?.note_data,
      mistakes: session.pitchMistakes.concat(session.timingMistakes),
      overridden: session.overridden,
      overrideKey: session.overrideKey,
      currentPairs: session.currentPairs,
      pitchFrames: session.analysisResult?.pitch_data?.pitches,
    });
    scoreViewer.setAnnotationColorMode(session.scoreColorMode);
    scoreViewer.setMistakeAnnotations(annotations);
  });

  // Mirrors perform.py's on_note_clicked/on_annotation_clicked: clicking a
  // note or mistake marker in the score seeks the transport there (ignored
  // while playing, matching desktop's scrub-only guard).
  //
  // Annotation clicks already carry app/MIDI time (from noteMeta.seekTime -
  // built off the same NoteData timeline playback.seek() uses) - no map
  // needed, matching desktop's on_annotation_clicked taking app_sec directly.
  function onAnnotationClicked(appSec) {
    if (playback.isPlaying) return;
    playback.seek(appSec);
  }

  // Plain note clicks only carry Verovio's OWN rendered-SVG time, which
  // drifts from MIDI time - convert via the barline map first (mirrors
  // perform.py's _score_note_start_from_viewer), then snap to the nearest
  // rendered score-note onset so the shared transport lands on a real
  // NoteData start time.
  function onNoteClicked(viewerSec) {
    if (playback.isPlaying) return;
    const appT = timeMap.appTime(viewerSec, scoreInfo());
    const starts = (session.scoreNotesActive ?? []).map((n) => n[1]);
    if (!starts.length) {
      playback.seek(appT);
      return;
    }
    const nearest = starts.reduce((best, t) =>
      Math.abs(t - appT) < Math.abs(best - appT) ? t : best
    );
    playback.seek(nearest);
  }

  // Drives the score cursor from the real playback clock - mirrors
  // app.py's time_changed, which calls ScoreViewer.set_playback_time via
  // perform.py's _score_viewer_time (the SAME barline-map conversion used
  // above, just the inverse direction). playback.svelte.js polls at the
  // same 10Hz cadence WallClock itself uses.
  $effect(() => {
    scoreReadyTick;
    if (scoreViewer?.ready) {
      scoreViewer.setPlaybackTime(timeMap.viewerTime(playback.currentTime, scoreInfo()));
    }
  });

  function base64ToBytes(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // Loads the real uploaded score into Verovio the moment /notedata returns
  // it (musicxml_b64 - see analyze_api.py), independent of Analyze ever
  // running, mirroring the desktop app's score-independent score loading.
  //
  // Re-anchors the barline time map right after (mirrors perform.py's
  // refresh_score_viewer -> _rebuild_time_map) - simpler here than desktop's
  // version, though: ScoreViewer.svelte's getMeasureTimemap() is a same-
  // origin, synchronous iframe call (unlike desktop's genuinely async Qt
  // runJavaScript-with-callback), and loadScore() itself runs Verovio's
  // render synchronously start to finish, so the timemap is already valid
  // the instant loadScore() returns - no async _store callback needed.
  let lastLoadedMusicXml = null;
  $effect(() => {
    scoreReadyTick;
    const b64 = session.noteData?.musicxml_b64;
    if (b64 && b64 !== lastLoadedMusicXml && scoreViewer?.ready) {
      lastLoadedMusicXml = b64;
      scoreViewer.loadScore(base64ToBytes(b64));
      const veroOnsets = scoreViewer.getMeasureTimemap();
      const appOnsets = session.noteData?.measure_onsets_og;
      if (veroOnsets?.length && appOnsets?.length) {
        timeMap.setAnchors(appOnsets, veroOnsets);
      } else {
        timeMap.clear();
      }
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
        <div class="score-viewer-wrap">
          <ScoreViewer bind:this={scoreViewer} {onNoteClicked} {onAnnotationClicked} onReady={() => scoreReadyTick++} />
        </div>
        <div class="score-legend-row">
          <span class="legend-swatches">
            {#if session.scoreColorMode === "volume"}
              <span class="legend-item"><span class="swatch volume-swatch"></span>volume</span>
            {:else if session.scoreColorMode === "timing"}
              <span class="legend-item"><span class="swatch" style="background:rgb(198,30,0)"></span>error</span>
            {:else}
              <span class="legend-item"><span class="swatch" style="background:rgb(198,30,0)"></span>wrong note</span>
              <span class="legend-item"><span class="swatch" style="background:rgb(212,130,0)"></span>off-pitch</span>
            {/if}
          </span>
          <label class="color-mode-picker">
            Colors:
            <select bind:value={session.scoreColorMode}>
              <option value="pitch">Pitch</option>
              <option value="timing">Timing</option>
              <option value="volume">Volume</option>
            </select>
          </label>
        </div>
      </div>
      <div class="overlay-pane">
        {#if session.analysisResult && session.scoreNotesActive}
          <NoteOverlay
            scoreNotes={session.scoreNotesActive}
            userNotes={session.analysisResult.note_data}
            pairs={session.currentPairs}
            pitchMistakes={session.pitchMistakes}
            timingMistakes={session.timingMistakes}
            pitchFrames={session.analysisResult.pitch_data?.pitches}
            vibratoPoints={session.analysisResult.vibrato?.points}
            vibMinCycles={session.analysisResult.config?.vib_min_cycles}
            pitchTolerance={session.pitchTolerance}
            currentTime={playback.currentTime}
            selectedMistake={session.selectedMistake}
            selectedMistakeOverridden={session.selectedMistake ? session.overridden.has(session.overrideKey(session.selectedMistake)) : false}
            onSeek={(t) => playback.seek(t)}
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
      <div class="results-pane">
        <ResultsView />
      </div>
      <div class="note-panel-pane">
        <NotePanel
          userNotesActive={session.analysisResult?.note_data}
          pitchFrames={session.analysisResult?.pitch_data?.pitches}
          vibratoPoints={session.analysisResult?.vibrato?.points}
          currentTime={playback.currentTime}
        />
      </div>
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
    display: flex;
    flex-direction: column;
    border-bottom: 4px solid var(--border);
  }
  .score-viewer-wrap {
    flex: 1;
    min-height: 0;
  }
  /* perform.py's _build_score_legend_row: swatches (left) + a right-aligned
     "Colors:" dropdown, same row shape as GuitarHero's own legend below. */
  .score-legend-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 2px 8px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    flex-shrink: 0;
  }
  .legend-swatches {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 0.3rem;
  }
  .volume-swatch {
    background: linear-gradient(to right, rgb(61, 1, 76), rgb(53, 74, 125), rgb(30, 131, 126), rgb(85, 181, 88), rgb(228, 208, 33));
  }
  .color-mode-picker {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  .color-mode-picker select {
    background: var(--bg-input, #2a2b30);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 0.8rem;
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
    display: flex;
    flex-direction: column;
    border-left: 4px solid var(--border);
    background: var(--bg-window);
    min-width: 240px;
    max-width: 420px;
  }
  /* app.py's right_column QSplitter: MistakeWidget+ToleranceWidget take the
     slack, NotePanel keeps a fixed size below it (setStretchFactor(1, 0)). */
  .results-pane {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .note-panel-pane {
    height: 300px;
    flex-shrink: 0;
    border-top: 4px solid var(--border);
    overflow: hidden;
  }
</style>
