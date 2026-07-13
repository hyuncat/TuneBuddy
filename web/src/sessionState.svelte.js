// Shared reactive session state, replacing what used to be prop-drilled
// through App -> UploadForm/ResultsView. Needed once the desktop app's
// actual panel split (toolbar upload, center score+overlay, right mistake
// table, bottom transport) pulled those three apart into separate
// components that all need the same underlying state.
//
// Single module-level instance: this app has exactly one score/recording
// in flight at a time (matching the desktop app's single active Recording),
// so a singleton is simpler than threading context through every component.
import { getNoteData } from "./noteDataCache.js";
import { realign, debounce } from "./realign.js";
import { classifyPitchMistakes, classifyTimingMistakes } from "./mistakes.js";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function createSessionState() {
  let scoreFile = $state(null);
  let audioFile = $state(null);
  let noteData = $state(null);
  let noteDataError = $state("");
  // User's Instrument-selector override (SettingsPanel); null means "use the
  // score's own default" (score_data.get_default_instrument()).
  let selectedInstrument = $state(null);

  let analysisResult = $state(null);
  let analyzeStatus = $state("idle"); // "idle" | "loading" | "error"
  let analyzeError = $state("");

  let pitchTolerance = $state(0.5);
  let timingTolerance = $state(0.25);
  let mode = $state("pitch"); // "pitch" | "timing" - mirrors MistakeWidget's mode dropdown

  let realignedPairs = $state(null);
  let realignError = $state("");
  let realigning = $state(false);

  // Client-side-only mistake dismissal (see ResultsView's original note on
  // this) - keyed by mode+pairIndex+type since a pair can appear in both an
  // onset AND duration timing mistake.
  let overridden = $state(new Set());

  // Transient StatusBar message (mirrors app.py's StatusBar.status_label).
  let statusMessage = $state("");

  const debouncedRealign = debounce(async (tolerance) => {
    if (!analysisResult || !noteData) return;
    const scoreNotes = scoreNotesForActiveInstrument();
    if (!scoreNotes) return;
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

  function activeInstrument() {
    if (selectedInstrument != null) return selectedInstrument;
    if (analysisResult) return analysisResult.recording.active_instrument;
    if (noteData) return noteData.active_instrument;
    return null;
  }

  function scoreNotesForActiveInstrument() {
    if (!noteData) return null;
    const ch = activeInstrument();
    return ch != null ? noteData.note_data[String(ch)] : null;
  }

  let currentPairs = $derived(realignedPairs ?? analysisResult?.alignment?.pairs ?? null);

  let scoreNotesActive = $derived(scoreNotesForActiveInstrument());

  let pitchMistakes = $derived.by(() => {
    if (!currentPairs || !analysisResult || !scoreNotesActive) return [];
    return classifyPitchMistakes(
      currentPairs,
      analysisResult.note_data,
      scoreNotesActive,
      pitchTolerance
    );
  });

  let timingMistakes = $derived.by(() => {
    if (!currentPairs || !analysisResult || !scoreNotesActive) return [];
    return classifyTimingMistakes(
      currentPairs,
      analysisResult.note_data,
      scoreNotesActive,
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

  async function pickScore(file) {
    scoreFile = file;
    noteDataError = "";
    selectedInstrument = null;
    if (!file) {
      noteData = null;
      return;
    }
    statusMessage = `Loading ${file.name}...`;
    try {
      noteData = await getNoteData(file, API_BASE_URL);
      statusMessage = `Loaded ${file.name}`;
    } catch (err) {
      noteDataError = err instanceof Error ? err.message : String(err);
      statusMessage = "Failed to load score.";
    }
  }

  function pickAudio(file) {
    audioFile = file;
    statusMessage = file ? `Recording set: ${file.name}` : "";
  }

  async function runAnalyze() {
    if (!scoreFile || !audioFile) return;
    analyzeStatus = "loading";
    analyzeError = "";
    statusMessage = "Analyzing...";

    const formData = new FormData();
    formData.append("score", scoreFile);
    formData.append("audio", audioFile);
    if (pitchTolerance != null) {
      formData.append("pitch_tolerance", String(pitchTolerance));
    }
    if (selectedInstrument != null) {
      formData.append("active_instrument", String(selectedInstrument));
    }

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Request failed (${response.status})`);
      }
      analysisResult = await response.json();
      analyzeStatus = "idle";
      overridden = new Set();
      realignedPairs = null; // fresh alignment supersedes any prior /realign result
      statusMessage = "Analysis complete.";
    } catch (err) {
      analyzeStatus = "error";
      analyzeError = err instanceof Error ? err.message : String(err);
      statusMessage = "Analysis failed.";
    }
  }

  function setPitchTolerance(value) {
    pitchTolerance = value;
    debouncedRealign(pitchTolerance); // no-op pre-analysis; guarded above
  }

  function setTimingTolerance(value) {
    // No /realign: timing-mistake reclassification is a pure client-side
    // threshold check over the existing pairs.
    timingTolerance = value;
  }

  function setSelectedInstrument(channel) {
    selectedInstrument = channel;
  }

  return {
    get scoreFile() { return scoreFile; },
    get audioFile() { return audioFile; },
    get noteData() { return noteData; },
    get noteDataError() { return noteDataError; },
    get analysisResult() { return analysisResult; },
    get analyzeStatus() { return analyzeStatus; },
    get analyzeError() { return analyzeError; },
    get pitchTolerance() { return pitchTolerance; },
    get timingTolerance() { return timingTolerance; },
    get mode() { return mode; },
    set mode(v) { mode = v; },
    get realigning() { return realigning; },
    get realignError() { return realignError; },
    get overridden() { return overridden; },
    get statusMessage() { return statusMessage; },
    get currentPairs() { return currentPairs; },
    get scoreNotesActive() { return scoreNotesActive; },
    get pitchMistakes() { return pitchMistakes; },
    get timingMistakes() { return timingMistakes; },
    get visibleMistakes() { return visibleMistakes; },
    get activeInstrument() { return activeInstrument(); },
    get selectedInstrument() { return selectedInstrument; },

    pickScore,
    pickAudio,
    runAnalyze,
    setPitchTolerance,
    setTimingTolerance,
    setSelectedInstrument,
    overrideKey,
    toggleOverride,
  };
}

export const session = createSessionState();
