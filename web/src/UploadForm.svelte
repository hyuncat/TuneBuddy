<script>
  // Uploads a score + a recording to the backend's POST /analyze and hands the
  // parsed JSON result back to the parent via onResult. Doesn't render results
  // itself - that's a separate concern (results view, not built yet).
  import { getNoteData } from "./noteDataCache.js";

  let { onResult, onNoteData } = $props();

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Mirrors web/api/analyze_api.py's SUPPORTED_SCORE_EXTENSIONS and the
  // extension set actually enforced in check_upload_file's call for audio.
  // This is a UX hint for the OS file picker only, NOT a validation boundary -
  // the backend's check_upload_file is the real authority, and this is
  // trivially bypassed (drag-and-drop, or picking "all files").
  const SCORE_ACCEPT = ".mid,.midi,.mxl,.musicxml,.xml,.mei";
  const AUDIO_ACCEPT = ".wav,.wave,.flac,.ogg,.aif,.aiff,.m4a";

  let scoreFile = $state(null);
  let audioFile = $state(null);
  let status = $state("idle"); // "idle" | "loading" | "error" | "success"
  let errorMessage = $state("");
  let noteDataError = $state("");

  // Fetches (or reuses the cache for) the score's note data as soon as it's
  // selected - decoupled from the Analyze submit, since the mistake widget
  // needs it independent of whether/when an analysis actually runs, and it
  // shouldn't block the Analyze button if it fails.
  async function handleScoreChange(e) {
    scoreFile = e.target.files?.[0] ?? null;
    noteDataError = "";
    if (!scoreFile) return;
    try {
      const data = await getNoteData(scoreFile, API_BASE_URL);
      onNoteData?.(data);
    } catch (err) {
      noteDataError = err instanceof Error ? err.message : String(err);
    }
  }

  function handleAudioChange(e) {
    audioFile = e.target.files?.[0] ?? null;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!scoreFile || !audioFile) return;

    status = "loading";
    errorMessage = "";

    // multipart/form-data, not JSON - the backend expects raw file uploads
    // (FastAPI's UploadFile = File(...) params), and these field names
    // ("score", "audio") must match those parameter names exactly.
    const formData = new FormData();
    formData.append("score", scoreFile);
    formData.append("audio", audioFile);

    try {
      // Deliberately NOT setting a Content-Type header: the browser sets
      // multipart/form-data with the correct boundary itself when the body
      // is a FormData instance. Setting it manually would break that.
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        // FastAPI's HTTPException responses look like {"detail": "..."};
        // surface that instead of a generic failure message.
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || `Request failed (${response.status})`);
      }

      const data = await response.json();
      status = "success";
      onResult?.(data);
    } catch (err) {
      status = "error";
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }
</script>

<form onsubmit={handleSubmit}>
  <label>
    Score (MIDI/MusicXML)
    <input type="file" accept={SCORE_ACCEPT} onchange={handleScoreChange} />
  </label>
  {#if noteDataError}
    <p class="error">Couldn't load score note data: {noteDataError}</p>
  {/if}
  <label>
    Recording
    <input type="file" accept={AUDIO_ACCEPT} onchange={handleAudioChange} />
  </label>
  <button type="submit" disabled={!scoreFile || !audioFile || status === "loading"}>
    {status === "loading" ? "Analyzing..." : "Analyze"}
  </button>
  {#if status === "error"}
    <p class="error">{errorMessage}</p>
  {/if}
</form>

<style>
  form {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    font-family: system-ui, sans-serif;
    max-width: 400px;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.9rem;
  }
  .error {
    color: #c0392b;
  }
</style>
