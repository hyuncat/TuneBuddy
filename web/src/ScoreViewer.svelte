<script>
  // Mirrors ui/ScoreViewer.py: owns the Verovio iframe, tracks readiness, and
  // exposes the same JS-API wrapper methods Python calls via runJavaScript
  // (window.loadScore, window.timeChanged, window.getMeasureTimemap,
  // window.getClipSelection, window.clearClipSelection, window.setClipRange,
  // window.clearClipRange - see resources/verovio/viewer.js). Same-origin
  // iframe, so these are direct synchronous calls instead of Python's
  // runJavaScript(..., callback) dance.
  let ready = $state(false);
  let iframeEl;

  function handleLoad() {
    ready = true;
  }

  function callIframe(fnName, ...args) {
    if (!ready || !iframeEl?.contentWindow) return undefined;
    const fn = iframeEl.contentWindow[fnName];
    if (typeof fn !== "function") return undefined;
    return fn(...args);
  }

  // 32KB chunks to avoid call-stack limits from spreading a large byte array
  // into String.fromCharCode at once.
  function bytesToBase64(bytes) {
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
  }

  // xmlBytes: Uint8Array of MusicXML content (uncompressed text, matching what
  // ScoreData.to_musicxml_bytes() produces server-side - not a raw .mxl zip).
  export function loadScore(xmlBytes) {
    if (!ready) {
      console.warn("[ScoreViewer] loadScore called before ready, ignoring.");
      return;
    }
    callIframe("loadScore", bytesToBase64(xmlBytes));
  }

  export function setPlaybackTime(sec) {
    callIframe("timeChanged", sec);
  }

  export function getMeasureTimemap() {
    return callIframe("getMeasureTimemap");
  }

  export function getClipSelection() {
    return callIframe("getClipSelection");
  }

  export function clearClipSelection() {
    callIframe("clearClipSelection");
  }

  export function setClipRange(startIdx, endIdx) {
    callIframe("setClipRange", startIdx, endIdx);
  }

  export function clearClipRange() {
    callIframe("clearClipRange");
  }

  export function isReady() {
    return ready;
  }
</script>

<div class="score-viewer">
  {#if !ready}
    <div class="loading">Loading...</div>
  {/if}
  <iframe
    bind:this={iframeEl}
    onload={handleLoad}
    src="/verovio/viewer.html"
    title="Score viewer"
    class:hidden={!ready}
  ></iframe>
</div>

<style>
  .score-viewer {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 300px;
  }
  .loading {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary, #999);
    background: var(--bg-surface, #242427);
  }
  iframe {
    width: 100%;
    height: 100%;
    border: none;
  }
  iframe.hidden {
    visibility: hidden;
  }
</style>
