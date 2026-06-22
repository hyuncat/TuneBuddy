let tk = null;
let currentPage = 1;

// small status helper
function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}
function clearNotation() {
  document.getElementById("notation").innerHTML = "";
}

// render a single page (we swap later if needed)
function renderPage(pageNo) {
  const svgStr = tk.renderToSVG(pageNo);
  document.getElementById("notation").innerHTML = `<div class="page">${svgStr}</div>`;
}

// called from Python
window.loadScoreBase64 = function (ext, b64) {
  if (!tk) { setStatus("Verovio not ready"); return; }
  try {
    setStatus("Loading score…");
    if (ext === ".mxl") tk.loadZipDataBase64(b64);
    else tk.loadData(atob(b64));

    currentPage = 1;
    renderPage(currentPage);
    setStatus("Ready");
  } catch (e) {
    console.error(e);
    setStatus("Error");
    alert("Failed to load/render: " + e);
  }
};

// called from Python every wall_clock tick during playback.
// tSec is SECONDS.
window.setPlaybackTime = function (tSec) {
  if (!tk) return;

  // 1) remove 'playing' from any notes previously highlighted
  const playingNotes = document.querySelectorAll("g.note.playing");
  for (const n of playingNotes) n.classList.remove("playing");

  // 2) ask verovio which elements are at this time (expects milliseconds)
  const currentElements = tk.getElementsAtTime(tSec * 1000); // guide pattern :contentReference[oaicite:2]{index=2}
  if (!currentElements || currentElements.page === 0) return;

  // 3) if the active elements are on a different page, render that page
  if (currentElements.page !== currentPage) {
    currentPage = currentElements.page;
    renderPage(currentPage);
  }

  // 4) highlight the notes by adding class 'playing'
  // currentElements.notes is a list of element IDs
  for (const noteId of currentElements.notes) {
    const el = document.getElementById(noteId);
    if (el) el.classList.add("playing");
  }
};

// init verovio runtime
(function init() {
  // make sure verovio was imported from verovio-toolkit-wasm.js
  if (typeof verovio === "undefined" || !verovio.module) {
    setStatus("Failed to load...");
    return;
  }
  // initialize toolkit once WASM runtime is ready
  verovio.module.onRuntimeInitialized = () => {
    tk = new verovio.toolkit();
    setStatus("Ready");
  };
})();

