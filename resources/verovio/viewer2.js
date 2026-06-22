// globals
let tk = null;
let currentPage = 1; // verovio pages start at 1
// rk: also have verovio loaded from verovio-toolkit-wasm.js

// --- HELPERS ---
function setStatus(msg) {
    document.getElementById("status").textContent = msg;
}
// render page utility - the guts of how the viewer
// actually displays the current page in the score: as an SVG
// string from verovio, wrapped in a div with class "page" (for styling)
function renderPage(pageNo) {
    const svgStr = tk.renderToSVG(pageNo);
    document.getElementById("notation").innerHTML = 
        `<div class="page">${svgStr}</div>`;
}

// --- INIT VEROVIO TOOLKIT ---
// (hangs until WASM is ready. sets toolkit -> tk.)
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

// --- PUBLIC API (called from python) ---
window.loadScore = function(b64) {
    if (!tk) { setStatus("Verovio not ready"); return; }
    try {
        setStatus("Loading score...");
        // lay the score out one system (line) per "page" so that paging
        // through the score = scrolling line by line. adjustPageHeight trims
        // each page down to the height of its single system, so the line fills
        // the widget instead of floating in a tall blank page.
        tk.setOptions({
            systemMaxPerPage: 1,
            adjustPageHeight: true,
            breaks: "auto",
        });
        // decode from base64 -> ascii and load into toolkit as string
        tk.loadData(atob(b64));

        // now render the loaded page (first line) with verovio
        currentPage = 1;
        renderPage(currentPage);
        setStatus("Ready");
    } catch (e) {
        console.error(e);
        setStatus("Error");
        alert("Failed to load/render: " + e);
    }
}

window.timeChanged = function(sec) {
    if (!tk) return;
    
    // 1) remove 'playing' from any notes previously highlighted
    const playingNotes = document.querySelectorAll("g.note.playing");
    for (const n of playingNotes) n.classList.remove("playing");

    // 2) ask verovio which elements are at this time (expects milliseconds)
    const currentElements = tk.getElementsAtTime(sec * 1000);
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
}
