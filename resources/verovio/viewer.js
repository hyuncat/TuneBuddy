// globals
let tk = null;
let currentPage = 1; // verovio pages start at 1
// rk: also have verovio loaded from verovio-toolkit-wasm.js

// --- CLIP SELECTION STATE ---
// Measure-range clipping lives entirely here: the user clicks a start measure
// then an end measure, and Python pulls the resulting time range on demand
// (window.getClipSelection). Two independent bits of state:
//   1. the in-progress SELECTION (selStartId/selEndId -> selInterval) drawn with
//      `.selected`, and
//   2. the active CLIP RANGE (clipRange, set by Python via window.setClipRange)
//      drawn by greying everything OUTSIDE it with `.clipped-out`.
// What CROSSES to Python is MEASURE INDICES, not seconds: Verovio's rendered
// timeline drifts ahead of the app's MIDI timeline (lossy MIDI->MusicXML round
// trip), so a clip resolved from Verovio seconds lands on the wrong notes. The
// measure index is the one landmark that's unambiguously 1:1 between the two, so
// Python resolves measures->notes off its own timeline (see
// ScoreData.note_index_range_for_measures). Seconds are kept only for the
// in-progress `.selected` highlight, which lives entirely in Verovio's frame.
// All of this is re-applied on every renderPage so it survives page flips (the
// viewer lays out one system per page, so off-page measures aren't in DOM).
let measureOnsets = new Map(); // measureId -> onset (sec)
let measureIndex = new Map();  // measureId -> index into measureOrder (score order)
let measureOrder = [];         // [{id, onset}] sorted by onset, whole score
let scoreEndSec = 0;           // largest onset seen (approx score end)
const TO_END = 1e9;            // sentinel end for "clip runs to the score end"

let selStartId = null;
let selEndId = null;
let selStage = 0;              // 0 none, 1 start placed, 2 range complete
let selInterval = null;        // {startSec, endSec, startIdx, endIdx} | null
let clipRange = null;          // {startIdx, endIdx} inclusive | null (grey-out focus)

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
    // innerHTML was just replaced, so (re)add hit areas, (re)bind clicks and
    // (re)paint overlays.
    addMeasureHitAreas();
    bindMeasureClicks();
    applyOverlays();
}

const SVGNS = "http://www.w3.org/2000/svg";

function makeRect(cls, bb) {
    const rect = document.createElementNS(SVGNS, "rect");
    rect.setAttribute("class", cls);
    rect.setAttribute("x", bb.x);
    rect.setAttribute("y", bb.y);
    rect.setAttribute("width", bb.width);
    rect.setAttribute("height", bb.height);
    return rect;
}

// The vertical extent of a measure's staff = the union of its 5 staff-line paths
// (the bare <path> children of <g class="staff">), i.e. top line -> bottom line.
// This is uniform across measures (the lines sit at fixed y), unlike the measure
// getBBox which hugs the notes and so varies in height. Width = the staff width.
function staffBBox(measureEl) {
    const staff = measureEl.querySelector("g.staff");
    if (!staff) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const child of staff.children) {
        if (child.tagName.toLowerCase() !== "path") continue; // staff lines only
        let bb;
        try { bb = child.getBBox(); } catch (e) { continue; }
        minX = Math.min(minX, bb.x); minY = Math.min(minY, bb.y);
        maxX = Math.max(maxX, bb.x + bb.width); maxY = Math.max(maxY, bb.y + bb.height);
    }
    if (minX === Infinity) return null;
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

// Verovio measures are <g> wrappers with no fill of their own, so only the drawn
// glyphs catch clicks — clicking blank space inside a measure hits nothing. Give
// each measure two rects, both BEHIND the notes (inserted as first children):
//   1. `clip-hit`       — the FULL measure bbox, transparent, pointer-events:all,
//                         so a click ANYWHERE in the measure registers (bubbles
//                         to the measure's click handler).
//   2. `clip-highlight` — the STAFF bbox (uniform height), the selection-highlight
//                         surface (filled translucent via CSS when `.selected`),
//                         pointer-events:none so it never steals clicks.
function addMeasureHitAreas() {
    const measures = document.querySelectorAll("#notation g.measure");
    for (const m of measures) {
        if (m.querySelector("rect.clip-hit")) continue; // already added

        const sb = staffBBox(m);
        if (sb) {
            const hl = makeRect("clip-highlight", sb);
            hl.setAttribute("fill", "none");
            hl.setAttribute("pointer-events", "none");
            m.insertBefore(hl, m.firstChild);
        }

        let mb;
        try { mb = m.getBBox(); } catch (e) { mb = null; }
        if (mb && mb.width > 0 && mb.height > 0) {
            const hit = makeRect("clip-hit", mb);
            hit.setAttribute("fill", "transparent");
            hit.setAttribute("pointer-events", "all");
            m.insertBefore(hit, m.firstChild);
        }
    }
}

// Build the whole-score measure -> onset map from Verovio's timemap so we know
// each measure's start time (and ordering) regardless of which page is rendered.
function buildMeasureMap() {
    measureOnsets = new Map();
    measureIndex = new Map();
    measureOrder = [];
    scoreEndSec = 0;

    let timemap = [];
    try {
        timemap = tk.renderToTimemap({ includeMeasures: true, includeRests: true });
    } catch (e) {
        console.warn("renderToTimemap failed:", e);
    }
    if (typeof timemap === "string") {
        try { timemap = JSON.parse(timemap); } catch (_) { timemap = []; }
    }
    if (!Array.isArray(timemap)) timemap = [];

    for (const entry of timemap) {
        const tsec = (entry.tstamp || 0) / 1000;
        if (tsec > scoreEndSec) scoreEndSec = tsec;
        const mid = entry.measureOn;
        if (mid && !measureOnsets.has(mid)) {
            measureOnsets.set(mid, tsec);
            measureOrder.push({ id: mid, onset: tsec });
        }
    }
    measureOrder.sort((a, b) => a.onset - b.onset);
    // index each measure by its position in score order, so a clicked measure id
    // maps straight to the index Python pairs with its own measure onsets.
    measureOrder.forEach((m, i) => measureIndex.set(m.id, i));
}

// Index of a measure id in score order, or -1 if unknown (off-map measure).
function measureIndexFor(id) {
    return measureIndex.has(id) ? measureIndex.get(id) : -1;
}

// Ordered measure-onset times (sec, Verovio's original-tempo timeframe) for the
// whole score, pulled by Python to anchor the playback cursor to the MIDI /
// NoteData timeline barline-by-barline (see ui.time.ScoreTimeMap). Returns null
// if the map isn't built yet so the host can fall back to the plain tempo scalar.
window.getMeasureTimemap = function() {
    if (!measureOrder || !measureOrder.length) return null;
    return measureOrder.map(m => m.onset);
}

// Onset (sec) for a measure id: prefer the prebuilt map, else ask Verovio
// directly (covers builds whose timemap lacks measure entries). Returns null
// if it can't be resolved.
function measureOnsetFor(id) {
    if (measureOnsets.has(id)) return measureOnsets.get(id);
    let t = -1;
    try { t = tk.getTimeForElement(id); } catch (e) { /* ignore */ }
    if (typeof t === "number" && t >= 0) {
        const s = t / 1000;
        measureOnsets.set(id, s);
        return s;
    }
    return null;
}

// Onset of the measure immediately AFTER `id` in score order, or TO_END if `id`
// is the last measure (so the clip extends through the final measure).
function onsetAfter(id) {
    const onset = measureOnsetFor(id);
    if (onset === null) return TO_END;
    for (const m of measureOrder) {
        if (m.onset > onset + 1e-6) return m.onset;
    }
    return TO_END;
}

// Recompute the selection interval [startSec, endSec) from the picked measures.
function recomputeSelInterval() {
    if (!selStartId) { selInterval = null; return; }
    const a = measureOnsetFor(selStartId);
    const b = measureOnsetFor(selEndId || selStartId);
    if (a === null || b === null) { selInterval = null; return; }
    const startSec = Math.min(a, b);
    // the later-onset measure ends the range; include its full bar
    const firstId = (a <= b) ? selStartId : (selEndId || selStartId);
    const lastId = (a <= b) ? (selEndId || selStartId) : selStartId;
    const endSec = onsetAfter(lastId);
    // measure indices (what Python clips on) + seconds (the in-progress highlight)
    selInterval = {
        startSec, endSec,
        startIdx: measureIndexFor(firstId),
        endIdx: measureIndexFor(lastId),
    };
}

// Paint `.selected` (in-progress pick) and `.clipped-out` (focus grey-out) on
// every measure currently in the DOM, by comparing each measure's onset to the
// two intervals. Idempotent; safe to call on every render.
function applyOverlays() {
    const measures = document.querySelectorAll("#notation g.measure");
    for (const m of measures) {
        // the in-progress pick is highlighted in Verovio's own (self-consistent)
        // seconds; the active clip grey-out is keyed on measure index.
        const onset = measureOnsetFor(m.id);
        const inSel = selInterval && onset !== null
            && onset >= selInterval.startSec - 1e-6
            && onset < selInterval.endSec - 1e-6;
        m.classList.toggle("selected", !!inSel);

        const idx = measureIndexFor(m.id);
        const outOfClip = clipRange && idx >= 0
            && (idx < clipRange.startIdx || idx > clipRange.endIdx);
        m.classList.toggle("clipped-out", !!outOfClip);
    }
}

function onMeasureClick(ev) {
    const id = ev.currentTarget.id;
    if (!id) return;
    if (selStage === 1) {
        // second click closes the range
        selEndId = id;
        selStage = 2;
    } else {
        // first click (stage 0) or a click after a complete range (stage 2):
        // start a fresh single-measure selection
        selStartId = id;
        selEndId = id;
        selStage = 1;
    }
    recomputeSelInterval();
    applyOverlays();
}

function bindMeasureClicks() {
    const measures = document.querySelectorAll("#notation g.measure");
    for (const m of measures) {
        m.addEventListener("click", onMeasureClick);
    }
}

// --- INIT VEROVIO TOOLKIT ---
// (hangs until WASM is ready. sets toolkit -> tk.)
// pendingLoadScore: window.loadScore can be called before the WASM runtime
// finishes initializing (a real race, not just theoretical - a host that
// calls loadScore as soon as the page/iframe itself has loaded, rather than
// waiting for Verovio's OWN readiness, can easily lose this race since WASM
// compile+instantiate takes noticeably longer than plain DOM load). This
// used to just drop the call silently (setStatus("Verovio not ready") and
// nothing else) with no retry - queue the most recent call instead and
// flush it once the toolkit is actually ready.
let pendingLoadScore = null;
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
        if (pendingLoadScore !== null) {
            const b64 = pendingLoadScore;
            pendingLoadScore = null;
            window.loadScore(b64);
        }
    };
})();

// --- PUBLIC API (called from python) ---
window.loadScore = function(b64) {
    if (!tk) { pendingLoadScore = b64; setStatus("Verovio not ready"); return; }
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

        // rebuild the measure->onset map for the freshly loaded score, and reset
        // the in-progress selection (a new layout invalidates it). The active
        // clip range is left to Python: it re-asserts it after re-renders (e.g.
        // instrument toggle) via setClipRange, and clears it on a new score.
        buildMeasureMap();
        selStartId = selEndId = null;
        selStage = 0;
        selInterval = null;

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

// --- CLIP API (called from python) ---
// The in-progress measure selection as inclusive measure INDICES (score order),
// or null if nothing is selected. Python pulls this when the user presses Clip
// and resolves the indices to notes off its own (drift-free) timeline. Returns
// null if either endpoint isn't on the measure map.
window.getClipSelection = function() {
    if (!selInterval) return null;
    if (selInterval.startIdx < 0 || selInterval.endIdx < 0) return null;
    return { startIdx: selInterval.startIdx, endIdx: selInterval.endIdx };
}

// Clear the in-progress selection + its `.selected` highlight.
window.clearClipSelection = function() {
    selStartId = selEndId = null;
    selStage = 0;
    selInterval = null;
    applyOverlays();
}

// Set the active clip range (grey out everything outside it). startIdx/endIdx are
// inclusive measure indices in score order (Python derives them from the clip's
// notes), so the grey-out tracks the same measures the clip actually holds.
window.setClipRange = function(startIdx, endIdx) {
    clipRange = { startIdx: startIdx, endIdx: endIdx };
    applyOverlays();
}

// Clear the active clip range (un-grey all measures).
window.clearClipRange = function() {
    clipRange = null;
    applyOverlays();
}
