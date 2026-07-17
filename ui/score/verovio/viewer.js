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
let selectionMode = false;     // armed by Python ("Select measures"); gates measure picking
const PLAYBACK_ONSET_LOOKAHEAD_MS = 10; // display-only: prefer the new note at exact onsets

// --- MISTAKE ANNOTATION STATE ---
// Python sends score-note-indexed mistakes (indices into ScoreData.NoteData, not
// Verovio seconds). JS maps those indices to Verovio's current note IDs by
// rendered part order, so red noteheads and insertion markers survive cursor
// time-map drift and page re-renders.
let activePartIndex = 0;
let noteIdToScoreIndex = new Map(); // Verovio note id -> ScoreData note index
let noteIndexToIds = new Map();     // ScoreData note index -> [Verovio note ids]
let mistakeAnnotations = { notes: {}, insertions: [], noteMeta: {}, volumes: {} };
let annotationColorMode = "pitch";
let popupMistakes = [];
let popupIndex = 0;
// --- THEME ---
// Python owns every color (ui/Colors.py) and pushes them here as soon as the
// page is ready, before any score loads. Each role is mirrored onto a
// --score-<role> CSS custom property (the notehead rules in viewer.css read
// those) and kept in this map, which fills the insertion markers THIS file
// injects as SVG. Empty until pushed: no second copy of the palette to drift.
let themeColors = {};

window.setThemeColors = function (colors) {
    themeColors = colors || {};
    for (const [role, value] of Object.entries(themeColors)) {
        document.documentElement.style.setProperty(`--score-${role}`, value);
    }
    applyMistakeAnnotations();
};

// --- QT BRIDGE (JS -> Python push channel) ---
// The rest of the JS API is pull-only (Python runs JS and reads the result),
// but clicks originate HERE, so they need a push path: the Qt host registers a
// 'bridge' object on the page's QWebChannel. Guarded so the page still runs
// standalone (plain browser, the offline node test) where qt/QWebChannel
// don't exist — bridge just stays null and note clicks are inert.
let bridge = null;
(function initBridge() {
    if (typeof QWebChannel === "undefined"
        || typeof qt === "undefined" || !qt.webChannelTransport) return;
    new QWebChannel(qt.webChannelTransport, (channel) => {
        bridge = channel.objects.bridge;
    });
})();

// --- HELPERS ---
function setStatus(msg) {
    // the visible status box was dropped; keep the hook (and standalone runs) alive
    const el = document.getElementById("status");
    if (el) el.textContent = msg;
}
// render page utility - the guts of how the viewer
// actually displays the current page in the score: as an SVG
// string from verovio, wrapped in a div with class "page" (for styling)
function renderPage(pageNo) {
    const svgStr = tk.renderToSVG(pageNo);
    document.getElementById("notation").innerHTML =
        `<div class="page">${svgStr}</div>`;
    hideMistakePopup();
    // innerHTML was just replaced, so (re)add hit areas, (re)bind clicks and
    // (re)paint overlays.
    addMeasureHitAreas();
    bindMeasureClicks();
    applyOverlays();
    applyMistakeAnnotations();
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
function staffLineBBox(staff) {
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

function staffBBox(measureEl) {
    return staffLineBBox(measureEl.querySelector("g.staff"));
}

// Client-rect (CSS px) counterpart of staffLineBBox: the union of the staff's
// line paths as rendered on screen. Overlay geometry (insertion noteheads) is
// computed in client space, then converted back to SVG user units per measure.
function staffLinesClientRect(staff) {
    if (!staff) return null;
    let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
    for (const child of staff.children) {
        if (child.tagName.toLowerCase() !== "path") continue; // staff lines only
        const r = child.getBoundingClientRect();
        if (!r || r.width <= 0) continue;
        left = Math.min(left, r.left); top = Math.min(top, r.top);
        right = Math.max(right, r.right); bottom = Math.max(bottom, r.bottom);
    }
    if (left === Infinity) return null;
    return { left, top, right, bottom, width: right - left, height: bottom - top };
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

function renderTimemap(options) {
    let timemap = [];
    try {
        timemap = tk.renderToTimemap(options);
    } catch (e) {
        console.warn("renderToTimemap failed:", e);
    }
    if (typeof timemap === "string") {
        try { timemap = JSON.parse(timemap); } catch (_) { timemap = []; }
    }
    return Array.isArray(timemap) ? timemap : [];
}

function childStaffs(measureEl) {
    return Array.from(measureEl.children).filter((el) => {
        return el.tagName && el.tagName.toLowerCase() === "g"
            && el.classList.contains("staff");
    });
}

function activeStaffForMeasure(measureEl) {
    const staffs = childStaffs(measureEl);
    if (!staffs.length) return null;
    const idx = Math.max(0, Math.min(activePartIndex, staffs.length - 1));
    return staffs[idx];
}

function activeNoteIdsFromSvg(svgStr) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(svgStr, "image/svg+xml");
    const ids = [];
    for (const measure of doc.querySelectorAll("g.measure")) {
        const staff = activeStaffForMeasure(measure);
        if (!staff) continue;
        for (const note of staff.querySelectorAll("g.note[id]")) {
            ids.push(note.id);
        }
    }
    // Fallback for unusual exports whose notes are not nested under measure/staff
    // the way Verovio normally emits them.
    if (!ids.length) {
        for (const note of doc.querySelectorAll("g.note[id]")) ids.push(note.id);
    }
    return ids;
}

function collectActiveNoteIdsAcrossPages() {
    const ordered = [];
    const pageCount = (typeof tk.getPageCount === "function") ? tk.getPageCount() : 1;
    for (let page = 1; page <= Math.max(1, pageCount); page++) {
        let svgStr = "";
        try { svgStr = tk.renderToSVG(page); } catch (e) { continue; }
        ordered.push(...activeNoteIdsFromSvg(svgStr));
    }
    return { ordered, ids: new Set(ordered) };
}

function rememberNoteIndex(noteId, scoreIndex) {
    noteIdToScoreIndex.set(noteId, scoreIndex);
    if (!noteIndexToIds.has(scoreIndex)) noteIndexToIds.set(scoreIndex, []);
    noteIndexToIds.get(scoreIndex).push(noteId);
}

function rebuildNoteIndexMap() {
    noteIdToScoreIndex = new Map();
    noteIndexToIds = new Map();
    if (!tk) return;

    const active = collectActiveNoteIdsAcrossPages();
    if (!active.ids.size) return;

    // Verovio timemap entries group simultaneous note-ons; NoteData also stores
    // one score Note per onset (with chord pitches merged), so all active note IDs
    // in the same onset map to the same score-note index.
    const timemap = renderTimemap({ includeMeasures: false, includeRests: false });
    let scoreIndex = -1;
    let lastStamp = null;
    for (const entry of timemap) {
        const on = Array.isArray(entry.on)
            ? entry.on.filter((id) => active.ids.has(id))
            : [];
        if (!on.length) continue;

        const stamp = Number(
            entry.tstamp !== undefined ? entry.tstamp
                : (entry.qstamp !== undefined ? entry.qstamp : scoreIndex + 1)
        );
        if (lastStamp === null || Math.abs(stamp - lastStamp) > 1e-6) {
            scoreIndex += 1;
            lastStamp = stamp;
        }
        for (const id of on) rememberNoteIndex(id, scoreIndex);
    }

    // Conservative fallback: if a Verovio build changes timemap note output, keep
    // annotations usable by mapping the active staff's rendered note order.
    if (!noteIdToScoreIndex.size) {
        active.ordered.forEach((id, idx) => rememberNoteIndex(id, idx));
    }
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

// Height (CSS px) of the rendered score content (#notation including its
// padding and the page's margins). The Qt host pulls this after each load to
// size the score pane so the whole system is visible without scrolling.
window.getContentHeight = function() {
    const el = document.getElementById("notation");
    return el ? el.offsetHeight : 0;
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
    if (!selectionMode) return; // measure picking only while armed
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

// Click on a NOTE (outside selection mode) -> push its Verovio-timeline time to
// Python so the app can seek the transport there. Document-level delegation:
// the SVG is recreated on every renderPage, but the document persists.
document.addEventListener("click", (ev) => {
    const popup = document.getElementById("mistake-popup");
    if (popup && popup.contains(ev.target)) return;
    if (selectionMode || !tk) return;

    const insertionMarker = ev.target.closest
        ? ev.target.closest(".mistake-insertion-marker")
        : null;
    if (insertionMarker) return;

    const note = ev.target.closest ? ev.target.closest("g.note") : null;
    if (!note || !note.id) {
        hideMistakePopup();
        return;
    }

    const scoreIndex = noteIdToScoreIndex.get(note.id);
    if (annotationColorMode === "volume") {
        const volume = volumeForNoteIndex(scoreIndex);
        if (volume) {
            ev.preventDefault();
            ev.stopPropagation();
            seekAnnotation(scoreIndex);
            showMistakePopup(volumePopupPayload(volume), ev.clientX, ev.clientY);
            return;
        }
    }

    const mistakes = visibleMistakesForNoteIndex(scoreIndex);
    if (mistakes.length) {
        ev.preventDefault();
        ev.stopPropagation();
        seekAnnotation(scoreIndex);
        showMistakePopup(mistakes, ev.clientX, ev.clientY);
        return;
    }

    if (!bridge) return;
    let t = -1;
    try { t = tk.getTimeForElement(note.id); } catch (e) { return; }
    if (typeof t === "number" && t >= 0) bridge.noteClicked(t / 1000);
});

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
window.loadScore = function(b64, partIndex = 0) {
    if (!tk) { setStatus("Verovio not ready"); return; }
    try {
        setStatus("Loading score...");
        activePartIndex = Number.isFinite(Number(partIndex)) ? Number(partIndex) : 0;
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
        rebuildNoteIndexMap();
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

// --- MISTAKE ANNOTATION API (called from python) ---
window.setMistakeAnnotations = function(annotations) {
    mistakeAnnotations = normalizeMistakeAnnotations(annotations);
    if (!hasMistakeAnnotations()) hideMistakePopup();
    applyMistakeAnnotations();
}

window.setAnnotationColorMode = function(mode) {
    const normalized = String(mode || "").toLowerCase();
    annotationColorMode = (normalized === "timing" || normalized === "volume")
        ? normalized
        : "pitch";
    document.body.classList.toggle("score-color-volume", annotationColorMode === "volume");
    document.body.classList.toggle("score-color-timing", annotationColorMode === "timing");
    document.body.classList.toggle("score-color-pitch", annotationColorMode === "pitch");
    applyMistakeAnnotations();
}

function normalizeMistakeAnnotations(annotations) {
    if (!annotations || typeof annotations !== "object") {
        return { notes: {}, insertions: [], noteMeta: {}, volumes: {} };
    }
    return {
        notes: annotations.notes && typeof annotations.notes === "object"
            ? annotations.notes
            : {},
        insertions: Array.isArray(annotations.insertions)
            ? annotations.insertions
            : [],
        noteMeta: annotations.noteMeta && typeof annotations.noteMeta === "object"
            ? annotations.noteMeta
            : {},
        volumes: annotations.volumes && typeof annotations.volumes === "object"
            ? annotations.volumes
            : {},
    };
}

function mistakesForNoteIndex(index) {
    if (index === undefined || index === null) return [];
    const notes = mistakeAnnotations.notes || {};
    const mistakes = notes[String(index)];
    return Array.isArray(mistakes) ? mistakes : [];
}

function visibleMistakesForNoteIndex(index) {
    const mistakes = mistakesForNoteIndex(index);
    if (annotationColorMode === "timing") {
        return mistakes.filter((m) => m.category === "timing");
    }
    if (annotationColorMode === "pitch") {
        return mistakes.filter((m) => m.category !== "timing");
    }
    return [];
}

function hasMistakeAnnotations() {
    const noteGroups = Object.values(mistakeAnnotations.notes || {});
    const hasNotes = noteGroups.some((mistakes) => Array.isArray(mistakes) && mistakes.length);
    const hasInsertions = (mistakeAnnotations.insertions || []).some((slot) => {
        return Array.isArray(slot.mistakes) && slot.mistakes.length;
    });
    return hasNotes || hasInsertions;
}

function noteMetaForIndex(index) {
    if (index === undefined || index === null) return null;
    return mistakeAnnotations.noteMeta[String(index)] || null;
}

function volumeForNoteIndex(index) {
    if (index === undefined || index === null) return null;
    return mistakeAnnotations.volumes[String(index)] || null;
}

function volumePopupPayload(volume) {
    const db = volume && Number.isFinite(Number(volume.db))
        ? `${Number(volume.db).toFixed(1)} dB`
        : "unavailable";
    return [{
        type: "volume",
        title: "Volume",
        rows: [{ value: db }],
    }];
}

function dominantMistakeKind(mistakes) {
    const types = new Set((mistakes || []).map((m) => m.type));
    if (types.has("deletion")) return "deletion";
    if (types.has("substitution")) return "substitution";
    if (types.has("early") || types.has("late") || types.has("long") || types.has("short")) {
        return "timing";
    }
    if (types.has("insertion")) return "insertion";
    return "";
}

function seekAnnotation(scoreIndex) {
    if (!bridge) return;
    const meta = noteMetaForIndex(scoreIndex);
    const sec = meta ? Number(meta.seekTime) : NaN;
    if (Number.isFinite(sec)) bridge.annotationClicked(sec);
}

function seekInsertionSlot(slot) {
    if (!bridge) return;
    const sec = Number(slot && slot.seekTime);
    if (Number.isFinite(sec)) bridge.annotationClicked(sec);
}

function clearMistakeVisuals() {
    for (const note of document.querySelectorAll("#notation g.note")) {
        note.classList.remove(
            "mistake",
            "mistake-deletion",
            "mistake-substitution",
            "mistake-insertion",
            "mistake-timing",
            "volume-colored",
        );
        note.style.removeProperty("--score-volume-color");
    }
    for (const marker of document.querySelectorAll("#notation .mistake-insertion-marker")) {
        marker.remove();
    }
}

function applyMistakeAnnotations() {
    clearMistakeVisuals();
    for (const note of document.querySelectorAll("#notation g.note[id]")) {
        const scoreIndex = noteIdToScoreIndex.get(note.id);
        if (scoreIndex === undefined) continue;
        note.dataset.scoreIndex = String(scoreIndex);
        const volume = volumeForNoteIndex(scoreIndex);
        if (annotationColorMode === "volume" && volume && volume.color) {
            note.classList.add("volume-colored");
            note.style.setProperty("--score-volume-color", volume.color);
            continue;
        }

        if (annotationColorMode === "volume") continue;
        const visibleMistakes = visibleMistakesForNoteIndex(scoreIndex);
        if (visibleMistakes.length) {
            const kind = annotationColorMode === "timing"
                ? "timing"
                : dominantMistakeKind(visibleMistakes);
            note.classList.add("mistake");
            if (kind) note.classList.add(`mistake-${kind}`);
        }
    }
    renderInsertionMarkers();
}

function noteElementsForIndex(index) {
    if (index === null || index === undefined) return [];
    const ids = noteIndexToIds.get(Number(index)) || [];
    return ids.map((id) => document.getElementById(id)).filter(Boolean);
}

function unionDomRect(elements) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const el of elements) {
        const rect = el.getBoundingClientRect();
        if (!rect || rect.width <= 0 || rect.height <= 0) continue;
        minX = Math.min(minX, rect.left);
        minY = Math.min(minY, rect.top);
        maxX = Math.max(maxX, rect.right);
        maxY = Math.max(maxY, rect.bottom);
    }
    if (minX === Infinity) return null;
    return { left: minX, top: minY, right: maxX, bottom: maxY, width: maxX - minX, height: maxY - minY };
}

// Union client rect of the NOTEHEADS at a score index (chords span several).
// Deliberately heads-only: stems, flags and accidentals would bias the
// insertion marker's midpoint toward whichever neighbor carries wider glyphs.
function noteheadRectForIndex(index) {
    const heads = noteElementsForIndex(index)
        .map((el) => el.querySelector("g.notehead") || el);
    return unionDomRect(heads);
}

function firstVisibleNoteForSlot(slot) {
    const right = noteElementsForIndex(slot.rightIndex);
    if (right.length) return right[0];
    const left = noteElementsForIndex(slot.leftIndex);
    return left.length ? left[left.length - 1] : null;
}

// midi -> on-screen y for the active staff, least-squares fitted from the
// rendered noteheads (whose score pitches Python ships in noteMeta.midis).
// Chord heads are paired with chord pitches by vertical order. Interpolating in
// semitones ignores diatonic spelling, but stays within a fraction of a staff
// space — fine for placing an insertion's played-pitch notehead.
function activeStaffPitchFit() {
    const pts = [];
    for (const [index, ids] of noteIndexToIds) {
        const meta = noteMetaForIndex(index);
        const midis = meta && Array.isArray(meta.midis)
            ? meta.midis.filter(Number.isFinite).slice().sort((a, b) => a - b)
            : [];
        if (!midis.length) continue;
        const heads = ids
            .map((id) => document.getElementById(id)) // filters to the current page
            .filter(Boolean)
            .map((el) => (el.querySelector("g.notehead") || el).getBoundingClientRect())
            .filter((r) => r && r.height > 0)
            .map((r) => r.top + r.height / 2)
            .sort((a, b) => b - a); // lowest pitch (largest y) first
        if (!heads.length) continue;
        if (heads.length === midis.length) {
            heads.forEach((y, i) => pts.push({ midi: midis[i], y }));
        } else {
            // head<->pitch pairing ambiguous; use the note as one mean anchor
            pts.push({
                midi: midis.reduce((s, m) => s + m, 0) / midis.length,
                y: heads.reduce((s, y) => s + y, 0) / heads.length,
            });
        }
    }
    if (!pts.length) return null;

    let sm = 0, sy = 0, smm = 0, smy = 0;
    for (const p of pts) { sm += p.midi; sy += p.y; smm += p.midi * p.midi; smy += p.midi * p.y; }
    const n = pts.length;
    const denom = n * smm - sm * sm;
    if (Math.abs(denom) > 1e-6) {
        const slope = (n * smy - sm * sy) / denom;
        if (slope < 0) return { slope, intercept: (sy - slope * sm) / n };
    }
    // degenerate fit (e.g. every rendered note is the same pitch): slope from
    // staff geometry — a semitone ~ 7/12 of a diatonic step = half a line gap.
    const measure = document.querySelector("#notation g.measure");
    const lines = staffLinesClientRect(measure ? activeStaffForMeasure(measure) : null);
    if (!lines || lines.height <= 0) return null;
    const slope = -(7 / 12) * (lines.height / 8);
    return { slope, intercept: pts[0].y - slope * pts[0].midi };
}

function insertionMarkerPosition(slot, pitchFit) {
    const anchor = firstVisibleNoteForSlot(slot);
    if (!anchor) return null;
    const measureEl = anchor.closest("g.measure");
    if (!measureEl) return null;
    const staffLines = staffLinesClientRect(anchor.closest("g.staff"));
    const leftRect = noteheadRectForIndex(slot.leftIndex);
    const rightRect = noteheadRectForIndex(slot.rightIndex);
    if (!leftRect && !rightRect) return null;

    let x;
    if (leftRect && rightRect) {
        // midpoint of the two notehead CENTERS = visually equal spacing
        x = ((leftRect.left + leftRect.right) + (rightRect.left + rightRect.right)) / 4;
    } else if (rightRect) {
        x = rightRect.left - 9;
    } else {
        x = leftRect.right + 9;
    }
    if (staffLines) {
        x = Math.max(staffLines.left + 10, Math.min(staffLines.right - 10, x));
    }

    // y: the PLAYED pitch (slot.midi = mean over the slot's insertions) through
    // the fitted midi->y map; above the staff only when there's no pitch to place.
    const gap = staffLines ? staffLines.height / 4 : 10;
    const midi = Number(slot.midi);
    let y = null;
    if (pitchFit && Number.isFinite(midi)) {
        y = pitchFit.slope * midi + pitchFit.intercept;
    }
    if (y === null) {
        y = (staffLines
            ? staffLines.top
            : Math.min(leftRect ? leftRect.top : Infinity, rightRect ? rightRect.top : Infinity)
        ) - 10;
    }
    // keep extreme extrapolations inside the page's SVG so the head can't clip away
    const svg = measureEl.ownerSVGElement;
    if (svg) {
        const sr = svg.getBoundingClientRect();
        y = Math.max(sr.top + gap / 2, Math.min(sr.bottom - gap / 2, y));
    }
    return { measureEl, x, y, gap };
}

// Draw the insertion notehead INSIDE the measure's SVG (not an HTML overlay):
// it scales with the score on resize and inherits the measure's clipped-out
// dimming. Client-space position/size are converted through the measure's CTM.
function appendInsertionNotehead(slot, pos, color, popupItemsFor) {
    const ctm = pos.measureEl.getScreenCTM();
    if (!ctm) return;
    const local = new DOMPoint(pos.x, pos.y).matrixTransform(ctm.inverse());
    const scale = Math.hypot(ctm.a, ctm.b) || 1; // client px per user unit
    const rx = 0.66 * pos.gap / scale;           // ~black-notehead proportions
    const ry = 0.48 * pos.gap / scale;

    const g = document.createElementNS(SVGNS, "g");
    g.setAttribute("class", "mistake-insertion-marker");

    const hit = document.createElementNS(SVGNS, "circle"); // easier click target
    hit.setAttribute("cx", local.x);
    hit.setAttribute("cy", local.y);
    hit.setAttribute("r", 2 * rx);
    hit.setAttribute("fill", "transparent");
    hit.setAttribute("pointer-events", "all");
    g.appendChild(hit);

    const head = document.createElementNS(SVGNS, "ellipse");
    head.setAttribute("cx", local.x);
    head.setAttribute("cy", local.y);
    head.setAttribute("rx", rx);
    head.setAttribute("ry", ry);
    head.setAttribute("fill", color);
    head.setAttribute("transform", `rotate(-20 ${local.x} ${local.y})`);
    g.appendChild(head);

    g.addEventListener("click", (ev) => {
        if (selectionMode) return; // fall through to measure picking
        ev.preventDefault();
        ev.stopPropagation();
        seekInsertionSlot(slot); // seekTime = the slot's FIRST insertion onset
        showMistakePopup(popupItemsFor(), ev.clientX, ev.clientY);
    });
    pos.measureEl.appendChild(g);
}

function renderInsertionMarkers() {
    if (annotationColorMode === "timing") return;
    const slots = mistakeAnnotations.insertions || [];
    if (!slots.length) return;
    const pitchFit = activeStaffPitchFit();

    for (const slot of slots) {
        const mistakes = Array.isArray(slot.mistakes) ? slot.mistakes : [];
        const visibleMistakes = annotationColorMode === "pitch"
            ? mistakes.filter((m) => m.category !== "timing")
            : mistakes;
        if (!visibleMistakes.length && annotationColorMode !== "volume") continue;
        const pos = insertionMarkerPosition(slot, pitchFit);
        if (!pos) continue;

        const volume = slot.volume || null;
        const color = annotationColorMode === "volume" && volume && volume.color
            ? volume.color
            : (themeColors.insertion || "currentColor");
        appendInsertionNotehead(slot, pos, color, () => (
            annotationColorMode === "volume"
                ? volumePopupPayload(slot.volume)
                : visibleMistakes
        ));
    }
}

function ensureMistakePopup() {
    let popup = document.getElementById("mistake-popup");
    if (popup) return popup;
    popup = document.createElement("div");
    popup.id = "mistake-popup";
    popup.addEventListener("click", (ev) => ev.stopPropagation());
    document.body.appendChild(popup);
    return popup;
}

function hideMistakePopup() {
    const popup = document.getElementById("mistake-popup");
    if (popup) popup.style.display = "none";
}

function showMistakePopup(mistakes, clientX, clientY) {
    if (!Array.isArray(mistakes) || !mistakes.length) return;
    popupMistakes = mistakes;
    popupIndex = 0;
    const popup = ensureMistakePopup();
    renderMistakePopup();
    popup.style.display = "block";
    positionMistakePopup(popup, clientX, clientY);
}

function renderMistakePopup() {
    const popup = ensureMistakePopup();
    const mistake = popupMistakes[popupIndex] || {};
    popup.innerHTML = "";

    const title = document.createElement("div");
    title.className = "mistake-popup-title";
    title.textContent = mistake.title || mistake.type || "Mistake";
    popup.appendChild(title);

    for (const row of mistake.rows || []) {
        const line = document.createElement("div");
        line.className = "mistake-popup-row";
        line.textContent = row.label ? `${row.label}: ${row.value}` : `${row.value}`;
        popup.appendChild(line);
    }

    if (popupMistakes.length > 1) {
        const controls = document.createElement("div");
        controls.className = "mistake-popup-controls";

        const prev = document.createElement("button");
        prev.type = "button";
        prev.textContent = "←";
        prev.addEventListener("click", (ev) => {
            ev.stopPropagation();
            popupIndex = (popupIndex - 1 + popupMistakes.length) % popupMistakes.length;
            renderMistakePopup();
        });

        const next = document.createElement("button");
        next.type = "button";
        next.textContent = "→";
        next.addEventListener("click", (ev) => {
            ev.stopPropagation();
            popupIndex = (popupIndex + 1) % popupMistakes.length;
            renderMistakePopup();
        });

        const count = document.createElement("span");
        count.className = "mistake-popup-count";
        count.textContent = `${popupIndex + 1}/${popupMistakes.length}`;

        controls.appendChild(prev);
        controls.appendChild(count);
        controls.appendChild(next);
        popup.appendChild(controls);
    }
}

function positionMistakePopup(popup, clientX, clientY) {
    const margin = 12;
    const x = Math.min(clientX + margin, window.innerWidth - popup.offsetWidth - margin);
    const y = Math.min(clientY + margin, window.innerHeight - popup.offsetHeight - margin);
    popup.style.left = `${Math.max(margin, x)}px`;
    popup.style.top = `${Math.max(margin, y)}px`;
}

window.timeChanged = function(sec) {
    if (!tk) return;

    // 1) remove 'playing' from any notes previously highlighted
    const playingNotes = document.querySelectorAll("g.note.playing");
    for (const n of playingNotes) n.classList.remove("playing");

    // 2) ask verovio which elements are at this time (expects milliseconds).
    // At exact boundaries, Verovio can still report the note just before the
    // onset after MusicXML quantization. Querying a few ms ahead is display-only:
    // it never feeds back into the app timeline, slider, GuitarHero, or players.
    const currentElements = tk.getElementsAtTime(
        sec * 1000 + PLAYBACK_ONSET_LOOKAHEAD_MS
    );
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
// Arm/disarm measure-range picking (the Clip menu's "Select measures"). Only
// while armed do measure clicks build a selection — otherwise clicks fall
// through to the note-seek handler. Arming clears any stale selection (fresh
// pick); disarming leaves the selection to Python (set_clip clears it).
window.setSelectionMode = function(on) {
    selectionMode = !!on;
    hideMistakePopup();
    if (selectionMode) {
        selStartId = selEndId = null;
        selStage = 0;
        selInterval = null;
    }
    document.body.classList.toggle("selecting", selectionMode);
    applyOverlays();
}

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
