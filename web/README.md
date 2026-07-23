# Attune Web

A web deployment of Attune, scoped to **upload-only** for now (no live
in-browser recording, no real-time Practice-mode gating — see project
history for why). This reuses the existing Python analysis pipeline
(`algorithms/`, `app_logic/`) unchanged via a thin FastAPI wrapper, and a
Svelte frontend that mirrors the desktop app's actual layout (toolbar, left
sidebar, center score + pitch overlay, right mistake table, transport bar).

**Current status**: functionally complete for the upload → analyze →
review workflow. Upload a score + recording, get real notation rendering,
mistake detection with adjustable pitch/timing tolerance, a live per-frame
pitch overlay, MIDI playback of the score with instrument/metronome muting,
and Range/Tuning controls that default to the score's own pitch span.
Mistakes are reachable from all three of the desktop app's interaction
surfaces: the mistake table (click-to-highlight, mirrors GuitarHero's
highlight-and-pan behavior), colored inline annotations directly on the
rendered score (click a note/insertion marker for a popup - see
`annotations.js`, ported from `ui/score/ScoreAnnotations.py`), and clicking a
user note in the pitch overlay itself (a GuitarHero-style popup with
pitch/cents/onset/duration/volume - see `NoteOverlay.svelte`, ported from
`ui/guitarhero/GuitarHero.py`'s `select_note`). Pitch, timing, and volume are
all wired up: a "Colors:" dropdown next to the score switches its annotation
color mode (mirrors `perform.py`'s `score_color_mode`), and GuitarHero's own
pitch/volume dot-coloring toggle is ported too (`colors.js` has the viridis
ramp + take-relative volume math, ported from `ui/Colors.py` +
`PitchData.mean_volume`/`volume_range_db`). Vibrato is the one NoteInfo field
still unported (shows "—" - no vibrato detection client-side yet). Not yet
done: deployment (still runs locally only), a proper ScoreTimeMap port (score clicks before
any analysis has run use Verovio's own rendered timeline as a best-effort
approximation - see `onNoteClicked` in `App.svelte`), and a few toolbar stubs
(Settings, Save, Clip) that are either dead ends in the desktop app itself
(Settings) or real features not yet built (Save, Clip) — see inline comments
in `Toolbar.svelte` for the current state of each.

## Running it

**One command, one terminal:**

```bash
cd web
npm run dev:all
```

This starts both the backend (`uvicorn`, port 8000) and frontend (`vite`,
usually port 5173 but falls back automatically if that's taken) as a single
process group, with output prefixed `[backend]`/`[frontend]` so the two
stay visually distinct. `Ctrl+C` stops both cleanly.

Open the URL the `[frontend]` lines print, upload a score
(`resources/demo/scales/major/major.mxl` is a good one to start with) and a
recording (`resources/demo/scales/major/ashwin.wav`), and click Analyze.

First-time setup (skip if you've already done this once):

```bash
cd web
npm install
npm run sync-verovio   # copies resources/verovio -> web-resources/
npm run sync-icons     # copies resources/icons -> web-resources/
npm run sync-synth     # copies the MuseScore soundfont + js-synthesizer's WASM runtime -> web-resources/
python3.14 -m venv ../venv           # if you don't already have one at the project root
source ../venv/bin/activate
pip install -r api/requirements.txt
```

**If you'd rather run the two processes separately** (e.g. to restart just
the backend without touching the frontend, or to watch their logs in
separate windows), that still works exactly as before:

```bash
# Terminal 1 — from the project root
source venv/bin/activate
uvicorn web.api.analyze_api:app --app-dir . --port 8000

# Terminal 2
cd web
npm run dev
```

## Prerequisites

- **Python 3.x** — verified working with 3.14. Avoid very new Python
  releases without a moment's checking first: `praat-parselmouth` (a
  dependency of the *desktop* app, deliberately excluded from
  `web/api/requirements.txt` since nothing `/analyze` touches imports it)
  fails to build from source on 3.14, which is exactly the kind of thing to
  watch for on a bleeding-edge interpreter.
- **Node.js** — verified working with v22.19.0 / npm 10.9.3.
- **ffmpeg** on your `PATH` — required by `pydub` to decode `.m4a` uploads
  (phone voice-memo recordings). Without it, `.m4a` uploads will fail even
  though `pip install` succeeds; `.wav`/`.flac`/`.ogg`/`.aif`/`.aiff` don't
  need it (handled by `soundfile` directly).

## Notes on setup

Run the backend from the **Attune project root** (not from `web/api/` —
the app needs to import `algorithms/` and `app_logic/`, which live at the
root); `npm run backend` (and therefore `npm run dev:all`) already handles
this by `cd`-ing up a level before invoking `uvicorn`.

Note on the `python3.14 -m venv` pin: `python -m venv` accepts multiple
directory arguments and creates a venv in each one, so if you add anything
after the target directory on that line — including a trailing `#`
comment, if whatever runs the command doesn't treat `#` as a shell comment
— it'll silently create extra venv-shaped folders named after each word.
Keep that line exactly as written, with nothing appended.

`python3.14` specifically, not a generic `python3` — that's the only
version this has actually been installed and run against. The pinned
versions in `requirements.txt` (scipy, PyQt6, etc.) haven't been verified
against older 3.x interpreters, and a plain `python3` could silently
resolve to something else on a machine with multiple Python versions
installed. If you only have an older 3.x available, it may well work, but
treat it as unverified rather than assumed-fine.

Verify the backend's up on its own:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

One thing worth knowing if you look at `requirements.txt`: it includes
`PyQt6`. That's not a mistake — `algorithms/PitchDetector.py` and
`algorithms/NoteDetector.py` subclass `QObject` for Qt signals the desktop
app uses, so importing them pulls in PyQt6 even though this API never
touches any GUI code. It imports and runs fine headless, no display
needed. It is real dead weight for a deployed container, though (see task
#6 in project notes) — worth trimming when this gets containerized.

CORS is set to allow any `localhost`/`127.0.0.1` port (not a fixed one) —
deliberately, since Vite's dev port isn't stable on every machine. Dev-only;
tighten this to the real deployed frontend origin before shipping.

**`npm run sync-verovio` / `sync-icons` / `sync-synth` are not optional on
a fresh checkout.** They copy assets from the project root
(`resources/verovio`, `resources/icons`, `resources/MuseScore_General.sf3`
+ the js-synthesizer WASM runtime) into `web/web-resources/`, which Vite
serves as static files. `web/web-resources/` is gitignored (generated, not
committed), so skipping any of these means the corresponding feature 404s
at runtime — the score viewer, the toolbar/mistake-table icons, or MIDI
playback, respectively. None of the three run automatically on `npm run
dev` — re-run them manually if the underlying files in `resources/` ever
change.

## Layout

```
web/
├── api/                      # FastAPI backend
│   ├── analyze_api.py        # POST /analyze, /notedata, /realign
│   └── requirements.txt
├── scripts/
│   ├── sync-verovio.mjs      # ui/score/verovio -> web-resources/verovio
│   ├── sync-icons.mjs        # resources/icons -> web-resources/icons
│   └── sync-synth.mjs        # soundfont + js-synthesizer WASM -> web-resources/synth
├── src/
│   ├── App.svelte            # top-level layout: toolbar/sidebar/center/results/transport
│   ├── Toolbar.svelte        # Upload/Settings/Save/Clip/Playback/Tempo/Metronome
│   ├── RecordingTree.svelte  # left sidebar: score + recording tree
│   ├── SettingsPanel.svelte  # left sidebar: Instrument/Range/Tuning/Transpose
│   ├── ScoreViewer.svelte    # Verovio notation iframe
│   ├── NoteOverlay.svelte    # GuitarHero-equivalent pitch overlay
│   ├── ResultsView.svelte    # mistake table + tolerance controls
│   ├── TransportBar.svelte   # play/pause/seek/Analyze
│   ├── StatusBar.svelte
│   ├── sessionState.svelte.js  # shared reactive app state (upload, analysis, mistakes)
│   ├── playback.svelte.js      # MIDI playback engine (js-synthesizer + SMF)
│   ├── smf.js                  # builds a Standard MIDI File from note_data
│   ├── mistakes.js             # client-side mistake classification + note-name<->MIDI/Hz
│   ├── annotations.js          # client-side port of ScoreAnnotations.py - builds the
│   │                            # score-note-indexed mistake payload ScoreViewer's iframe renders
│   ├── colors.js                # viridis ramp + volume math, ported from ui/Colors.py + PitchData
│   ├── noteDataCache.js        # content-hash-keyed cache for /notedata responses
│   ├── realign.js              # debounced /realign calls for pitch-tolerance changes
│   ├── theme.css               # dark theme, extracted from qdarktheme's real stylesheet
│   └── main.js
├── web-resources/            # generated by the sync-* scripts, gitignored
├── JSPitchDetector.js        # JS port of the real-time pitch detector - NOT
│                              # used yet (deferred; only needed if/when live
│                              # Practice-mode recording comes back)
├── index.html
├── package.json
└── vite.config.js
```
