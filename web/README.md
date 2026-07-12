# Attune Web (prototype)

A web deployment of Attune, scoped to **upload-only** for now (no live
in-browser recording, no real-time Practice-mode gating — see project
history for why). This reuses the existing Python analysis pipeline
(`algorithms/`, `app_logic/`) unchanged via a thin FastAPI wrapper, and a
Svelte frontend for rendering/UI.

**Current status**: early prototype. The pieces below are individually built
and verified working, but not yet wired into one integrated app:
- Backend `/analyze` endpoint: done, runs the real pipeline end-to-end.
- Frontend score rendering (`ScoreViewer.svelte` + Verovio): done.
- Not yet built: MIDI playback, the real upload UI, results display,
  deployment.

## Running it

This is **two separate processes in two separate terminals** — the backend
and frontend don't talk to each other yet (that wiring is still-pending task
#4), so right now you're running and checking each independently, not one
connected app.

**Terminal 1 — backend** (first time only: create the venv and install
requirements; see "Running the backend" below for the full commands):

```bash
source venv/bin/activate
uvicorn web.api.analyze_api:app --reload --app-dir .
```

- Health check: `http://localhost:8000/health`
- Try `/analyze` without writing any code: open `http://localhost:8000/docs`
  (FastAPI's auto-generated interactive form) and upload a score + audio file
  through the browser — e.g. `resources/demo/scales/major/major.mxl` +
  `resources/demo/scales/major/evan.wav`.

**Terminal 2 — frontend** (first time only: `npm install` and
`npm run sync-verovio`; see "Running the frontend" below):

```bash
cd web
npm run dev
```

- Open the URL it prints (port varies — Vite picks another one automatically
  if 5173 is already taken on your machine).
- Click "Load test score (ScoreViewer verification)" to confirm the score
  viewer renders.

Full one-time setup for each is below.

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

## Running the backend

Run from the **Attune project root** (not from `web/api/` — the app needs to
import `algorithms/` and `app_logic/`, which live at the root):

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r web/api/requirements.txt
uvicorn web.api.analyze_api:app --reload --app-dir .
```

On Windows, activate with `venv\Scripts\activate` instead of the `source`
line above.

Note on the `python3.14` pin: `python -m venv` accepts multiple directory
arguments and creates a venv in each one, so if you add anything after
`venv` on that line — including a trailing `#` comment, if whatever runs the
command doesn't treat `#` as a shell comment — it'll silently create extra
venv-shaped folders named after each word. Keep that line exactly as written,
with nothing appended.

`python3.14` specifically, not a generic `python3` — that's the only version
this has actually been installed and run against. The pinned versions in
`requirements.txt` (scipy, PyQt6, etc.) haven't been verified against older
3.x interpreters, and a plain `python3` could silently resolve to something
else on a machine with multiple Python versions installed. If you only have
an older 3.x available, it may well work, but treat it as unverified rather
than assumed-fine.

Verify it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

One thing worth knowing if you look at `requirements.txt`: it includes
`PyQt6`. That's not a mistake — `algorithms/PitchDetector.py` and
`algorithms/NoteDetector.py` subclass `QObject` for Qt signals the desktop
app uses, so importing them pulls in PyQt6 even though this API never touches
any GUI code. It imports and runs fine headless, no display needed.

CORS is set to allow any `localhost`/`127.0.0.1` port (not a fixed one) —
deliberately, since Vite's dev port isn't stable on every machine (see the
frontend section). Dev-only; tighten this to the real deployed frontend
origin before shipping.

## Running the frontend

```bash
cd web
npm install
npm run sync-verovio   # REQUIRED before first run - see below
npm run dev
```

Then open the URL it prints. That port is **not guaranteed to be 5173** —
Vite falls back to another port automatically if 5173 is already in use on
your machine, and the backend's CORS setting (above) already accounts for
that, so nothing further to configure either way.

**`npm run sync-verovio` is not optional on a fresh checkout.** The Verovio
viewer assets (`resources/verovio/` at the project root — the same files the
desktop app's score viewer uses) get copied into `web/web-resources/verovio/`
by this script, which Vite then serves. `web/web-resources/` is gitignored
(generated, not committed), so skipping this step means the score viewer will
404 trying to load `/verovio/viewer.html`. It's deliberately not run
automatically on every `npm run dev` (see the comment in
`web/scripts/sync-verovio.mjs` for why) — re-run it manually if
`resources/verovio` ever changes upstream.

## Quick smoke test

**Backend**, from the project root, using the real demo fixtures (or use
`http://localhost:8000/docs` instead if you'd rather do this through a
browser form than `curl`):

```bash
curl -X POST http://localhost:8000/analyze \
  -F "score=@resources/demo/scales/major/major.mxl" \
  -F "audio=@resources/demo/scales/major/evan.wav"
```

Should return a JSON payload with `pitch_data`, `note_data`, and `alignment`
sections populated (not empty).

**Frontend**: with `npm run dev` running, open the printed URL, click "Load
test score (ScoreViewer verification)". A treble clef with a single note
should render inside the frame below the button, and its status label
(bottom-left of the frame) should read "Ready". That button and the score it
loads are temporary verification scaffolding in `App.svelte`, not real app
content yet — it'll be replaced once the actual upload UI (task #4) exists.

## Layout

```
web/
├── api/                   # FastAPI backend (the /analyze endpoint)
│   ├── analyze_api.py
│   └── requirements.txt
├── scripts/
│   └── sync-verovio.mjs   # copies resources/verovio -> web-resources/verovio
├── src/
│   ├── App.svelte         # currently just the ScoreViewer test harness
│   ├── ScoreViewer.svelte
│   └── main.js
├── web-resources/         # generated by sync-verovio, gitignored
├── PitchDetector.js       # JS port of the real-time pitch detector - NOT
│                           # used yet (deferred; only needed if/when live
│                           # Practice-mode recording comes back)
├── index.html
├── package.json
└── vite.config.js
```
