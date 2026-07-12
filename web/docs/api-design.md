# API & client-cache design notes

Design decisions for `web/api/analyze_api.py`'s three POST endpoints and
`web/src/noteDataCache.js`, captured while they were being made.

## Scope constraint driving all of this

It is important to note that the web version will be **upload-only**,
and these design decisions are meant to reflect this.

## `POST /analyze`

Runs the full pipeline once: score + audio in, aligned note data +
mistake pairs out. Mirrors `perform.py`'s analyze call sequence
exactly, using the same `JsonHandler` the desktop app uses for its
local `.json.xz` cache — one serialization format, not two to keep in
sync.

**Optional `pitch_tolerance` form field.** The desktop app lets a user set
the tolerance slider *before* ever clicking Analyze
(`on_tolerance_applied` / `reanalyze_if_analyzed` gate on "does a
recording exist," not "has analysis run yet"). Matching that meant
`/analyze` needed to accept a starting tolerance rather than always
aligning at `Config`'s fixed 0.5-semitone default and forcing a second
round-trip to fix it up.

**Mistake calculation handled client-side.** We handle the mistake
calculation on the client side in order to quickly fit the tolerance —
as a result, we return both note data and mistake pairs at the current
tolerance so that we can adapt this data on the client side.

## `POST /notedata`

Parses a score file only — no audio, no analysis — and returns its
note-by-note data per instrument channel.

**Why it exists as a separate call:** score parsing is cheap and
doesn't need a recording; analysis is expensive and does. Additionally,
a separate API endpoint allows for the same decoupling of score and
recording the desktop app gets for free by having the score loaded into
memory independent of any performance/recording.

**No related GET /notedata endpoint.** A server-side "save this parse,
hand back an id, GET it later" design would need a persistence layer
this web app doesn't otherwise have. Instead the client hashes the
file's own bytes and caches the response itself — see
[noteDataCache](#client-cache-notedatacachejs) below.

## `POST /realign`

Re-runs *only* the pitch-alignment step at a new tolerance, given note
data the client already has (from `/analyze` or `/notedata`) — no
re-upload of audio, no re-running pitch/note detection.

**Why call back into Python instead of porting the DP to JS:** pitch
tolerance isn't a post-hoc filter on a fixed alignment — it's baked into
`MistakeDetector._substitution_cost`'s cost matrix, so changing it can
change *which* notes pair with which, not just relabel a fixed pairing.
A hand-ported JS version risks silently diverging from what the desktop
app would actually produce. Calling the real
`MistakeDetector.detect_pitch_mistakes()` (the same `onset_aware=False`
path `Recording.detect_mistakes()` uses by default) makes that
divergence impossible by construction.

**Scope is deliberately narrow — pitch only.** We can actually
realign timing mistakes without having to go through all the same call
structures, but pitch requires much deeper system calls.

**Kept separate from `/analyze`, not merged.** Considered folding
`/realign`'s logic into `/analyze` itself (e.g. giving `_align` a
tolerance parameter it could be called with directly) — reverted that
attempt (see `git log` around `algorithms/MistakeDetector.py`) once it
became clear the actual ask was de-duplicating logic *within*
`analyze_api.py`, not changing the shared desktop+web algorithm file.
The two endpoints stay separate because they sit at genuinely different
costs: `/analyze` re-runs pitch detection, note detection, and the full
mistake-correction loop (expensive, audio-bound); `/realign` re-runs one
DP alignment (cheap, called on every slider drag via `debounce`, 250ms).
Merging them would force the expensive path to run on every tolerance
tweak, or require a "just realign" flag that's really a second endpoint
wearing the first one's name.

## Client cache: `noteDataCache.js`

A plain in-memory `Map`, keyed by SHA-256 hash of the score file's
**content** (not filename), wrapping `/notedata`. The cache holds note
data for the scores that have been loaded.

**Why content-hash, not filename:** two uploads can share a filename
while differing in content (e.g. a re-exported edit saved over the same
name). A filename key would silently serve stale note data for the
second upload. For example, a user might use `c major scale` as the
file name for multiple scores of c major scales in different octaves.

**Why this exists instead of a server-side cache:** `/notedata` has no
GET-by-id (see above) — there's nothing server-side to reference. The
client already holds the file it hashed and the response it got back, so
letting it hold the mapping directly is strictly simpler than inventing
a server-side store this app doesn't otherwise need.

**Session-scoped, not persisted.** A page reload clears the `Map` and
the next score selection just re-fetches. No `localStorage`/IndexedDB —
there was no requirement motivating that durability, and adding it would
be solving a problem that doesn't exist yet.

**Fired eagerly, not gated on Analyze.** `UploadForm.svelte` calls
`getNoteData()` from `handleScoreChange` (the moment a score file is
picked), independent of whether/when the user ever clicks Analyze. This
is what lets the tolerance widget and mistake-table structure exist
before an analysis has run, mirroring the desktop app's score-independent
score loading.
