"""FastAPI /analyze endpoint.

Upload a score (MIDI/MusicXML) + a recording, run Attune's existing analysis
pipeline unchanged, get the results back as JSON. This is a thin adapter, not
a reimplementation: every analysis step below is the same call sequence
app.py/perform.py already make (Recording.load_audio + detect_pitches for
import, PerformTab.analyze()'s note/mistake pipeline for analysis), just
triggered by an HTTP request instead of a button click, and serialized with
the same JsonHandler used for the desktop app's local `.json.xz` cache.

Run (from the Attune project root):
    pip install -r web/api/requirements.txt
    uvicorn web.api.analyze_api:app --reload --app-dir .
"""
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# This file lives at web/api/analyze_api.py; make the Attune project root (two
# levels up) importable regardless of the working directory it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.Config import Config  # noqa: E402
from algorithms.MistakeDetector import MistakeDetector  # noqa: E402
from app_logic.JsonHandler import JsonHandler  # noqa: E402
from app_logic.midi.ScoreData import ScoreData  # noqa: E402
from app_logic.user.ds.Recording import Recording  # noqa: E402

# ScoreData.load()'s supported extensions
SUPPORTED_SCORE_EXTENSIONS = {".mid", ".midi", ".mxl", ".musicxml", ".xml", ".mei"}
# Audio extensions supported by Soundfile and pydub
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}

app = FastAPI(title="Attune Analyze API")

# Dev-only: any localhost/127.0.0.1 port. Vite's dev port isn't fixed on every
# machine (autoPort reassigns it if 5173 is already taken by another project),
# so a fixed origin list breaks in exactly that case. Tighten this to the real
# deployed frontend origin before shipping.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    #Post is the only relevant method because frontend will only send Post to the API
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}



@app.post("/analyze")
async def analyze(score: UploadFile = File(...), audio: UploadFile = File(...)) -> dict:
    #checking filetype
    score_suffix = await check_upload_file(score, SUPPORTED_SCORE_EXTENSIONS)
    # AudioData.load_data only special-cases .m4a (via pydub); everything else
    # falls to soundfile.read(), which can't decode .mp3 - keep this list to
    # what that call path actually supports (mirrors app.py's writable_suffixes).
    audio_suffix = await check_upload_file(
        audio, {".wav", ".wave", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / f"score{score_suffix}"
        audio_path = Path(tmpdir) / f"audio{audio_suffix}"
        score_path.write_bytes(await score.read())
        audio_path.write_bytes(await audio.read())

        try:
            score_data = ScoreData()
            score_data.load(str(score_path))

            rec = Recording(score_data=score_data)
            rec.load_audio(str(audio_path), score_filepath=str(score_path), load_cache=False)
            rec.detect_pitches()

            # mirrors perform.py analyze method's call sequence
            rec.reset_analysis()
            rec.detect_notes()
            rec.detect_mistakes()
            rec.mistake_checker.mistake_correction_loop()
            rec.reindex_mistakes()
            rec.update_alignment_distances()
            rec.mistake_detector.detect_timing_mistakes()
            rec.trim_end()

            payload = JsonHandler(rec).to_cache_payload(
                score_filepath=str(score_path), recording_name="upload"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Mistake classification now happens client-side, against whatever
    # tolerance is currently set in the UI - the client needs the raw
    # alignment (which user note maps to which score note) to redo that
    # itself, not mistakes pre-filtered at whatever tolerance this server-side
    # Config happened to use. JsonHandler's own payload shape is shared with
    # the desktop app's local .json.xz cache format and deliberately left
    # unmodified - this trims only the API response.
    payload["alignment"] = {"pairs": payload["alignment"]["pairs"]}
    return payload


@app.post("/notedata")
async def note_data(score: UploadFile = File(...)) -> dict:
    """Parse an uploaded score and return its note-by-note data, per
    instrument channel. No GET counterpart by design - the client caches this
    response itself (keyed by a hash of the score file's content) rather than
    the server persisting it behind an id; see project notes on why. Index i
    in note_data[channel] corresponds to index i in /analyze's alignment pairs
    for that same channel (both walk NoteData.times, which is kept sorted) -
    verified against app_logic/NoteData.py, not assumed.
    """
    score_data = await _parse_uploaded_score(score)
    # _note_data_to_payload doesn't touch JsonHandler's own recording state,
    # so a handler with no Recording attached is fine here - reusing it avoids
    # a second implementation of the same note serialization drifting out of
    # sync with the desktop app's cache format.
    handler = JsonHandler()
    return {
        "title": score_data.title,
        "bpm": score_data.bpm,
        "active_instrument": score_data.active_instrument,
        "instruments": sorted(int(ch) for ch in score_data.instruments),
        "note_data": {
            str(channel): handler._note_data_to_payload(nd)
            for channel, nd in score_data.note_datas.items()
        },
    }


class RealignRequest(BaseModel):
    # Same array shape /analyze's "note_data" and /notedata's per-channel
    # note_data already return - the client sends back exactly what it
    # already has, no reshaping on either side.
    user_notes: list
    score_notes: list
    pitch_tolerance: float


@app.post("/realign")
async def realign(payload: RealignRequest) -> dict:
    """Re-run pitch-mistake alignment at a new pitch tolerance, without
    re-uploading audio or re-running pitch/note detection. This calls the
    real MistakeDetector.detect_pitch_mistakes() - the same production path
    Recording.detect_mistakes() uses by default (onset_aware=False) - rather
    than a JS reimplementation: pitch tolerance is baked into the alignment's
    own DP cost matrix (see algorithms/MistakeDetector.py's
    _substitution_cost), so a tolerance change can genuinely change which
    notes pair with which, not just relabel a fixed pairing. Reusing the
    real algorithm here means zero risk of a hand-ported version silently
    producing a different (wrong) alignment than the desktop app would.

    Only pitch tolerance goes through this endpoint - timing-mistake
    reclassification (early/late/short/long) is a simple fixed-pairs
    threshold check with no re-alignment involved, so that stays purely
    client-side against /analyze's existing pairs.
    """
    handler = JsonHandler()
    try:
        user_nd = handler._note_data_from_payload(payload.user_notes)
        score_nd = handler._note_data_from_payload(payload.score_notes)
    except (IndexError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Malformed note data: {e}")

    config = Config(pitch_tolerance=payload.pitch_tolerance)
    detector = MistakeDetector(config=config)
    try:
        aligned_pairs, _mistakes = detector.detect_pitch_mistakes(
            user_string=user_nd, midi_string=score_nd
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Realignment failed: {e}")

    # Index maps built from the SAME ordering (NoteData.times) the client's
    # original arrays used, so returned indices resolve correctly against
    # data the client already holds - reuses the identical helpers
    # JsonHandler._alignment_to_payload uses for /analyze's own pairs, so
    # indexing behavior can't drift between the two endpoints.
    user_notes_full = [user_nd.data[t] for t in user_nd.times]
    score_notes_full = [score_nd.data[t] for t in score_nd.times]
    user_maps = JsonHandler._note_index_maps(user_notes_full)
    score_maps = JsonHandler._note_index_maps(score_notes_full)

    return {
        "pairs": [
            [
                JsonHandler._lookup_note_index(u, user_maps),
                JsonHandler._lookup_note_index(s, score_maps),
            ]
            for u, s in aligned_pairs
        ]
    }


#Helper function to check file extension (refactored for both score and audio to use the same call)
async def check_upload_file(file: UploadFile, allowed_extensions: set[str]) -> str:
    """Check that the uploaded file has an allowed extension, and return it."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {sorted(allowed_extensions)}"
            ),
        )
    return suffix

#helper method for converting an uploaded score file into a ScoreData object, with validation and cleanup - refactored for both analyze and notedata endpoints to use the same call
async def _parse_uploaded_score(score: UploadFile) -> ScoreData:
    """Validate + parse an uploaded score file into a ScoreData object.
    ScoreData.load() extracts everything it needs into the object graph, so
    the temp file is safe to clean up before returning - nothing reads from
    the path afterward."""
    score_suffix = await check_upload_file(score, SUPPORTED_SCORE_EXTENSIONS)
    with tempfile.TemporaryDirectory() as tmpdir:
        score_path = Path(tmpdir) / f"score{score_suffix}"
        score_path.write_bytes(await score.read())
        score_data = ScoreData()
        try:
            score_data.load(str(score_path))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return score_data
