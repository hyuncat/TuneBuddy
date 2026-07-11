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
    uvicorn web.api.main:app --reload --app-dir .
"""
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# This file lives at web/api/main.py; make the Attune project root (two
# levels up) importable regardless of the working directory it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    return payload


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
