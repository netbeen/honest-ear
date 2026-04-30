"""FastAPI app exposing the HonestEar Phase 1 processing pipeline and demo UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
import urllib.request
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from honest_ear.asr import warmup_asr_models
from honest_ear.config import get_settings
from honest_ear.llm import LLMRequestError
from honest_ear.pipeline import run_pipeline
from honest_ear.samples import load_sample_records
from honest_ear.schemas import PipelineResult, ProcessAudioRequest, SampleRecord


# #region debug-point C:report-runtime-event
def _report_debug_event(
    hypothesis_id: str, location: str, message: str, data: dict, trace_id: str
) -> None:
    """Reports one runtime debug event to the local debug server when enabled."""

    debug_env_path = Path(__file__).resolve().parents[2] / ".dbg" / "process-upload-latency.env"
    debug_url = "http://127.0.0.1:7777/event"
    session_id = "process-upload-latency"
    try:
        if debug_env_path.exists():
            for line in debug_env_path.read_text().splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_url = line.split("=", 1)[1].strip()
                if line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip()
        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {message}",
            "data": data,
            "traceId": trace_id,
            "ts": int(time.time() * 1000),
        }
        request = urllib.request.Request(
            debug_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=0.5).read()
    except Exception:
        pass


# #endregion
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Preloads ASR models before serving requests unless explicitly skipped."""

    _ = _app
    settings = get_settings()
    if not settings.skip_asr_warmup:
        try:
            warmup_asr_models(settings)
        except FileNotFoundError as exc:
            raise RuntimeError(f"ASR startup failed. {exc}") from exc
    yield


app = FastAPI(title="HonestEar Phase 1 API", version="0.1.0", lifespan=lifespan)
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def _save_upload_to_temp_file(upload: UploadFile) -> Path:
    """Persists one uploaded audio file to a temporary wav file for pipeline processing."""

    suffix = Path(upload.filename or "recording.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(upload.file.read())
        return Path(temp_file.name)


def _build_tts_audio_url(tts_output: str | None) -> str | None:
    """Builds one local API URL that streams the generated TTS audio file."""

    if not tts_output:
        return None
    return f"/v1/tts-audio?path={quote(tts_output, safe='')}"


def _attach_tts_audio_url(result: PipelineResult) -> PipelineResult:
    """Adds one browser-playable TTS audio URL to the pipeline result when available."""

    return result.model_copy(update={"tts_audio_url": _build_tts_audio_url(result.tts_output)})


def _resolve_tts_audio_path(raw_path: str) -> Path:
    """Validates that the requested TTS file stays within the local temp directory."""

    candidate = Path(raw_path).expanduser().resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    allowed_suffixes = {".aiff", ".aif", ".wav", ".m4a"}

    if temp_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="TTS audio path must stay inside the temp directory.")
    if candidate.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Unsupported TTS audio file type.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="TTS audio file not found.")

    return candidate


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    """Serves the minimal recording UI used for Phase 1 validation."""

    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Provides a simple readiness probe for local development."""

    return {"status": "ok"}


@app.get("/v1/tts-audio", response_class=FileResponse)
def get_tts_audio(path: str = Query(..., description="Absolute path of the generated local TTS audio file.")) -> FileResponse:
    """Streams one generated local TTS audio file back to the browser."""

    tts_audio_path = _resolve_tts_audio_path(path)
    return FileResponse(tts_audio_path)


@app.get("/v1/samples", response_model=list[SampleRecord])
def list_samples() -> list[SampleRecord]:
    """Returns the built-in Phase 1 evaluation samples."""

    settings = get_settings()
    return load_sample_records(settings.sample_dataset_path)


@app.post("/v1/process", response_model=PipelineResult)
def process_audio(request: ProcessAudioRequest) -> PipelineResult:
    """Runs the full Phase 1 loop for a local audio file."""

    try:
        result = run_pipeline(
            audio_path=request.audio_path,
            mode=request.mode,
            speak_reply=request.speak_reply,
            settings=get_settings(),
        )
        return _attach_tts_audio_url(result)
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/process-upload", response_model=PipelineResult)
def process_uploaded_audio(
    audio: UploadFile = File(...),
    mode: str = Form("accuracy"),
    speak_reply: bool = Form(False),
) -> PipelineResult:
    """Accepts one uploaded recording, stores it temporarily, and runs the pipeline."""

    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded audio filename.")

    trace_id = str(uuid.uuid4())
    request_started_at = time.perf_counter()
    # #region debug-point C:upload-start
    _report_debug_event(
        "C",
        "api.py:process_uploaded_audio:start",
        "Upload processing started.",
        {"filename": audio.filename, "mode": mode, "speak_reply": speak_reply},
        trace_id,
    )
    # #endregion

    save_started_at = time.perf_counter()
    temp_audio_path = _save_upload_to_temp_file(audio)
    # #region debug-point C:upload-saved
    _report_debug_event(
        "C",
        "api.py:process_uploaded_audio:save",
        "Uploaded file persisted to temp path.",
        {
            "duration_ms": round((time.perf_counter() - save_started_at) * 1000, 2),
            "temp_audio_path": str(temp_audio_path),
            "size_bytes": temp_audio_path.stat().st_size,
        },
        trace_id,
    )
    # #endregion

    try:
        try:
            result = run_pipeline(
                audio_path=temp_audio_path,
                mode=mode,
                speak_reply=speak_reply,
                settings=get_settings(),
                trace_id=trace_id,
            )
            # #region debug-point D:upload-finished
            _report_debug_event(
                "D",
                "api.py:process_uploaded_audio:done",
                "Upload processing finished.",
                {
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
                    "temp_audio_path": str(temp_audio_path),
                },
                trace_id,
            )
            # #endregion
            return _attach_tts_audio_url(result)
        except LLMRequestError as exc:
            # #region debug-point B:upload-llm-error
            _report_debug_event(
                "B",
                "api.py:process_uploaded_audio:llm-error",
                "LLM request failed during upload processing.",
                {
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000, 2),
                    "error": str(exc),
                },
                trace_id,
            )
            # #endregion
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        temp_audio_path.unlink(missing_ok=True)
