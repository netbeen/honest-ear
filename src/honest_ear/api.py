"""FastAPI app exposing the HonestEar Phase 1 processing pipeline and demo UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from honest_ear.asr import warmup_asr_models
from honest_ear.config import get_settings
from honest_ear.llm import LLMRequestError
from honest_ear.pipeline import run_pipeline
from honest_ear.samples import load_sample_records
from honest_ear.schemas import PipelineResult, ProcessAudioRequest, SampleRecord


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


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    """Serves the minimal recording UI used for Phase 1 validation."""

    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Provides a simple readiness probe for local development."""

    return {"status": "ok"}


@app.get("/v1/samples", response_model=list[SampleRecord])
def list_samples() -> list[SampleRecord]:
    """Returns the built-in Phase 1 evaluation samples."""

    settings = get_settings()
    return load_sample_records(settings.sample_dataset_path)


@app.post("/v1/process", response_model=PipelineResult)
def process_audio(request: ProcessAudioRequest) -> PipelineResult:
    """Runs the full Phase 1 loop for a local audio file."""

    try:
        return run_pipeline(
            audio_path=request.audio_path,
            mode=request.mode,
            speak_reply=request.speak_reply,
            settings=get_settings(),
        )
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

    temp_audio_path = _save_upload_to_temp_file(audio)
    try:
        try:
            return run_pipeline(
                audio_path=temp_audio_path,
                mode=mode,
                speak_reply=speak_reply,
                settings=get_settings(),
            )
        except LLMRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        temp_audio_path.unlink(missing_ok=True)
