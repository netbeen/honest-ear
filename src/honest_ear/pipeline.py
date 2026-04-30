"""End-to-end orchestration for the HonestEar Phase 1 pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Optional
import urllib.request
import uuid

from honest_ear.asr import build_asr_providers
from honest_ear.config import Settings, get_settings
from honest_ear.fusion import fuse_transcripts
from honest_ear.llm import request_correction
from honest_ear.schemas import PipelineResult
from honest_ear.tts import speak_with_macos_say


# #region debug-point A:report-runtime-event
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
def run_pipeline(
    audio_path: Path,
    mode: str,
    speak_reply: bool,
    settings: Optional[Settings] = None,
    trace_id: Optional[str] = None,
) -> PipelineResult:
    """Runs dual-channel ASR, fusion, LLM correction, and optional local TTS."""

    active_settings = settings or get_settings()
    active_trace_id = trace_id or str(uuid.uuid4())
    pipeline_started_at = time.perf_counter()

    # #region debug-point D:pipeline-start
    _report_debug_event(
        "D",
        "pipeline.py:run_pipeline:start",
        "Pipeline execution started.",
        {"audio_path": str(audio_path), "mode": mode, "speak_reply": speak_reply},
        active_trace_id,
    )
    # #endregion

    providers_started_at = time.perf_counter()
    faithful_provider, intended_provider = build_asr_providers(active_settings)
    # #region debug-point D:provider-build
    _report_debug_event(
        "D",
        "pipeline.py:run_pipeline:providers",
        "ASR providers ready.",
        {"duration_ms": round((time.perf_counter() - providers_started_at) * 1000, 2)},
        active_trace_id,
    )
    # #endregion

    def _transcribe_with_timing(provider, channel: str):
        """Runs one ASR transcription and reports its wall-clock latency."""

        started_at = time.perf_counter()
        result = provider.transcribe(audio_path)
        # #region debug-point A:asr-finished
        _report_debug_event(
            "A",
            "pipeline.py:run_pipeline:transcribe",
            "ASR transcription finished.",
            {
                "channel": channel,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "model_name": result.model_name,
                "text_length": len(result.text),
            },
            active_trace_id,
        )
        # #endregion
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        faithful_future = executor.submit(_transcribe_with_timing, faithful_provider, "faithful")
        intended_future = executor.submit(_transcribe_with_timing, intended_provider, "intended")
        faithful_result = faithful_future.result()
        intended_result = intended_future.result()

    fusion_started_at = time.perf_counter()
    fusion = fuse_transcripts(faithful_result, intended_result, active_settings)
    # #region debug-point D:fusion-finished
    _report_debug_event(
        "D",
        "pipeline.py:run_pipeline:fusion",
        "Transcript fusion finished.",
        {
            "duration_ms": round((time.perf_counter() - fusion_started_at) * 1000, 2),
            "diff_spans": len(fusion.diff_spans),
            "should_correct": fusion.should_correct,
        },
        active_trace_id,
    )
    # #endregion

    llm_started_at = time.perf_counter()
    llm_response = request_correction(fusion, mode, active_settings)
    # #region debug-point B:llm-finished
    _report_debug_event(
        "B",
        "pipeline.py:run_pipeline:llm",
        "LLM correction request finished.",
        {
            "duration_ms": round((time.perf_counter() - llm_started_at) * 1000, 2),
            "reply_length": len(llm_response.reply),
            "correction_count": len(llm_response.corrections),
        },
        active_trace_id,
    )
    # #endregion

    tts_output = None
    if speak_reply:
        tts_started_at = time.perf_counter()
        tts_output = str(speak_with_macos_say(llm_response.reply, active_settings))
        # #region debug-point D:tts-finished
        _report_debug_event(
            "D",
            "pipeline.py:run_pipeline:tts",
            "TTS finished.",
            {"duration_ms": round((time.perf_counter() - tts_started_at) * 1000, 2)},
            active_trace_id,
        )
        # #endregion

    # #region debug-point D:pipeline-finished
    _report_debug_event(
        "D",
        "pipeline.py:run_pipeline:done",
        "Pipeline execution finished.",
        {"duration_ms": round((time.perf_counter() - pipeline_started_at) * 1000, 2)},
        active_trace_id,
    )
    # #endregion

    return PipelineResult(
        audio_path=audio_path,
        faithful_asr=faithful_result,
        intended_asr=intended_result,
        fusion=fusion,
        llm=llm_response,
        tts_output=tts_output,
    )
