"""End-to-end orchestration for the HonestEar Phase 1 pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from honest_ear.asr import build_asr_providers
from honest_ear.config import Settings, get_settings
from honest_ear.fusion import fuse_transcripts
from honest_ear.llm import request_correction
from honest_ear.schemas import PipelineResult
from honest_ear.tts import speak_with_macos_say


def run_pipeline(
    audio_path: Path,
    mode: str,
    speak_reply: bool,
    settings: Optional[Settings] = None,
) -> PipelineResult:
    """Runs dual-channel ASR, fusion, LLM correction, and optional local TTS."""

    active_settings = settings or get_settings()
    faithful_provider, intended_provider = build_asr_providers(active_settings)

    with ThreadPoolExecutor(max_workers=2) as executor:
        faithful_future = executor.submit(faithful_provider.transcribe, audio_path)
        intended_future = executor.submit(intended_provider.transcribe, audio_path)
        faithful_result = faithful_future.result()
        intended_result = intended_future.result()

    fusion = fuse_transcripts(faithful_result, intended_result, active_settings)
    llm_response = request_correction(fusion, mode, active_settings)
    tts_output = None
    if speak_reply:
        tts_output = str(speak_with_macos_say(llm_response.reply, active_settings))

    return PipelineResult(
        audio_path=audio_path,
        faithful_asr=faithful_result,
        intended_asr=intended_result,
        fusion=fusion,
        llm=llm_response,
        tts_output=tts_output,
    )
