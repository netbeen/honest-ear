"""Shared Pydantic schemas for HonestEar pipeline payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TokenScore(BaseModel):
    """Represents a token with its confidence and optional timing."""

    token: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_ms: Optional[int] = Field(default=None, ge=0)
    end_ms: Optional[int] = Field(default=None, ge=0)


class ASRResult(BaseModel):
    """Stores channel transcription output in a normalized format."""

    channel: Literal["faithful", "intended"]
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    tokens: list[TokenScore] = Field(default_factory=list)
    model_name: str


class DiffSpan(BaseModel):
    """Represents a focused phrase-level mismatch between two channels."""

    faithful: str
    intended: str
    start_ms: Optional[int] = Field(default=None, ge=0)
    end_ms: Optional[int] = Field(default=None, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class FusionResult(BaseModel):
    """Stores the structured intermediate result before the LLM call."""

    faithful_text: str
    intended_text: str
    faithful_confidence: float = Field(ge=0.0, le=1.0)
    intended_confidence: float = Field(ge=0.0, le=1.0)
    diff_spans: list[DiffSpan] = Field(default_factory=list)
    should_correct: bool
    gating_reason: str


class CorrectionItem(BaseModel):
    """Represents one LLM-generated correction suggestion."""

    wrong: str
    right: str
    why: str
    confidence: float = Field(ge=0.0, le=1.0)


class CorrectionResponse(BaseModel):
    """Represents the structured response required by the implementation plan."""

    reply: str
    should_show_correction: bool
    corrections: list[CorrectionItem] = Field(default_factory=list)
    faithful_text: str
    intended_text: str
    naturalness_score: int = Field(ge=0, le=100)
    mode: Literal["fluency", "accuracy"]
    meta: dict[str, str] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Represents the end-to-end Phase 1 output for one audio sample."""

    audio_path: Path
    faithful_asr: ASRResult
    intended_asr: ASRResult
    fusion: FusionResult
    llm: CorrectionResponse
    tts_output: Optional[str] = None
    tts_audio_url: Optional[str] = None


class ProcessAudioRequest(BaseModel):
    """Accepts the audio path and execution mode for API and CLI usage."""

    audio_path: Path
    mode: Literal["fluency", "accuracy"] = "accuracy"
    speak_reply: bool = True


class SampleRecord(BaseModel):
    """Represents a labeled evaluation sample for Phase 1 validation."""

    id: str
    accent: str
    topic: str
    audio_file: str
    faithful_reference: str
    intended_reference: str
    expected_error_types: list[str] = Field(default_factory=list)
