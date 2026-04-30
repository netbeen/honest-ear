"""LLM client for HonestEar correction decisions."""

from __future__ import annotations

import json
import re

import httpx

from honest_ear.config import Settings
from honest_ear.schemas import CorrectionResponse, FusionResult


def build_correction_prompt(fusion: FusionResult, mode: str) -> str:
    """Builds the structured prompt that constrains the model output."""

    payload = {
        "mode": mode,
        "policy": {
            "audio_access": False,
            "max_corrections": 2,
            "only_use_text_inputs": True,
            "avoid_low_confidence_claims": True,
        },
        "fusion_result": fusion.model_dump(mode="json"),
        "required_json_schema": {
            "reply": "string",
            "should_show_correction": "boolean",
            "corrections": [
                {
                    "wrong": "string",
                    "right": "string",
                    "why": "string",
                    "confidence": "0_to_1_float",
                }
            ],
            "faithful_text": "string",
            "intended_text": "string",
            "naturalness_score": "0_to_100_int",
            "mode": mode,
            "meta": {
                "decision_reason": "string",
            },
        },
    }
    return (
        "You are an English speaking coach. "
        "You only see two transcript channels and confidence metadata, never audio. "
        "Return one valid JSON object only, with no markdown fences and no extra explanation.\n\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def _extract_json_object(raw_text: str) -> dict:
    """Extracts the first JSON object from a model response."""

    stripped = raw_text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise
        return json.loads(match.group(0))


def _build_fallback_response(fusion: FusionResult, mode: str) -> CorrectionResponse:
    """Returns a deterministic response when the remote LLM is unavailable."""

    corrections = []
    for span in fusion.diff_spans[:2]:
        corrections.append(
            {
                "wrong": span.faithful,
                "right": span.intended,
                "why": f"Detected by local diff span: {span.reason}.",
                "confidence": round(span.confidence, 2),
            }
        )

    return CorrectionResponse(
        reply="I understood you. Here is a cleaner way to say it.",
        should_show_correction=fusion.should_correct and bool(corrections),
        corrections=corrections,
        faithful_text=fusion.faithful_text,
        intended_text=fusion.intended_text,
        naturalness_score=80 if fusion.should_correct else 88,
        mode=mode,  # type: ignore[arg-type]
        meta={"decision_reason": "fallback_response"},
    )


def _extract_ark_output_text(response_payload: dict) -> str:
    """Extracts assistant text from Ark responses API payload."""

    output_items = response_payload.get("output", [])
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if content_item.get("type") == "output_text":
                return content_item.get("text", "")
    raise ValueError("Ark response does not contain assistant output_text.")


def _request_correction_via_ark(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls Ark responses API and validates the structured JSON output."""

    prompt = build_correction_prompt(fusion, mode)
    payload = {
        "model": settings.ark_model,
        "reasoning_effort": "minimal",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                settings.ark_api_url,
                headers={
                    "Authorization": f"Bearer {settings.ark_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        raw_content = _extract_ark_output_text(response.json())
        parsed = _extract_json_object(raw_content)
        return CorrectionResponse.model_validate(parsed)
    except Exception:
        return _build_fallback_response(fusion, mode)


def _request_correction_via_openai_compatible(
    fusion: FusionResult, mode: str, settings: Settings
) -> CorrectionResponse:
    """Calls an OpenAI-compatible chat endpoint and validates the JSON output."""

    prompt = build_correction_prompt(fusion, mode)
    payload = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful English speaking coach. Output JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(raw_content)
        return CorrectionResponse.model_validate(parsed)
    except Exception:
        return _build_fallback_response(fusion, mode)


def request_correction(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls the configured LLM backend and validates the structured JSON output."""

    if settings.use_ark_responses():
        return _request_correction_via_ark(fusion, mode, settings)
    return _request_correction_via_openai_compatible(fusion, mode, settings)
