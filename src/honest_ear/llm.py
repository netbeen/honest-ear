"""LLM client for HonestEar correction decisions."""

from __future__ import annotations

import json
import re

import httpx
from pydantic import ValidationError

from honest_ear.config import Settings
from honest_ear.schemas import CorrectionResponse, FusionResult


class LLMRequestError(RuntimeError):
    """Raised when the remote LLM request fails or returns invalid data."""


def _build_request_headers(api_key: str) -> dict[str, str]:
    """Builds JSON request headers and only sends auth when explicitly configured."""

    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_correction_prompt(fusion: FusionResult, mode: str) -> str:
    """Builds the structured prompt that constrains the model output."""

    payload = {
        "mode": mode,
        "policy": {
            "audio_access": False,
            "max_corrections": 2,
            "only_use_text_inputs": True,
            "avoid_low_confidence_claims": True,
            "reply_style": "warm_speaking_coach",
            "reply_language": "english",
            "reply_goal": "keep_the_user_talking",
        },
        "reply_requirements": {
            "tone": [
                "sound like a friendly English speaking coach, not a grammar checker",
                "be encouraging, natural, and conversational",
                "avoid robotic phrases like 'Here is a cleaner way to say it'",
            ],
            "structure": [
                "start with a brief encouraging reaction to what the learner said",
                "if correction is needed, naturally model the better sentence inside the reply",
                "end with one short follow-up question that invites the learner to continue speaking",
            ],
            "length": "2_to_3_short_sentences",
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
        "Your most important job is to keep the learner willing to continue speaking. "
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


def _extract_chat_completions_content(response: httpx.Response) -> str:
    """Extracts assistant content from either JSON or SSE chat completions responses."""

    headers = getattr(response, "headers", {}) or {}
    content_type = headers.get("Content-Type", "")
    response_text = getattr(response, "text", "")
    if "text/event-stream" in content_type or response_text.lstrip().startswith("data:"):
        chunks: list[str] = []
        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            payload = json.loads(data)
            for choice in payload.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    chunks.append(content)
        merged = "".join(chunks).strip()
        if not merged:
            raise ValueError("Streaming chat completions response did not contain assistant content.")
        return merged

    return response.json()["choices"][0]["message"]["content"]


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
                headers=_build_request_headers(settings.ark_api_key),
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip()
        raise LLMRequestError(f"Ark request failed with status {exc.response.status_code}: {response_text}") from exc
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"Ark request failed: {exc}") from exc

    try:
        raw_content = _extract_ark_output_text(response.json())
        parsed = _extract_json_object(raw_content)
        return CorrectionResponse.model_validate(parsed)
    except (ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise LLMRequestError(f"Ark returned invalid response payload: {exc}") from exc


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
        "stream": False,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    if settings.llm_reasoning_effort:
        payload["reasoning"] = {"effort": settings.llm_reasoning_effort}

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                settings.get_chat_completions_url(),
                headers=_build_request_headers(settings.openai_api_key),
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip()
        raise LLMRequestError(
            f"OpenAI-compatible request failed with status {exc.response.status_code}: {response_text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"OpenAI-compatible request failed: {exc}") from exc

    try:
        raw_content = _extract_chat_completions_content(response)
        parsed = _extract_json_object(raw_content)
        return CorrectionResponse.model_validate(parsed)
    except (ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise LLMRequestError(f"OpenAI-compatible backend returned invalid response payload: {exc}") from exc


def request_correction(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls the configured LLM backend and validates the structured JSON output."""

    if settings.use_ark_responses():
        return _request_correction_via_ark(fusion, mode, settings)
    return _request_correction_via_openai_compatible(fusion, mode, settings)
