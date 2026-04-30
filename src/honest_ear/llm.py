"""LLM client for HonestEar correction decisions."""

from __future__ import annotations

import json
import re

import httpx
from pydantic import ValidationError
from volcenginesdkarkruntime import Ark  # pyright: ignore[reportMissingImports]

from honest_ear.config import Settings
from honest_ear.schemas import CorrectionResponse, FusionResult


class LLMRequestError(RuntimeError):
    """Raised when the remote LLM request fails or returns invalid data."""


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


def _build_ark_client(settings: Settings) -> Ark:
    """Builds one Ark SDK client using the configured base URL and API key."""

    return Ark(
        base_url=settings.ark_base_url,
        api_key=settings.ark_api_key,
    )


def _extract_chat_completion_content(response) -> str:
    """Extracts assistant content from one Ark SDK chat completion response."""

    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Ark SDK response does not contain choices.")
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError("Ark SDK response choice does not contain message.")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ark SDK response message does not contain text content.")
    return content


def _build_request_headers(api_key: str) -> dict[str, str]:
    """Builds JSON request headers and only sends auth when explicitly configured."""

    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_openai_chat_content(response: httpx.Response) -> str:
    """Extracts assistant content from one OpenAI-compatible JSON response."""

    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("LM Studio response does not contain choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LM Studio response does not contain assistant text content.")
    return content


def _request_correction_via_ark_sdk(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls Ark chat completions via the official Python SDK and validates the JSON output."""

    prompt = build_correction_prompt(fusion, mode)

    try:
        client = _build_ark_client(settings)
        response = client.chat.completions.create(
            model=settings.ark_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful English speaking coach. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            reasoning_effort=settings.llm_reasoning_effort,
            temperature=0.2,
            stream=False,
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
    except Exception as exc:
        raise LLMRequestError(f"Ark SDK request failed: {exc}") from exc

    try:
        raw_content = _extract_chat_completion_content(response)
        parsed = _extract_json_object(raw_content)
        return CorrectionResponse.model_validate(parsed)
    except (ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise LLMRequestError(f"Ark SDK returned invalid response payload: {exc}") from exc


def _request_correction_via_lm_studio(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls one LM Studio OpenAI-compatible endpoint and validates the JSON output."""

    prompt = build_correction_prompt(fusion, mode)
    payload = {
        "model": settings.lm_studio_model,
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
    }

    try:
        response = httpx.post(
            f"{settings.lm_studio_base_url.rstrip('/')}/chat/completions",
            headers=_build_request_headers(settings.lm_studio_api_key),
            json=payload,
            timeout=180.0,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text.strip()
        raise LLMRequestError(
            f"LM Studio request failed with status {exc.response.status_code}: {response_text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"LM Studio request failed: {exc}") from exc

    try:
        raw_content = _extract_openai_chat_content(response)
        parsed = _extract_json_object(raw_content)
        return CorrectionResponse.model_validate(parsed)
    except (ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise LLMRequestError(f"LM Studio returned invalid response payload: {exc}") from exc


def request_correction(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls the configured LLM backend and validates the structured JSON output."""

    if settings.llm_backend == "lm_studio":
        return _request_correction_via_lm_studio(fusion, mode, settings)
    return _request_correction_via_ark_sdk(fusion, mode, settings)
