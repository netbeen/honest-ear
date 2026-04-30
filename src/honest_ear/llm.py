"""LLM client for HonestEar correction decisions."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import httpx
from pydantic import ValidationError
from volcenginesdkarkruntime import Ark  # pyright: ignore[reportMissingImports]

from honest_ear.config import Settings
from honest_ear.schemas import CorrectionResponse, FusionResult


class LLMRequestError(RuntimeError):
    """Raised when the remote LLM request fails or returns invalid data."""


def _explain_diff_reason(reason: str) -> str:
    """Maps one internal diff reason label to a short learner-facing explanation."""

    reason_map = {
        "likely_grammar_inflection": "The grammar form needs to match the subject or tense.",
        "same_head_word_variation": "A small grammar change makes the sentence sound more natural.",
        "phrase_length_mismatch": "This part is clearer when rewritten as a shorter natural phrase.",
        "token_mismatch": "This word choice should be adjusted to match your intended meaning.",
    }
    return reason_map.get(reason, "This part can be said in a more natural grammatical way.")


def _normalize_surface_text(text: str) -> str:
    """Normalizes one short transcript fragment for surface-level comparison."""

    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_surface_only_change(wrong: str, right: str) -> bool:
    """Returns whether two phrases differ only by case or punctuation."""

    return bool(wrong.strip() and right.strip()) and _normalize_surface_text(wrong) == _normalize_surface_text(right)


def _looks_like_spelling_only_change(wrong: str, right: str) -> bool:
    """Returns whether one single-word correction looks like ASR spelling noise rather than grammar."""

    wrong_words = wrong.strip().split()
    right_words = right.strip().split()
    if len(wrong_words) != 1 or len(right_words) != 1:
        return False

    wrong_word = wrong_words[0].lower()
    right_word = right_words[0].lower()
    if wrong_word == right_word:
        return False
    if not wrong_word.isalpha() or not right_word.isalpha():
        return False

    similarity = SequenceMatcher(a=wrong_word, b=right_word).ratio()
    return similarity >= 0.78


def _should_keep_speech_coach_correction(wrong: str, right: str, reason: str | None = None) -> bool:
    """Returns whether one correction is appropriate for a speech-coaching experience."""

    if not wrong.strip() or not right.strip() or wrong.strip() == right.strip():
        return False
    if _is_surface_only_change(wrong, right):
        return False
    if _looks_like_spelling_only_change(wrong, right):
        return False
    if reason in {"token_mismatch", "phrase_length_mismatch"}:
        return False
    return True


def _build_correction_items_from_fusion(fusion: FusionResult) -> list[dict[str, object]]:
    """Builds fallback correction items from fusion diffs when the model omits them."""

    items: list[dict[str, object]] = []
    for span in fusion.diff_spans[:2]:
        wrong = span.faithful.strip()
        right = span.intended.strip()
        if not _should_keep_speech_coach_correction(wrong, right, span.reason):
            continue
        items.append(
            {
                "wrong": wrong,
                "right": right,
                "why": _explain_diff_reason(span.reason),
                "confidence": span.confidence,
            }
        )
    return items


def _filter_model_corrections(response: CorrectionResponse) -> list[dict[str, object]]:
    """Filters model-generated corrections that do not make sense for speech coaching."""

    filtered_items: list[dict[str, object]] = []
    for item in response.corrections:
        if not _should_keep_speech_coach_correction(item.wrong, item.right):
            continue
        filtered_items.append(item.model_dump(mode="json"))
    return filtered_items


def _normalize_correction_response(fusion: FusionResult, response: CorrectionResponse) -> CorrectionResponse:
    """Ensures correction items are present whenever fusion indicates a confident grammar issue."""

    normalized_payload = response.model_dump(mode="json")
    normalized_payload["corrections"] = _filter_model_corrections(response)
    normalized_payload["should_show_correction"] = bool(normalized_payload["corrections"])

    if not fusion.should_correct:
        return CorrectionResponse.model_validate(normalized_payload)
    if normalized_payload["corrections"]:
        return CorrectionResponse.model_validate(normalized_payload)

    generated_corrections = _build_correction_items_from_fusion(fusion)
    if not generated_corrections:
        return CorrectionResponse.model_validate(normalized_payload)

    normalized_payload["should_show_correction"] = True
    normalized_payload["corrections"] = generated_corrections
    normalized_payload["faithful_text"] = fusion.faithful_text
    normalized_payload["intended_text"] = fusion.intended_text
    return CorrectionResponse.model_validate(normalized_payload)


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
            "coach_priority_order": [
                "understand_the_learner_meaning_first",
                "correct_clear_grammar_errors_second",
                "continue_the_conversation_third",
            ],
        },
        "reply_requirements": {
            "tone": [
                "sound like a friendly English speaking coach, not a grammar checker",
                "be encouraging, natural, and conversational",
                "avoid robotic phrases like 'Here is a cleaner way to say it'",
            ],
            "structure": [
                "first show that you understood what the learner meant",
                "if correction is needed, clearly give the corrected sentence in the reply",
                "end with one short follow-up question that invites the learner to continue speaking",
            ],
            "length": "2_to_3_short_sentences",
        },
        "correction_requirements": {
            "when_to_correct": "If fusion_result.should_correct is true and the diff spans are confident, you must correct the grammar.",
            "how_to_correct": [
                "set should_show_correction to true",
                "include 1 to 2 concrete items in corrections",
                "each correction item must explain the learner wording, the corrected wording, and a short reason",
                "the reply should contain one natural corrected sentence before the follow-up question",
            ],
            "do_not_skip": "Do not skip a clear grammar correction just to sound encouraging.",
            "never_correct": [
                "capitalization differences such as english versus English",
                "punctuation differences",
                "ASR spelling-like differences such as hallo versus hello",
                "surface transcript cleanup that is not an audible grammar mistake",
            ],
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
        validated = CorrectionResponse.model_validate(parsed)
        return _normalize_correction_response(fusion, validated)
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
        validated = CorrectionResponse.model_validate(parsed)
        return _normalize_correction_response(fusion, validated)
    except (ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise LLMRequestError(f"LM Studio returned invalid response payload: {exc}") from exc


def request_correction(fusion: FusionResult, mode: str, settings: Settings) -> CorrectionResponse:
    """Calls the configured LLM backend and validates the structured JSON output."""

    if settings.llm_backend == "lm_studio":
        return _request_correction_via_lm_studio(fusion, mode, settings)
    return _request_correction_via_ark_sdk(fusion, mode, settings)
