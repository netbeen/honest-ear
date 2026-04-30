"""Tests for LLM backend selection and Ark response parsing."""

from __future__ import annotations

import json
import pytest

from honest_ear.config import Settings
from honest_ear.llm import (
    LLMRequestError,
    _extract_chat_completions_content,
    _extract_ark_output_text,
    _request_correction_via_ark,
    _request_correction_via_openai_compatible,
    request_correction,
)
from honest_ear.schemas import CorrectionResponse, DiffSpan, FusionResult


def _build_fusion_result() -> FusionResult:
    """Builds a deterministic fusion result fixture for LLM tests."""

    return FusionResult(
        faithful_text="he dont like the coffee what you make",
        intended_text="he doesn't like the coffee you made",
        faithful_confidence=0.82,
        intended_confidence=0.95,
        diff_spans=[
            DiffSpan(
                faithful="dont",
                intended="doesn't",
                start_ms=100,
                end_ms=180,
                confidence=0.87,
                reason="likely_grammar_inflection",
            )
        ],
        should_correct=True,
        gating_reason="stable_diff_detected",
    )


def test_extract_ark_output_text_reads_message_output() -> None:
    """Ensures Ark responses payload is parsed from output message blocks."""

    payload = {
        "output": [
            {"type": "reasoning", "content": []},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"reply":"ok","should_show_correction":false,"corrections":[],"faithful_text":"a","intended_text":"b","naturalness_score":90,"mode":"accuracy","meta":{"decision_reason":"done"}}',
                    }
                ],
            },
        ]
    }

    result = _extract_ark_output_text(payload)

    assert '"reply":"ok"' in result


def test_extract_chat_completions_content_supports_sse_chunks() -> None:
    """Ensures SSE chat completions responses are merged into one assistant message."""

    response = type(
        "FakeResponse",
        (),
        {
            "headers": {"Content-Type": "text/event-stream"},
            "text": (
                'data: {"choices":[{"delta":{"content":"{\\"reply\\":\\"Hi\\""}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"}"}}]}\n\n'
                "data: [DONE]\n"
            ),
        },
    )()

    result = _extract_chat_completions_content(response)

    assert result == '{"reply":"Hi"}'


def test_request_correction_prefers_ark_when_configured(monkeypatch) -> None:
    """Ensures Ark responses backend is selected when Ark config is present."""

    def _fake_ark(fusion: FusionResult, mode: str, _settings: Settings):
        """Returns a deterministic correction response for backend selection tests."""

        _ = _settings
        return CorrectionResponse(
            reply="ark",
            should_show_correction=True,
            corrections=[],
            faithful_text=fusion.faithful_text,
            intended_text=fusion.intended_text,
            naturalness_score=88,
            mode=mode,
            meta={"decision_reason": "ark"},
        )

    def _fake_openai(*_args, **_kwargs):
        """Fails fast if backend routing unexpectedly falls through to OpenAI mode."""

        _ = (_args, _kwargs)
        raise AssertionError("should not call openai-compatible backend")

    monkeypatch.setattr("honest_ear.llm._request_correction_via_ark", _fake_ark)
    monkeypatch.setattr("honest_ear.llm._request_correction_via_openai_compatible", _fake_openai)

    settings = Settings(
        llm_api_style="ark_responses",
        ark_api_url="https://ark-cn-beijing.bytedance.net/api/v3/responses",
        ark_api_key="demo",
        ark_model="ep-demo",
    )
    result = request_correction(_build_fusion_result(), "accuracy", settings)

    assert result.reply == "ark"


def test_request_correction_propagates_ark_failure(monkeypatch) -> None:
    """Ensures Ark failures surface as errors instead of fallback replies."""

    def _fake_ark(*_args, **_kwargs):
        """Raises a deterministic backend error for propagation tests."""

        _ = (_args, _kwargs)
        raise LLMRequestError("Ark request failed with status 500: upstream timeout")

    monkeypatch.setattr("honest_ear.llm._request_correction_via_ark", _fake_ark)

    settings = Settings(
        llm_api_style="ark_responses",
        ark_api_url="https://ark-cn-beijing.bytedance.net/api/v3/responses",
        ark_api_key="demo",
        ark_model="ep-demo",
    )

    with pytest.raises(LLMRequestError, match="upstream timeout"):
        request_correction(_build_fusion_result(), "accuracy", settings)


def test_ark_request_payload_does_not_send_reasoning_effort(monkeypatch) -> None:
    """Ensures the current Ark responses API payload omits unsupported fields."""

    captured_payload: dict = {}

    class _FakeResponse:
        """Provides the minimal response surface used by the Ark client."""

        def raise_for_status(self) -> None:
            """Simulates one successful HTTP response."""

        def json(self) -> dict:
            """Returns one valid Ark-style JSON payload."""

            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "reply": "Nice. What fruit do you like most?",
                                        "should_show_correction": True,
                                        "corrections": [],
                                        "faithful_text": "a",
                                        "intended_text": "b",
                                        "naturalness_score": 90,
                                        "mode": "accuracy",
                                        "meta": {"decision_reason": "ok"},
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }

    class _FakeClient:
        """Captures the outgoing request payload without making one real HTTP call."""

        def __init__(self, *args, **kwargs) -> None:
            """Accepts the same constructor shape as httpx.Client."""

            _ = (args, kwargs)

        def __enter__(self):
            """Supports use as one context manager."""

            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            """Closes the fake client without suppressing errors."""

            _ = (exc_type, exc, tb)
            return False

        def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
            """Stores the JSON body so the test can assert on it."""

            _ = (url, headers)
            captured_payload.update(json)
            return _FakeResponse()

    monkeypatch.setattr("honest_ear.llm.httpx.Client", _FakeClient)

    settings = Settings(
        ark_api_url="https://ark-cn-beijing.bytedance.net/api/v3/responses",
        ark_api_key="demo",
        ark_model="ep-demo",
    )

    result = _request_correction_via_ark(_build_fusion_result(), "accuracy", settings)

    assert result.reply.startswith("Nice.")
    assert "reasoning_effort" not in captured_payload


def test_chat_completions_payload_uses_reasoning_effort_and_direct_url(monkeypatch) -> None:
    """Ensures chat completions requests send reasoning.effort to the configured full URL."""

    captured_request: dict = {}

    class _FakeResponse:
        """Provides the minimal response surface used by the chat completions client."""

        def raise_for_status(self) -> None:
            """Simulates one successful HTTP response."""

        def json(self) -> dict:
            """Returns one valid OpenAI-compatible JSON payload."""

            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "reply": "Nice job. What do you want to eat next?",
                                    "should_show_correction": False,
                                    "corrections": [],
                                    "faithful_text": "a",
                                    "intended_text": "b",
                                    "naturalness_score": 90,
                                    "mode": "accuracy",
                                    "meta": {"decision_reason": "ok"},
                                }
                            )
                        }
                    }
                ]
            }

    class _FakeClient:
        """Captures the outgoing request without making one real HTTP call."""

        def __init__(self, *args, **kwargs) -> None:
            """Accepts the same constructor shape as httpx.Client."""

            _ = (args, kwargs)

        def __enter__(self):
            """Supports use as one context manager."""

            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            """Closes the fake client without suppressing errors."""

            _ = (exc_type, exc, tb)
            return False

        def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
            """Stores the request parameters so the test can assert on them."""

            captured_request["url"] = url
            captured_request["headers"] = headers
            captured_request["json"] = json
            return _FakeResponse()

    monkeypatch.setattr("honest_ear.llm.httpx.Client", _FakeClient)

    settings = Settings(
        llm_api_style="chat_completions",
        llm_reasoning_effort="none",
        openai_chat_completions_url="https://example.internal/api/v3/bots/chat/completions",
        openai_api_key="",
        openai_model="test-calculator",
    )

    result = _request_correction_via_openai_compatible(_build_fusion_result(), "accuracy", settings)

    assert result.reply.startswith("Nice job.")
    assert captured_request["url"] == "https://example.internal/api/v3/bots/chat/completions"
    assert "Authorization" not in captured_request["headers"]
    assert captured_request["json"]["reasoning"] == {"effort": "none"}
    assert captured_request["json"]["stream"] is False
