"""Tests for Ark SDK-based LLM requests and response parsing."""

from __future__ import annotations

import json
import pytest

from honest_ear.config import Settings
from honest_ear.llm import (
    LLMRequestError,
    _extract_chat_completion_content,
    _request_correction_via_ark_sdk,
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


def test_extract_chat_completion_content_reads_message_text() -> None:
    """Ensures Ark SDK payloads are parsed from the first choice message content."""

    response = type(
        "FakeResponse",
        (),
        {
            "choices": [
                type(
                    "FakeChoice",
                    (),
                    {
                        "message": type(
                            "FakeMessage",
                            (),
                            {"content": '{"reply":"ok","should_show_correction":false}'},
                        )()
                    },
                )()
            ]
        },
    )()

    result = _extract_chat_completion_content(response)

    assert '"reply":"ok"' in result


def test_request_correction_uses_ark_sdk_backend(monkeypatch) -> None:
    """Ensures request_correction delegates to the Ark SDK backend."""

    def _fake_ark_sdk(fusion: FusionResult, mode: str, _settings: Settings):
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

    monkeypatch.setattr("honest_ear.llm._request_correction_via_ark_sdk", _fake_ark_sdk)

    settings = Settings(
        ark_base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        ark_api_key="demo",
        ark_model="ep-demo",
    )
    result = request_correction(_build_fusion_result(), "accuracy", settings)

    assert result.reply == "ark"


def test_request_correction_propagates_ark_sdk_failure(monkeypatch) -> None:
    """Ensures Ark SDK failures surface as errors instead of fallback replies."""

    def _fake_ark_sdk(*_args, **_kwargs):
        """Raises a deterministic backend error for propagation tests."""

        _ = (_args, _kwargs)
        raise LLMRequestError("Ark SDK request failed: upstream timeout")

    monkeypatch.setattr("honest_ear.llm._request_correction_via_ark_sdk", _fake_ark_sdk)

    settings = Settings(
        ark_base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        ark_api_key="demo",
        ark_model="ep-demo",
    )

    with pytest.raises(LLMRequestError, match="upstream timeout"):
        request_correction(_build_fusion_result(), "accuracy", settings)


def test_ark_sdk_request_sends_reasoning_effort(monkeypatch) -> None:
    """Ensures Ark SDK calls include the configured reasoning effort."""

    captured_request: dict = {}

    class _FakeCompletions:
        """Captures Ark SDK request parameters without making a real API call."""

        def create(self, **kwargs):
            """Stores one outgoing SDK request and returns one fake response."""

            captured_request.update(kwargs)
            return type(
                "FakeResponse",
                (),
                {
                    "choices": [
                        type(
                            "FakeChoice",
                            (),
                            {
                                "message": type(
                                    "FakeMessage",
                                    (),
                                    {
                                        "content": json.dumps(
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
                                        )
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()

    class _FakeChat:
        """Provides the chat namespace expected by the Ark SDK."""

        def __init__(self) -> None:
            """Initializes the nested completions mock."""

            self.completions = _FakeCompletions()

    class _FakeArk:
        """Provides the minimal Ark client surface used by the LLM layer."""

        def __init__(self, **kwargs) -> None:
            """Stores constructor arguments for later assertions."""

            captured_request["client_kwargs"] = kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr("honest_ear.llm.Ark", _FakeArk)

    settings = Settings(
        ark_base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        ark_api_key="demo",
        ark_model="ep-demo",
        llm_reasoning_effort="low",
    )

    result = _request_correction_via_ark_sdk(_build_fusion_result(), "accuracy", settings)

    assert result.reply.startswith("Nice.")
    assert captured_request["client_kwargs"]["base_url"] == "https://ark-cn-beijing.bytedance.net/api/v3"
    assert captured_request["client_kwargs"]["api_key"] == "demo"
    assert captured_request["model"] == "ep-demo"
    assert captured_request["reasoning_effort"] == "low"
    assert captured_request["stream"] is False
    assert captured_request["response_format"] == {"type": "json_object"}

def test_ark_sdk_failure_is_wrapped(monkeypatch) -> None:
    """Ensures SDK exceptions are normalized into one LLMRequestError."""

    class _ExplodingArk:
        """Raises one deterministic exception when the SDK client is used."""

        def __init__(self, **_kwargs) -> None:
            """Accepts Ark constructor arguments."""

        @property
        def chat(self):
            """Raises when the caller tries to access chat completions."""

            raise RuntimeError("network timeout")

    monkeypatch.setattr("honest_ear.llm.Ark", _ExplodingArk)

    settings = Settings(
        ark_base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        ark_api_key="demo",
        ark_model="ep-demo",
    )

    with pytest.raises(LLMRequestError, match="Ark SDK request failed: network timeout"):
        _request_correction_via_ark_sdk(_build_fusion_result(), "accuracy", settings)
