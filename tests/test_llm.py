"""Tests for Ark SDK and LM Studio based LLM requests and response parsing."""

from __future__ import annotations

import json
import pytest

from honest_ear.config import Settings
from honest_ear.llm import (
    LLMRequestError,
    _extract_chat_completion_content,
    _normalize_correction_response,
    _request_correction_via_ark_sdk,
    _request_correction_via_lm_studio,
    request_correction,
)
from honest_ear.schemas import CorrectionItem, CorrectionResponse, DiffSpan, FusionResult


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
        llm_backend="ark_sdk",
        ark_base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        ark_api_key="demo",
        ark_model="ep-demo",
    )
    result = request_correction(_build_fusion_result(), "accuracy", settings)

    assert result.reply == "ark"


def test_request_correction_uses_lm_studio_backend(monkeypatch) -> None:
    """Ensures request_correction delegates to LM Studio when configured."""

    def _fake_lm_studio(fusion: FusionResult, mode: str, _settings: Settings):
        """Returns a deterministic correction response for backend selection tests."""

        _ = _settings
        return CorrectionResponse(
            reply="lm_studio",
            should_show_correction=True,
            corrections=[],
            faithful_text=fusion.faithful_text,
            intended_text=fusion.intended_text,
            naturalness_score=86,
            mode=mode,
            meta={"decision_reason": "lm_studio"},
        )

    monkeypatch.setattr("honest_ear.llm._request_correction_via_lm_studio", _fake_lm_studio)

    settings = Settings(
        llm_backend="lm_studio",
        lm_studio_base_url="http://127.0.0.1:1234/v1",
        lm_studio_model="qwen/qwen3.5-35b-a3b",
    )
    result = request_correction(_build_fusion_result(), "accuracy", settings)

    assert result.reply == "lm_studio"


def test_request_correction_propagates_ark_sdk_failure(monkeypatch) -> None:
    """Ensures Ark SDK failures surface as errors instead of fallback replies."""

    def _fake_ark_sdk(*_args, **_kwargs):
        """Raises a deterministic backend error for propagation tests."""

        _ = (_args, _kwargs)
        raise LLMRequestError("Ark SDK request failed: upstream timeout")

    monkeypatch.setattr("honest_ear.llm._request_correction_via_ark_sdk", _fake_ark_sdk)

    settings = Settings(
        llm_backend="ark_sdk",
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
    assert result.should_show_correction is True
    assert result.corrections[0].wrong == "dont"
    assert result.corrections[0].right == "doesn't"
    assert captured_request["client_kwargs"]["base_url"] == "https://ark-cn-beijing.bytedance.net/api/v3"
    assert captured_request["client_kwargs"]["api_key"] == "demo"
    assert captured_request["model"] == "ep-demo"
    assert captured_request["reasoning_effort"] == "low"
    assert captured_request["stream"] is False
    assert captured_request["response_format"] == {"type": "json_object"}


def test_lm_studio_request_omits_json_object_response_format(monkeypatch) -> None:
    """Ensures LM Studio uses a compatible OpenAI request shape without json_object mode."""

    captured_request: dict = {}

    class _FakeResponse:
        """Provides the minimal LM Studio response surface used by the client."""

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

    def _fake_post(url: str, headers: dict, json: dict, timeout: float) -> _FakeResponse:
        """Captures one outgoing LM Studio request without making a real HTTP call."""

        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["json"] = json
        captured_request["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("honest_ear.llm.httpx.post", _fake_post)

    settings = Settings(
        llm_backend="lm_studio",
        lm_studio_base_url="http://127.0.0.1:1234/v1",
        lm_studio_api_key="",
        lm_studio_model="qwen/qwen3.5-35b-a3b",
    )

    result = _request_correction_via_lm_studio(_build_fusion_result(), "accuracy", settings)

    assert result.reply.startswith("Nice job.")
    assert result.should_show_correction is True
    assert result.corrections[0].wrong == "dont"
    assert result.corrections[0].right == "doesn't"
    assert captured_request["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert "Authorization" not in captured_request["headers"]
    assert captured_request["json"]["stream"] is False
    assert "response_format" not in captured_request["json"]
    assert captured_request["json"]["model"] == "qwen/qwen3.5-35b-a3b"


def test_ark_sdk_failure_is_wrapped(monkeypatch) -> None:
    """Ensures SDK exceptions are normalized into one LLMRequestError."""

    class _ExplodingArk:
        """Raises one deterministic exception when the SDK client is used."""

        def __init__(self, **_kwargs) -> None:
            """Accepts Ark constructor arguments."""

            _ = _kwargs

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


def test_lm_studio_failure_is_wrapped(monkeypatch) -> None:
    """Ensures LM Studio HTTP exceptions are normalized into one LLMRequestError."""

    def _fake_post(*_args, **_kwargs):
        """Raises one deterministic HTTP exception for wrapping tests."""

        _ = (_args, _kwargs)
        raise httpx.ConnectError("connection refused")

    import httpx

    monkeypatch.setattr("honest_ear.llm.httpx.post", _fake_post)

    settings = Settings(
        llm_backend="lm_studio",
        lm_studio_base_url="http://127.0.0.1:1234/v1",
        lm_studio_model="qwen/qwen3.5-35b-a3b",
    )

    with pytest.raises(LLMRequestError, match="LM Studio request failed"):
        _request_correction_via_lm_studio(_build_fusion_result(), "accuracy", settings)


def test_normalize_correction_response_filters_surface_only_items() -> None:
    """Ensures capitalization or spelling-like cleanup is not shown as learner correction."""

    fusion = FusionResult(
        faithful_text="hallo hallo can you teach my english",
        intended_text="hello hello can you teach my English",
        faithful_confidence=0.99,
        intended_confidence=0.74,
        diff_spans=[
            DiffSpan(
                faithful="hallo hallo",
                intended="hello hello",
                start_ms=0,
                end_ms=120,
                confidence=0.8,
                reason="token_mismatch",
            ),
            DiffSpan(
                faithful="english",
                intended="English",
                start_ms=121,
                end_ms=180,
                confidence=0.9,
                reason="token_mismatch",
            ),
        ],
        should_correct=True,
        gating_reason="stable_diff_detected",
    )
    response = CorrectionResponse(
        reply="Hello! I'd be happy to help you with your English.",
        should_show_correction=True,
        corrections=[
            CorrectionItem(
                wrong="english",
                right="English",
                why="Names of languages are proper nouns.",
                confidence=0.8,
            ),
            CorrectionItem(
                wrong="hallo",
                right="hello",
                why="Hello is the standard spelling.",
                confidence=0.7,
            ),
        ],
        faithful_text=fusion.faithful_text,
        intended_text=fusion.intended_text,
        naturalness_score=80,
        mode="accuracy",
        meta={"decision_reason": "surface_only"},
    )

    normalized = _normalize_correction_response(fusion, response)

    assert normalized.should_show_correction is False
    assert normalized.corrections == []


def test_normalize_correction_response_keeps_grammar_items() -> None:
    """Ensures real grammar fixes (e.g. dont->doesn't) are preserved for the learner."""

    fusion = _build_fusion_result()
    response = CorrectionResponse(
        reply="Nice. What fruit do you like most?",
        should_show_correction=True,
        corrections=[
            CorrectionItem(
                wrong="dont",
                right="doesn't",
                why="Use doesn't for he/she/it in the present simple.",
                confidence=0.9,
            )
        ],
        faithful_text=fusion.faithful_text,
        intended_text=fusion.intended_text,
        naturalness_score=90,
        mode="accuracy",
        meta={"decision_reason": "grammar"},
    )

    normalized = _normalize_correction_response(fusion, response)

    assert normalized.should_show_correction is True
    assert normalized.corrections[0].wrong == "dont"
    assert normalized.corrections[0].right == "doesn't"
