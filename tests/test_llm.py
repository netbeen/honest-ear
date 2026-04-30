"""Tests for LLM backend selection and Ark response parsing."""

from __future__ import annotations

from honest_ear.config import Settings
from honest_ear.llm import _extract_ark_output_text, request_correction
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
        ark_api_url="https://ark-cn-beijing.bytedance.net/api/v3/responses",
        ark_api_key="demo",
        ark_model="ep-demo",
    )
    result = request_correction(_build_fusion_result(), "accuracy", settings)

    assert result.reply == "ark"
