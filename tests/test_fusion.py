"""Focused tests for the Phase 1 fusion layer."""

from __future__ import annotations

from pathlib import Path

from honest_ear.config import Settings
from honest_ear.fusion import fuse_transcripts
from honest_ear.samples import load_sample_records
from honest_ear.schemas import ASRResult, TokenScore


def _build_asr_result(channel: str, text: str, confidence: float) -> ASRResult:
    """Builds compact ASR fixtures for fusion tests."""

    tokens = [
        TokenScore(
            token=token,
            confidence=confidence,
            start_ms=index * 100,
            end_ms=(index + 1) * 100,
        )
        for index, token in enumerate(text.split())
    ]
    return ASRResult(
        channel=channel,  # type: ignore[arg-type]
        text=text,
        confidence=confidence,
        tokens=tokens,
        model_name="fixture",
    )


def test_fusion_blocks_corrections_on_low_faithful_confidence() -> None:
    """Ensures weak faithful confidence disables explicit correction output."""

    settings = Settings(faithful_confidence_threshold=0.7, max_diff_spans=2)
    faithful = _build_asr_result("faithful", "he dont like coffee", 0.42)
    intended = _build_asr_result("intended", "he doesn't like coffee", 0.88)

    result = fuse_transcripts(faithful, intended, settings)

    assert result.should_correct is False
    assert result.gating_reason == "faithful_confidence_below_threshold"
    assert len(result.diff_spans) == 1


def test_fusion_limits_diff_spans() -> None:
    """Ensures the local fusion layer only forwards the top candidate spans."""

    settings = Settings(faithful_confidence_threshold=0.5, max_diff_spans=2)
    faithful = _build_asr_result("faithful", "he dont like the coffee what you make", 0.91)
    intended = _build_asr_result("intended", "he doesn't like the coffee you made", 0.95)

    result = fuse_transcripts(faithful, intended, settings)

    assert result.should_correct is True
    assert len(result.diff_spans) <= 2


def test_sample_dataset_contains_phase1_coverage() -> None:
    """Ensures the bundled sample set meets the target size for Phase 1."""

    dataset_path = Path(__file__).resolve().parent.parent / "data" / "samples" / "phase1_eval_samples.jsonl"
    records = load_sample_records(dataset_path)

    assert len(records) == 30
    assert any(record.accent == "Cantonese" for record in records)
    assert any("third_person_singular" in record.expected_error_types for record in records)

