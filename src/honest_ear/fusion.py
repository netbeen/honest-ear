"""Diff and confidence fusion logic for HonestEar Phase 1."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional, Tuple

from honest_ear.config import Settings
from honest_ear.schemas import ASRResult, DiffSpan, FusionResult


def _pick_span_timing(
    faithful: ASRResult, start_index: int, end_index: int
) -> Tuple[Optional[int], Optional[int]]:
    """Maps a faithful-token slice to coarse start and end timestamps."""

    if not faithful.tokens:
        return None, None
    if start_index >= len(faithful.tokens):
        return faithful.tokens[-1].start_ms, faithful.tokens[-1].end_ms

    safe_end_index = min(max(end_index - 1, start_index), len(faithful.tokens) - 1)
    return faithful.tokens[start_index].start_ms, faithful.tokens[safe_end_index].end_ms


def _classify_reason(faithful_phrase: str, intended_phrase: str) -> str:
    """Returns a compact reason label for a candidate correction span."""

    faithful_words = faithful_phrase.split()
    intended_words = intended_phrase.split()
    if len(faithful_words) != len(intended_words):
        return "phrase_length_mismatch"
    if any("'" in word for word in intended_words):
        return "likely_grammar_inflection"
    if faithful_words and intended_words and faithful_words[0] == intended_words[0]:
        return "same_head_word_variation"
    return "token_mismatch"


def _collect_diff_spans(faithful: ASRResult, intended: ASRResult, settings: Settings) -> list[DiffSpan]:
    """Collects phrase-level differences and trims them to the configured maximum."""

    faithful_words = faithful.text.split()
    intended_words = intended.text.split()
    matcher = SequenceMatcher(a=faithful_words, b=intended_words)

    spans: list[DiffSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        faithful_phrase = " ".join(faithful_words[i1:i2]).strip()
        intended_phrase = " ".join(intended_words[j1:j2]).strip()
        if not faithful_phrase and not intended_phrase:
            continue
        start_ms, end_ms = _pick_span_timing(faithful, i1, i2 if i2 > i1 else i1 + 1)
        local_confidence = faithful.confidence * 0.6 + intended.confidence * 0.4
        spans.append(
            DiffSpan(
                faithful=faithful_phrase,
                intended=intended_phrase,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=min(max(local_confidence, 0.0), 1.0),
                reason=_classify_reason(faithful_phrase, intended_phrase),
            )
        )
    spans.sort(key=lambda item: item.confidence, reverse=True)
    return spans[: settings.max_diff_spans]


def fuse_transcripts(faithful: ASRResult, intended: ASRResult, settings: Settings) -> FusionResult:
    """Produces the intermediate JSON required before calling the text LLM."""

    diff_spans = _collect_diff_spans(faithful, intended, settings)

    gating_reason = "stable_diff_detected"
    should_correct = True
    if faithful.confidence < settings.faithful_confidence_threshold:
        gating_reason = "faithful_confidence_below_threshold"
        should_correct = False
    elif not diff_spans:
        gating_reason = "no_phrase_level_diff"
        should_correct = False

    return FusionResult(
        faithful_text=faithful.text,
        intended_text=intended.text,
        faithful_confidence=faithful.confidence,
        intended_confidence=intended.confidence,
        diff_spans=diff_spans,
        should_correct=should_correct,
        gating_reason=gating_reason,
    )
