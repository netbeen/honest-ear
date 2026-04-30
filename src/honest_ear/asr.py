"""Local ASR providers used by the HonestEar Phase 1 pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import math

from honest_ear.config import Settings
from honest_ear.schemas import ASRResult, TokenScore


@dataclass
class AudioMetadata:
    """Stores basic metadata required for confidence and timing heuristics."""

    duration_ms: int
    sample_rate: int


def read_audio_metadata(audio_path: Path) -> AudioMetadata:
    """Reads lightweight metadata without keeping the full waveform in memory."""

    import soundfile as sf

    info = sf.info(str(audio_path))
    duration_ms = math.ceil((info.frames / info.samplerate) * 1000)
    return AudioMetadata(duration_ms=duration_ms, sample_rate=info.samplerate)


def _split_tokens_with_even_timings(text: str, duration_ms: int, confidence: float) -> list[TokenScore]:
    """Builds coarse token timings when the source model does not expose word timestamps."""

    tokens = [token for token in text.strip().split() if token]
    if not tokens:
        return []

    slot = max(duration_ms // len(tokens), 1)
    results: list[TokenScore] = []
    for index, token in enumerate(tokens):
        start_ms = index * slot
        end_ms = duration_ms if index == len(tokens) - 1 else min(duration_ms, (index + 1) * slot)
        results.append(
            TokenScore(
                token=token,
                confidence=confidence,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return results


class BaseASRProvider(ABC):
    """Defines the contract for one local transcription channel."""

    channel: str

    @abstractmethod
    def transcribe(self, audio_path: Path) -> ASRResult:
        """Transcribes one audio file and returns a normalized result."""


class WhisperASRProvider(BaseASRProvider):
    """Runs the intended-text channel with faster-whisper."""

    channel = "intended"

    def __init__(self, settings: Settings) -> None:
        """Stores model config and delays heavy imports until first use."""

        self._settings = settings
        self._model = None

    def _get_model(self):
        """Loads the faster-whisper model lazily to keep startup light."""

        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._settings.whisper_model_size, device="auto", compute_type="auto")
        return self._model

    def transcribe(self, audio_path: Path) -> ASRResult:
        """Returns the intended-text transcript and per-word confidence when available."""

        metadata = read_audio_metadata(audio_path)
        model = self._get_model()
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=None,
        )

        texts: list[str] = []
        confidences: list[float] = []
        tokens: list[TokenScore] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
            segment_confidence = min(max(math.exp(segment.avg_logprob), 0.0), 1.0)
            confidences.append(segment_confidence)
            for word in getattr(segment, "words", []) or []:
                word_text = (word.word or "").strip()
                if not word_text:
                    continue
                tokens.append(
                    TokenScore(
                        token=word_text,
                        confidence=min(max(float(getattr(word, "probability", segment_confidence)), 0.0), 1.0),
                        start_ms=int(word.start * 1000),
                        end_ms=int(word.end * 1000),
                    )
                )

        transcript = " ".join(texts).strip()
        overall_confidence = sum(confidences) / len(confidences) if confidences else min(
            max(float(getattr(info, "language_probability", 0.7)), 0.0), 1.0
        )
        if not tokens:
            tokens = _split_tokens_with_even_timings(transcript, metadata.duration_ms, overall_confidence)

        return ASRResult(
            channel="intended",
            text=transcript,
            confidence=overall_confidence,
            tokens=tokens,
            model_name=f"faster-whisper:{self._settings.whisper_model_size}",
        )


class Wav2Vec2FaithfulProvider(BaseASRProvider):
    """Runs the faithful-text channel with raw CTC decoding and no LM."""

    channel = "faithful"

    def __init__(self, settings: Settings) -> None:
        """Stores model config and delays heavy imports until first use."""

        self._settings = settings
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Loads the Hugging Face processor and CTC model lazily."""

        if self._processor is None or self._model is None:
            from transformers import AutoModelForCTC, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._settings.wav2vec2_model_name)
            self._model = AutoModelForCTC.from_pretrained(self._settings.wav2vec2_model_name)

    def transcribe(self, audio_path: Path) -> ASRResult:
        """Returns the faithful-text transcript with coarse confidence heuristics."""

        import numpy as np
        import soundfile as sf
        import torch

        self._ensure_loaded()
        metadata = read_audio_metadata(audio_path)

        waveform, sample_rate = sf.read(str(audio_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError("wav2vec2 baseline currently expects 16kHz mono wav input.")

        inputs = self._processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        frame_scores = probabilities.max(dim=-1).values[0].detach().cpu().numpy()
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self._processor.batch_decode(predicted_ids)[0].strip().lower()
        confidence = float(np.clip(frame_scores.mean() if frame_scores.size else 0.0, 0.0, 1.0))
        tokens = _split_tokens_with_even_timings(transcript, metadata.duration_ms, confidence)

        return ASRResult(
            channel="faithful",
            text=transcript,
            confidence=confidence,
            tokens=tokens,
            model_name=f"transformers:{self._settings.wav2vec2_model_name}",
        )


def build_asr_providers(settings: Settings) -> tuple[Wav2Vec2FaithfulProvider, WhisperASRProvider]:
    """Builds the two local ASR channels required by Phase 1."""

    return Wav2Vec2FaithfulProvider(settings), WhisperASRProvider(settings)
