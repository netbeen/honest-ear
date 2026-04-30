"""Local ASR providers used by the HonestEar Phase 1 pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from pathlib import Path
import math

from honest_ear.config import Settings
from honest_ear.schemas import ASRResult, TokenScore


def _suggest_download_command(model_ref: str, provider_name: str) -> str:
    """Builds one shell command suggestion for downloading the missing local model."""

    model_dir = Path(model_ref).expanduser()
    model_name = model_dir.name
    if provider_name == "Whisper":
        return f"./scripts/download-asr-models.sh --whisper-only --whisper-model {model_name}"

    inferred_model_name = model_name.replace("--", "/")
    return (
        "./scripts/download-asr-models.sh "
        f"--wav2vec2-only --wav2vec2-model {inferred_model_name}"
    )


def _require_local_model_dir(model_ref: str, provider_name: str) -> Path:
    """Validates that one ASR model is pre-downloaded into a local directory."""

    model_dir = Path(model_ref).expanduser()
    if not model_dir.is_absolute():
        model_dir = model_dir.resolve()

    if not model_dir.exists() or not model_dir.is_dir():
        suggested_command = _suggest_download_command(model_ref, provider_name)
        raise FileNotFoundError(
            f"{provider_name} model directory not found: {model_dir}. "
            "Please pre-download the model before starting the app. "
            f"Suggested command: {suggested_command}"
        )

    return model_dir


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
    def warmup(self) -> None:
        """Loads model assets eagerly so the first request avoids cold starts."""

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

            model_dir = _require_local_model_dir(self._settings.whisper_model_size, "Whisper")
            self._model = WhisperModel(str(model_dir), device="auto", compute_type="auto")
        return self._model

    def warmup(self) -> None:
        """Preloads the whisper model during application startup."""

        self._get_model()

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

            model_dir = _require_local_model_dir(self._settings.wav2vec2_model_name, "wav2vec2")
            self._processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
            self._model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True)

    def warmup(self) -> None:
        """Preloads the wav2vec2 processor and model during application startup."""

        self._ensure_loaded()

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


_PROVIDERS_LOCK = Lock()
_CACHED_PROVIDER_KEY: tuple[str, str] | None = None
_CACHED_PROVIDERS: tuple[Wav2Vec2FaithfulProvider, WhisperASRProvider] | None = None


def build_asr_providers(settings: Settings) -> tuple[Wav2Vec2FaithfulProvider, WhisperASRProvider]:
    """Builds or reuses singleton ASR providers for the active model configuration."""

    global _CACHED_PROVIDER_KEY, _CACHED_PROVIDERS

    provider_key = (settings.wav2vec2_model_name, settings.whisper_model_size)
    with _PROVIDERS_LOCK:
        if _CACHED_PROVIDERS is None or _CACHED_PROVIDER_KEY != provider_key:
            _CACHED_PROVIDER_KEY = provider_key
            _CACHED_PROVIDERS = (
                Wav2Vec2FaithfulProvider(settings),
                WhisperASRProvider(settings),
            )
        return _CACHED_PROVIDERS


def warmup_asr_models(settings: Settings) -> None:
    """Loads both ASR models before the application starts serving requests."""

    faithful_provider, intended_provider = build_asr_providers(settings)
    faithful_provider.warmup()
    intended_provider.warmup()
