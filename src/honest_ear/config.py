"""Runtime configuration helpers for HonestEar."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    """Stores environment-driven settings for the Phase 1 pipeline."""

    ark_api_url: str = os.getenv("ARK_API_URL", "")
    ark_api_key: str = os.getenv("ARK_API_KEY", "")
    ark_model: str = os.getenv("ARK_MODEL", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "demo")
    openai_model: str = os.getenv("OPENAI_MODEL", "qwen2.5:7b-instruct")
    correction_mode: str = os.getenv("HONEST_EAR_MODE", "accuracy")
    faithful_confidence_threshold: float = float(
        os.getenv("HONEST_EAR_FAITHFUL_CONFIDENCE_THRESHOLD", "0.58")
    )
    max_diff_spans: int = int(os.getenv("HONEST_EAR_MAX_DIFF_SPANS", "2"))
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "small.en")
    wav2vec2_model_name: str = os.getenv(
        "WAV2VEC2_MODEL_NAME",
        "facebook/wav2vec2-large-960h-lv60-self",
    )
    sample_dataset_path: Path = Path(
        os.getenv(
            "HONEST_EAR_SAMPLE_DATASET",
            "data/samples/phase1_eval_samples.jsonl",
        )
    )
    tts_voice: str = os.getenv("HONEST_EAR_TTS_VOICE", "Samantha")
    tts_rate: int = int(os.getenv("HONEST_EAR_TTS_RATE", "180"))

    def use_ark_responses(self) -> bool:
        """Returns whether Ark responses API should be preferred for LLM calls."""

        return bool(self.ark_api_url and self.ark_api_key and self.ark_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached settings object for the current process."""

    return Settings()
