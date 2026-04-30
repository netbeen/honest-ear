"""Runtime configuration helpers for HonestEar."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """Stores environment-driven settings for the Phase 1 pipeline."""

    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "none")
    ark_base_url: str = os.getenv("ARK_BASE_URL", "https://ark-cn-beijing.bytedance.net/api/v3")
    ark_api_key: str = os.getenv("ARK_API_KEY", "")
    ark_model: str = os.getenv("ARK_MODEL", "")
    correction_mode: str = os.getenv("HONEST_EAR_MODE", "accuracy")
    faithful_confidence_threshold: float = float(
        os.getenv("HONEST_EAR_FAITHFUL_CONFIDENCE_THRESHOLD", "0.58")
    )
    max_diff_spans: int = int(os.getenv("HONEST_EAR_MAX_DIFF_SPANS", "2"))
    whisper_model_size: str = os.getenv(
        "WHISPER_MODEL_SIZE",
        str(PROJECT_ROOT / "models/whisper/small.en"),
    )
    wav2vec2_model_name: str = os.getenv(
        "WAV2VEC2_MODEL_NAME",
        str(PROJECT_ROOT / "models/wav2vec2/facebook--wav2vec2-large-960h-lv60-self"),
    )
    sample_dataset_path: Path = Path(
        os.getenv(
            "HONEST_EAR_SAMPLE_DATASET",
            "data/samples/phase1_eval_samples.jsonl",
        )
    )
    skip_asr_warmup: bool = os.getenv("HONEST_EAR_SKIP_ASR_WARMUP", "0") == "1"
    tts_voice: str = os.getenv("HONEST_EAR_TTS_VOICE", "Samantha")
    tts_rate: int = int(os.getenv("HONEST_EAR_TTS_RATE", "180"))

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached settings object for the current process."""

    return Settings()
