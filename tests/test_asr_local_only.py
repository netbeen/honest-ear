"""Tests for strict local-only ASR model loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from honest_ear.asr import _require_local_model_dir
from honest_ear.config import Settings


def test_require_local_model_dir_accepts_existing_directory(tmp_path: Path) -> None:
    """Accepts one pre-downloaded local model directory."""

    model_dir = tmp_path / "models" / "whisper" / "small.en"
    model_dir.mkdir(parents=True)

    resolved = _require_local_model_dir(str(model_dir), "Whisper")

    assert resolved == model_dir


def test_require_local_model_dir_rejects_missing_directory(tmp_path: Path) -> None:
    """Rejects startup when one local model directory is missing."""

    missing_dir = tmp_path / "models" / "wav2vec2" / "missing-model"

    with pytest.raises(FileNotFoundError, match="Suggested command:"):
        _require_local_model_dir(str(missing_dir), "wav2vec2")


def test_require_local_model_dir_includes_missing_path_and_command(tmp_path: Path) -> None:
    """Includes the missing directory and one actionable download command."""

    missing_dir = tmp_path / "models" / "whisper" / "medium.en"

    with pytest.raises(FileNotFoundError) as exc_info:
        _require_local_model_dir(str(missing_dir), "Whisper")

    message = str(exc_info.value)
    assert str(missing_dir) in message
    assert "./scripts/download-asr-models.sh --whisper-only --whisper-model medium.en" in message


def test_settings_default_to_project_local_model_paths() -> None:
    """Uses project-local directories as the default ASR model references."""

    settings = Settings()

    assert settings.whisper_model_size.endswith("models/whisper/small.en")
    assert settings.wav2vec2_model_name.endswith(
        "models/wav2vec2/facebook--wav2vec2-large-960h-lv60-self"
    )
