"""Local TTS helpers for the HonestEar Phase 1 loop."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import uuid
from typing import Optional

from honest_ear.config import Settings


def speak_with_macos_say(text: str, settings: Settings, output_path: Optional[Path] = None) -> Path:
    """Uses macOS `say` and converts the result into one browser-playable wav file."""

    temp_dir = Path(tempfile.gettempdir())
    file_stem = f"honest-ear-reply-{uuid.uuid4().hex[:8]}"
    intermediate_path = temp_dir / f"{file_stem}.aiff"
    target_path = output_path or temp_dir / f"{file_stem}.wav"
    command = [
        "say",
        "-v",
        settings.tts_voice,
        "-r",
        str(settings.tts_rate),
        "-o",
        str(intermediate_path),
        text,
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16",
            str(intermediate_path),
            str(target_path),
        ],
        check=True,
    )
    intermediate_path.unlink(missing_ok=True)
    return target_path
