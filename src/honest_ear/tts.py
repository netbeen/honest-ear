"""Local TTS helpers for the HonestEar Phase 1 loop."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Optional

from honest_ear.config import Settings


def speak_with_macos_say(text: str, settings: Settings, output_path: Optional[Path] = None) -> Path:
    """Uses the built-in macOS `say` command for local TTS output."""

    target_path = output_path or Path(tempfile.gettempdir()) / "honest-ear-reply.aiff"
    command = [
        "say",
        "-v",
        settings.tts_voice,
        "-r",
        str(settings.tts_rate),
        "-o",
        str(target_path),
        text,
    ]
    subprocess.run(command, check=True)
    return target_path
