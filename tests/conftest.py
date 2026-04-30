"""Pytest configuration for local package imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
os.environ["HONEST_EAR_SKIP_ASR_WARMUP"] = "1"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
