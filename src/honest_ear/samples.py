"""Sample dataset helpers for Phase 1 evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from honest_ear.schemas import SampleRecord


def load_sample_records(dataset_path: Path) -> list[SampleRecord]:
    """Loads the JSONL Phase 1 evaluation set into validated records."""

    records: list[SampleRecord] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(SampleRecord.model_validate(json.loads(stripped)))
    return records

