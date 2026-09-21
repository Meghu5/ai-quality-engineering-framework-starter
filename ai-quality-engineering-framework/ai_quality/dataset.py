from __future__ import annotations

import json
from pathlib import Path

from ai_quality.models import GoldenCase


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "ai" / "golden_cases.json"


def load_golden_cases(path: Path = DATASET_PATH) -> list[GoldenCase]:
    with path.open(encoding="utf-8") as file:
        raw_cases = json.load(file)
    return [GoldenCase.model_validate(case) for case in raw_cases]
