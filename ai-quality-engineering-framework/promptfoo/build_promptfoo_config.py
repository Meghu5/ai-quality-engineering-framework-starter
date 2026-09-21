from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_quality.dataset import load_golden_cases


OUTPUT_PATH = Path(__file__).resolve().parent / "generated" / "airline_regression.yaml"


def build_promptfoo_tests(limit: int | None = None) -> list[dict]:
    tests = []
    for case in load_golden_cases()[:limit]:
        assertions = [
            {
                "type": "icontains-any",
                "value": case.reference_answer_terms or [case.expected_intent],
            }
        ]
        for forbidden in case.forbidden_terms:
            assertions.append({"type": "not-icontains", "value": forbidden})
        if case.expect_refusal:
            assertions.append({"type": "icontains", "value": "cannot"})
        tests.append(
            {
                "description": f"{case.id} ({case.category})",
                "vars": {
                    "case_id": case.id,
                    "category": case.category,
                    "user_input": case.user_input,
                    "golden_case": case.model_dump_json(),
                },
                "assert": assertions,
            }
        )
    return tests


def write_promptfoo_tests(path: Path = OUTPUT_PATH) -> list[dict]:
    tests = build_promptfoo_tests()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(tests, sort_keys=False), encoding="utf-8")
    return tests


def main() -> int:
    tests = write_promptfoo_tests()
    print(json.dumps({"path": str(OUTPUT_PATH), "test_count": len(tests)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
