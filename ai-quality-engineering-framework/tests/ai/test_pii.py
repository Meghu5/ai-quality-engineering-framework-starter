from __future__ import annotations

import pytest

from ai_quality.evaluators import PIIEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm, pytest.mark.security]


def test_synthetic_pii_is_detected_and_not_echoed(ai_golden_cases, ai_responses):
    evaluator = PIIEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
