from __future__ import annotations

import pytest

from ai_quality.evaluators import SafetyEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm, pytest.mark.security]


def test_safety_sensitive_cases_use_expected_refusal(ai_golden_cases, ai_responses):
    evaluator = SafetyEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
