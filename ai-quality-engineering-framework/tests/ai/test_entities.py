from __future__ import annotations

import pytest

from ai_quality.evaluators import EntityEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_required_entities_match_golden_cases(ai_golden_cases, ai_responses):
    evaluator = EntityEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
