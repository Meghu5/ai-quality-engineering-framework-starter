from __future__ import annotations

import pytest

from ai_quality.evaluators import RelevanceEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_responses_are_reference_relevant(ai_golden_cases, ai_responses):
    evaluator = RelevanceEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
