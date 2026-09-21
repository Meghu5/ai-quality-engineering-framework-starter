from __future__ import annotations

import pytest

from ai_quality.evaluators import IntentEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_intent_classification_matches_golden_cases(ai_golden_cases, ai_responses):
    evaluator = IntentEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
