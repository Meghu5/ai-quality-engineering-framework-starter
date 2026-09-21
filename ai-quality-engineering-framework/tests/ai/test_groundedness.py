from __future__ import annotations

import pytest

from ai_quality.evaluators import GroundednessEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_groundedness_uses_supplied_context(ai_golden_cases, ai_responses):
    evaluator = GroundednessEvaluator()
    context_pairs = [
        (case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
        if case.required_context_terms
    ]
    checks = [evaluator.evaluate(case, response) for case, response in context_pairs]

    assert checks
    assert all(check.passed for check in checks)
