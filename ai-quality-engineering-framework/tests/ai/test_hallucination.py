from __future__ import annotations

import pytest

from ai_quality.evaluators import HallucinationEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_hallucination_traps_do_not_include_forbidden_claims(ai_golden_cases, ai_responses):
    evaluator = HallucinationEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
