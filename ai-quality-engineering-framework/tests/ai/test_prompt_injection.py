from __future__ import annotations

import pytest

from ai_quality.evaluators import PromptInjectionEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm, pytest.mark.security]


def test_prompt_injection_cases_are_refused(ai_golden_cases, ai_responses):
    evaluator = PromptInjectionEvaluator()
    injection_pairs = [
        (case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
        if case.category == "prompt_injection"
    ]
    checks = [evaluator.evaluate(case, response) for case, response in injection_pairs]

    assert checks
    assert all(check.passed for check in checks)
