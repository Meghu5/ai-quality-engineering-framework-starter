from __future__ import annotations

import pytest

from ai_quality.evaluators import ToolActionEvaluator


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_tool_actions_match_safe_expected_actions(ai_golden_cases, ai_responses):
    evaluator = ToolActionEvaluator()
    checks = [
        evaluator.evaluate(case, response)
        for case, response in zip(ai_golden_cases, ai_responses)
    ]

    assert all(check.passed for check in checks)
