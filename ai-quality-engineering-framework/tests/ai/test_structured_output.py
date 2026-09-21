from __future__ import annotations

import pytest

from ai_quality.evaluators import StructuredOutputEvaluator
from ai_quality.models import AirlineAssistantResponse


pytestmark = [pytest.mark.ai, pytest.mark.llm]


def test_structured_outputs_validate_against_contract(ai_responses):
    evaluator = StructuredOutputEvaluator()
    checks = [evaluator.evaluate(response) for response in ai_responses]

    assert all(check.passed for check in checks)


def test_structured_output_rejects_unexpected_critical_fields(ai_responses):
    payload = ai_responses[0].model_dump(mode="json")
    payload["api_key"] = "synthetic-not-secret"

    with pytest.raises(Exception):
        AirlineAssistantResponse.model_validate(payload)
