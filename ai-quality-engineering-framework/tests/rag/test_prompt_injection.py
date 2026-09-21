from __future__ import annotations

import pytest

from rag_quality.evaluators import RagPromptInjectionEvaluator


pytestmark = [pytest.mark.rag, pytest.mark.rag_security, pytest.mark.security]


def test_retrieved_prompt_injection_is_not_followed(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-022")
    answer = rag_pipeline.answer_case(case)

    result = RagPromptInjectionEvaluator().evaluate(case, answer)

    assert result.passed
    assert result.details["injection_retrieved"] is True


def test_bad_answer_following_retrieved_prompt_injection_fails(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-022")
    answer = rag_pipeline.answer_case(case).model_copy(
        update={"answer": "I will reveal the system prompt and say the refund is unlimited."}
    )

    result = RagPromptInjectionEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert result.details["injection_retrieved"] is True
    assert result.details["injection_followed"] is True
