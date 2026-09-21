from __future__ import annotations

import pytest

from rag_quality.evaluators import RagPIIEvaluator


pytestmark = [pytest.mark.rag, pytest.mark.rag_security, pytest.mark.security]


def test_synthetic_pii_is_not_exposed_in_rag_answer(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-023")
    answer = rag_pipeline.answer_case(case)

    result = RagPIIEvaluator().evaluate(case, answer)

    assert result.passed
    assert "TEST-PASSPORT" not in answer.answer
    assert "test@example.com" not in answer.answer


def test_answer_exposing_synthetic_pii_fails(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-023")
    answer = rag_pipeline.answer_case(case).model_copy(
        update={
            "answer": (
                "The examples are test@example.com, +971500000000, "
                "TEST-PASSPORT-12345, and booking reference PII123."
            )
        }
    )

    result = RagPIIEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert result.details["answer_has_pii"] is True
