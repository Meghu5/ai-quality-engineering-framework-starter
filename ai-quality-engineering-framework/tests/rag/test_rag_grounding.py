from __future__ import annotations

import pytest

from rag_quality.evaluators import RagEvaluator


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_answer_is_grounded_for_refund_policy(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-006")
    answer = rag_pipeline.answer_case(case)

    evaluation = RagEvaluator().evaluate(case, answer)

    assert evaluation.groundedness.passed
    assert evaluation.citations.passed
    assert evaluation.overall_passed
