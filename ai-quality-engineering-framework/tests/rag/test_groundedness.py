from __future__ import annotations

import pytest

from rag_quality.evaluators import RagGroundednessEvaluator
from rag_quality.models import RagAnswer, RagCase, RagChunk, RetrievedChunk


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_grounded_answer_passes(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-002")
    answer = rag_pipeline.answer_case(case)

    result = RagGroundednessEvaluator().evaluate(case, answer)

    assert result.passed


def test_unsupported_statement_fails_groundedness(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-002")
    answer = rag_pipeline.answer_case(case).model_copy(
        update={"answer": "Economy passengers receive 3 checked bags. [Source: AIR-BAG-002#chunk-001]"}
    )

    result = RagGroundednessEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert "3 checked bags" in result.details["unsupported_claims"]


@pytest.mark.parametrize(
    ("context", "answer_text", "expected_unsupported"),
    [
        (
            "Baggage allowance is 7 kg.",
            "Baggage allowance is 25 kg.",
            "25 kg",
        ),
        (
            "Refund processing takes 7 days.",
            "Refund processing takes 7 days and costs AED 500.",
            "AED 500",
        ),
        (
            "Seat selection is available for eligible fares.",
            "All passengers receive free seat selection.",
            "free seat selection",
        ),
    ],
)
def test_unsupported_claims_are_calculated_from_context(context, answer_text, expected_unsupported):
    case = RagCase(
        case_id="grounding-negative",
        question="What does the policy say?",
        expected_document_ids=["DOC-GROUND-001"],
        expected_keywords=["policy"],
        expected_top_k=1,
        category="grounding",
        difficulty="negative",
        expected_answer_summary="Policy answer.",
        forbidden_terms=[expected_unsupported],
    )
    answer = RagAnswer(
        case_id=case.case_id,
        question=case.question,
        answer=answer_text,
        citations=["[Source: DOC-GROUND-001#chunk-001]"],
        retrieved_context=[_retrieved("DOC-GROUND-001", context)],
    )

    result = RagGroundednessEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert expected_unsupported in result.details["unsupported_claims"]


def _retrieved(document_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=RagChunk(
            document_id=document_id,
            chunk_id=f"{document_id}#chunk-001",
            title="Grounding Test",
            category="grounding",
            version=1,
            effective_date="2026-01-01",
            source="unit-test",
            content=content,
            start=0,
            end=len(content),
        ),
        score=1.0,
        rank=1,
        matched_terms=[],
    )
