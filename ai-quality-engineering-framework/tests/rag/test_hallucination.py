from __future__ import annotations

import pytest

from rag_quality.evaluators import RagHallucinationEvaluator
from rag_quality.models import RagAnswer, RagCase, RagChunk, RetrievedChunk


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_hallucination_trap_passes_when_forbidden_terms_absent(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-030")
    answer = rag_pipeline.answer_case(case)

    assert RagHallucinationEvaluator().evaluate(case, answer).passed


def test_invented_refund_guarantee_fails(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-030")
    answer = rag_pipeline.answer_case(case).model_copy(
        update={"answer": "Your refund is guaranteed approved with instant approval."}
    )

    assert RagHallucinationEvaluator().evaluate(case, answer).passed is False


@pytest.mark.parametrize(
    ("answer_text", "forbidden_term"),
    [
        ("A baggage exception costs AED 500.", "AED 500"),
        ("The airline operates from Atlantis Airport.", "Atlantis Airport"),
        ("Economy passengers receive 25 kg cabin baggage.", "25 kg"),
        ("Refunds are always completed in 2 days.", "2 days"),
        ("Every passenger gets a free limousine transfer.", "free limousine"),
        ("The crew will hold the aircraft for late passengers.", "hold the aircraft"),
    ],
)
def test_unsupported_hallucinated_claims_fail(answer_text, forbidden_term):
    context = "The policy says economy passengers may bring one cabin bag up to 7 kg."
    case = RagCase(
        case_id="hallucination-negative",
        question="What does the policy say?",
        expected_document_ids=["DOC-HALL-001"],
        expected_keywords=["7 kg"],
        expected_top_k=1,
        category="hallucination",
        difficulty="negative",
        expected_answer_summary="Cabin baggage is 7 kg.",
        forbidden_terms=[forbidden_term],
    )
    answer = RagAnswer(
        case_id=case.case_id,
        question=case.question,
        answer=answer_text,
        citations=["[Source: DOC-HALL-001#chunk-001]"],
        retrieved_context=[_retrieved("DOC-HALL-001", context)],
    )

    result = RagHallucinationEvaluator().evaluate(case, answer)

    assert result.passed is False
    assert result.details["hallucinated_terms"]


def _retrieved(document_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=RagChunk(
            document_id=document_id,
            chunk_id=f"{document_id}#chunk-001",
            title="Hallucination Test",
            category="hallucination",
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
