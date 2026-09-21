from __future__ import annotations

import pytest

from rag_quality.evaluators import ContextRelevanceEvaluator
from rag_quality.models import RagChunk, RetrievedChunk


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_relevant_context_scores_high(rag_retriever, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-005")
    retrieved = rag_retriever.retrieve(case.question, top_k=3)

    assert ContextRelevanceEvaluator().evaluate(case, retrieved).passed


def test_irrelevant_context_scores_low(rag_retriever, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-005")
    irrelevant = rag_retriever.retrieve("loyalty points retroactive credit", top_k=2)

    assert ContextRelevanceEvaluator().evaluate(case, irrelevant).passed is False


def test_mixed_context_scores_reflect_relevant_fraction(rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-019")
    retrieved = [
        _retrieved("AIR-BAG-001", "Cabin baggage includes one cabin bag up to 7 kg.", rank=1),
        _retrieved("AIR-LOYALTY-001", "Loyalty members earn points on eligible flights.", rank=2),
        _retrieved("AIR-XFER-001", "Transfer passengers should follow airport signs.", rank=3),
    ]

    result = ContextRelevanceEvaluator().evaluate(case, retrieved)

    assert result.passed is False
    assert result.score == pytest.approx(1 / 3)


def test_baggage_query_with_refund_context_fails(rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-019")
    retrieved = [_retrieved("AIR-REFUND-001", "Refund processing depends on fare conditions.", rank=1)]

    result = ContextRelevanceEvaluator().evaluate(case, retrieved)

    assert result.passed is False
    assert result.score == 0.0


def _retrieved(document_id: str, content: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=RagChunk(
            document_id=document_id,
            chunk_id=f"{document_id}#chunk-001",
            title=f"{document_id} title",
            category="context",
            version=1,
            effective_date="2026-01-01",
            source="unit-test",
            content=content,
            start=0,
            end=len(content),
        ),
        score=1.0 / rank,
        rank=rank,
        matched_terms=[],
    )
