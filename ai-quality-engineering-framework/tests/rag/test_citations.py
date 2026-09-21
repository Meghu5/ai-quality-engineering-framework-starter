from __future__ import annotations

import pytest

from rag_quality.evaluators import CitationEvaluator
from rag_quality.models import RagAnswer, RagCase, RagChunk, RetrievedChunk


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_valid_citation_points_to_retrieved_context(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-001")
    answer = rag_pipeline.answer_case(case)

    result = CitationEvaluator().evaluate(answer)

    assert result.passed


def test_missing_citation_fails_when_context_exists(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-001")
    answer = rag_pipeline.answer_case(case).model_copy(update={"citations": []})

    assert CitationEvaluator().evaluate(answer).passed is False


def test_unknown_and_malformed_citations_fail(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-001")
    base = rag_pipeline.answer_case(case)
    answer = RagAnswer(
        case_id=base.case_id,
        question=base.question,
        answer=base.answer,
        citations=["[Source: AIR-NOPE-999#chunk-001]", "bad citation"],
        retrieved_context=base.retrieved_context,
    )

    result = CitationEvaluator().evaluate(answer)

    assert result.passed is False
    assert result.details["unknown"]
    assert result.details["malformed"]


def test_citation_to_retrieved_but_irrelevant_chunk_fails():
    case = _case()
    answer = RagAnswer(
        case_id=case.case_id,
        question=case.question,
        answer="Economy includes 1 checked bag up to 23 kg.",
        citations=["[Source: AIR-LOYALTY-001#chunk-001]"],
        retrieved_context=[
            _retrieved("AIR-BAG-002", "Checked baggage includes 1 checked bag up to 23 kg.", rank=1),
            _retrieved("AIR-LOYALTY-001", "Loyalty members may earn points on eligible flights.", rank=2),
        ],
    )

    result = CitationEvaluator().evaluate(answer, case)

    assert result.passed is False
    assert result.details["irrelevant"] == ["AIR-LOYALTY-001#chunk-001"]


def test_citation_supporting_only_part_of_answer_fails():
    case = _case()
    answer = RagAnswer(
        case_id=case.case_id,
        question=case.question,
        answer="Economy includes 1 checked bag up to 23 kg.",
        citations=["[Source: AIR-BAG-002#chunk-001]"],
        retrieved_context=[
            _retrieved("AIR-BAG-002", "Checked baggage includes 1 checked bag.", rank=1),
        ],
    )

    result = CitationEvaluator().evaluate(answer, case)

    assert result.passed is False
    assert "23 kg" in result.details["unsupported_keywords"]


def test_nonexistent_chunk_in_existing_document_fails(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-001")
    base = rag_pipeline.answer_case(case)
    answer = base.model_copy(update={"citations": ["[Source: AIR-BAG-001#chunk-999]"]})

    result = CitationEvaluator().evaluate(answer, case)

    assert result.passed is False
    assert result.details["unknown"]


def _case() -> RagCase:
    return RagCase(
        case_id="citation-negative",
        question="How many checked bags does economy include?",
        expected_document_ids=["AIR-BAG-002"],
        expected_keywords=["1 checked bag", "23 kg"],
        expected_top_k=2,
        category="baggage",
        difficulty="negative",
        expected_answer_summary="Economy includes 1 checked bag up to 23 kg.",
    )


def _retrieved(document_id: str, content: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=RagChunk(
            document_id=document_id,
            chunk_id=f"{document_id}#chunk-001",
            title=f"{document_id} title",
            category="citation",
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
