from __future__ import annotations

import pytest

from rag_quality.evaluators import ContextCompletenessEvaluator


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_complete_context_for_multi_document_question(rag_retriever, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-015")
    retrieved = rag_retriever.retrieve(case.question, top_k=5)

    assert ContextCompletenessEvaluator().evaluate(case, retrieved).passed


def test_incomplete_context_fails_multi_document_question(rag_retriever, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-015")
    retrieved = [
        item
        for item in rag_retriever.retrieve(case.question, top_k=5)
        if item.chunk.document_id != "AIR-XFER-001"
    ]

    result = ContextCompletenessEvaluator().evaluate(case, retrieved)

    assert result.passed is False
    assert result.score < 1.0
    assert "AIR-XFER-001" in result.details["missing_documents"]
