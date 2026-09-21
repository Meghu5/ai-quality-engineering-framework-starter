from __future__ import annotations

import pytest

from rag_quality.retrieval import LexicalRetriever


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_no_documents_returns_no_retrieval_results():
    assert LexicalRetriever([]).retrieve("baggage", top_k=3) == []


def test_unrelated_question_fails_safely(rag_pipeline, rag_cases):
    case = next(case for case in rag_cases if case.case_id == "rag-018")
    answer = rag_pipeline.answer_case(case)

    assert not answer.retrieved_context
    assert "do not have enough" in answer.answer.lower()


def test_extremely_short_and_long_queries_do_not_crash(rag_retriever):
    assert rag_retriever.retrieve("?", top_k=3) == []
    long_query = "baggage " * 300
    assert rag_retriever.retrieve(long_query, top_k=3)
