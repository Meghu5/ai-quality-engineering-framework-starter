from __future__ import annotations

import pytest

from rag_quality.metrics import aggregate_retrieval_metrics, hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from rag_quality.models import RagCase, RagChunk, RetrievedChunk


pytestmark = [pytest.mark.rag, pytest.mark.rag_retrieval]


def test_retrieval_metric_functions(rag_cases, rag_retriever):
    case = next(case for case in rag_cases if case.case_id == "rag-001")
    retrieved = rag_retriever.retrieve(case.question, top_k=5)

    assert hit_at_k(case, retrieved, 1) == 1.0
    assert recall_at_k(case, retrieved, 3) == 1.0
    assert precision_at_k(case, retrieved, 3) > 0
    assert reciprocal_rank(case, retrieved) == 1.0


def test_metric_functions_use_hand_verifiable_ranked_results():
    case = RagCase(
        case_id="metric-001",
        question="baggage and refund",
        expected_document_ids=["DOC-A", "DOC-C"],
        expected_keywords=["baggage", "refund"],
        expected_top_k=3,
        category="metric",
        difficulty="unit",
        expected_answer_summary="Baggage and refund documents are relevant.",
    )
    retrieved = [
        _retrieved("DOC-X", 1),
        _retrieved("DOC-A", 2),
        _retrieved("DOC-B", 3),
        _retrieved("DOC-C", 4),
    ]

    assert hit_at_k(case, retrieved, 1) == 0.0
    assert hit_at_k(case, retrieved, 3) == 1.0
    assert recall_at_k(case, retrieved, 3) == 0.5
    assert precision_at_k(case, retrieved, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank(case, retrieved) == 0.5


def test_negative_expected_documents_reward_empty_retrieval_only():
    case = RagCase(
        case_id="metric-negative",
        question="mars hotel",
        expected_document_ids=[],
        expected_keywords=[],
        expected_top_k=3,
        category="unsupported",
        difficulty="unit",
        expected_answer_summary="No answer expected.",
        expect_no_answer=True,
    )

    assert hit_at_k(case, [], 3) == 1.0
    assert recall_at_k(case, [], 3) == 1.0
    assert precision_at_k(case, [], 3) == 1.0
    assert reciprocal_rank(case, []) == 1.0
    assert hit_at_k(case, [_retrieved("DOC-X", 1)], 3) == 0.0
    assert precision_at_k(case, [_retrieved("DOC-X", 1)], 3) == 0.0


def test_aggregate_retrieval_metrics_meet_threshold_shape(rag_cases, rag_retriever):
    retrieved_by_case = {
        case.case_id: rag_retriever.retrieve(case.question, top_k=max(case.expected_top_k, 5))
        for case in rag_cases
    }
    metrics = aggregate_retrieval_metrics(rag_cases, retrieved_by_case)

    assert set(metrics) == {"hit_at_1", "hit_at_3", "hit_at_5", "recall_at_k", "precision_at_k", "mrr"}
    assert metrics["hit_at_3"] >= 0.9
    assert metrics["mrr"] >= 0.75


def _retrieved(document_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=RagChunk(
            document_id=document_id,
            chunk_id=f"{document_id}#chunk-001",
            title=f"{document_id} title",
            category="metric",
            version=1,
            effective_date="2026-01-01",
            source="unit-test",
            content=f"{document_id} policy content",
            start=0,
            end=20,
        ),
        score=1.0 / rank,
        rank=rank,
        matched_terms=[],
    )
