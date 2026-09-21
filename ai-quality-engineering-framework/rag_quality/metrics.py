from __future__ import annotations

from rag_quality.models import RagCase, RetrievedChunk


def hit_at_k(case: RagCase, retrieved: list[RetrievedChunk], k: int) -> float:
    expected = set(case.expected_document_ids)
    if not expected:
        return 1.0 if not retrieved else 0.0
    retrieved_ids = {item.chunk.document_id for item in retrieved[:k]}
    return 1.0 if expected.intersection(retrieved_ids) else 0.0


def recall_at_k(case: RagCase, retrieved: list[RetrievedChunk], k: int) -> float:
    expected = set(case.expected_document_ids)
    if not expected:
        return 1.0 if not retrieved else 0.0
    retrieved_ids = {item.chunk.document_id for item in retrieved[:k]}
    return len(expected.intersection(retrieved_ids)) / len(expected)


def precision_at_k(case: RagCase, retrieved: list[RetrievedChunk], k: int) -> float:
    if k <= 0:
        return 0.0
    expected = set(case.expected_document_ids)
    top = retrieved[:k]
    if not top:
        return 1.0 if not expected else 0.0
    if not expected:
        return 1.0 if not top else 0.0
    relevant = sum(item.chunk.document_id in expected for item in top)
    return relevant / len(top)


def reciprocal_rank(case: RagCase, retrieved: list[RetrievedChunk]) -> float:
    expected = set(case.expected_document_ids)
    if not expected:
        return 1.0 if not retrieved else 0.0
    for item in retrieved:
        if item.chunk.document_id in expected:
            return 1.0 / item.rank
    return 0.0


def aggregate_retrieval_metrics(
    cases: list[RagCase],
    retrieved_by_case: dict[str, list[RetrievedChunk]],
) -> dict[str, float]:
    if not cases:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
        }
    return {
        "hit_at_1": _avg(hit_at_k(case, retrieved_by_case[case.case_id], 1) for case in cases),
        "hit_at_3": _avg(hit_at_k(case, retrieved_by_case[case.case_id], 3) for case in cases),
        "hit_at_5": _avg(hit_at_k(case, retrieved_by_case[case.case_id], 5) for case in cases),
        "recall_at_k": _avg(
            recall_at_k(case, retrieved_by_case[case.case_id], case.expected_top_k)
            for case in cases
        ),
        "precision_at_k": _avg(
            precision_at_k(case, retrieved_by_case[case.case_id], case.expected_top_k)
            for case in cases
        ),
        "mrr": _avg(reciprocal_rank(case, retrieved_by_case[case.case_id]) for case in cases),
    }


def _avg(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
