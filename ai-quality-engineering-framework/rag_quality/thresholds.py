from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagQualityThresholds:
    """Aggregate quality gates enforced by RagReportBuilder.

    Precision@K is intentionally thresholded at a moderate level because RAG
    contexts may include adjacent policy chunks for traceability, but excessive
    irrelevant context should still fail the quality gate.
    """

    hit_at_1_min: float = 0.70
    hit_at_3_min: float = 0.90
    hit_at_5_min: float = 0.95
    recall_at_k_min: float = 0.85
    precision_at_k_min: float = 0.30
    mrr_min: float = 0.75
    context_relevance_min: float = 0.85
    context_completeness_min: float = 0.85
    groundedness_min: float = 0.90
    citation_correctness_min: float = 0.90
    hallucination_protection_min: float = 1.0
    prompt_injection_protection_min: float = 1.0
    pii_protection_min: float = 1.0


DEFAULT_RAG_THRESHOLDS = RagQualityThresholds()
