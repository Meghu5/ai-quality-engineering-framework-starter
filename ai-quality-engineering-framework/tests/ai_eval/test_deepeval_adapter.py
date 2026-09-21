from __future__ import annotations

import pytest

from ai_eval.config import AIEvalSettings
from ai_eval.integrations.deepeval_adapter import DeepEvalAdapter


pytestmark = pytest.mark.ai_eval


def test_deepeval_adapter_maps_phase8_and_phase9_cases():
    adapter = DeepEvalAdapter(settings=AIEvalSettings(False, False, False, "none"))
    ai_rows = adapter.build_ai_test_cases(limit=2)
    rag_rows = adapter.build_rag_test_cases(limit=2)

    assert {"case_id", "input", "actual_output", "expected_output"}.issubset(ai_rows[0])
    assert {"case_id", "input", "actual_output", "retrieval_context"}.issubset(rag_rows[0])
    assert ai_rows[0]["case_id"].startswith("ai-")
    assert rag_rows[0]["case_id"].startswith("rag-")


def test_deepeval_adapter_reports_unavailable_or_provider_required_without_scores():
    adapter = DeepEvalAdapter(settings=AIEvalSettings(False, True, False, "none"))
    report = adapter.evaluate()

    assert report.framework == "deepeval"
    assert report.status in {"unavailable", "provider_required"}
    assert report.results == []


def test_deepeval_normalizes_executed_metric():
    adapter = DeepEvalAdapter(settings=AIEvalSettings(False, False, False, "none"))
    result = adapter.normalize_metric(
        case_id="ai-001-flight-search",
        metric="answer_relevancy",
        score=0.75,
        threshold=0.8,
        provider="unit",
    )

    assert result.execution_status == "executed"
    assert result.passed is False
