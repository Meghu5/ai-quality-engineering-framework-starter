from __future__ import annotations

import pytest

from ai_eval.config import AIEvalSettings
from ai_eval.integrations.ragas_adapter import RagasAdapter


pytestmark = pytest.mark.ai_eval


def test_ragas_adapter_maps_phase9_cases_to_ragas_shape():
    adapter = RagasAdapter(settings=AIEvalSettings(False, False, False, "none"))
    rows = adapter.build_dataset(limit=2)

    assert len(rows) == 2
    assert {"case_id", "question", "answer", "contexts", "ground_truth"}.issubset(rows[0])
    assert isinstance(rows[0]["contexts"], list)
    assert rows[0]["case_id"].startswith("rag-")


def test_ragas_adapter_reports_unavailable_or_provider_required_without_scores():
    adapter = RagasAdapter(settings=AIEvalSettings(True, False, False, "none"))
    report = adapter.evaluate()

    assert report.framework == "ragas"
    assert report.status in {"unavailable", "provider_required"}
    assert report.results == []


def test_ragas_normalizes_executed_metric():
    adapter = RagasAdapter(settings=AIEvalSettings(False, False, False, "none"))
    result = adapter.normalize_metric(
        case_id="rag-001",
        metric="faithfulness",
        score=0.9,
        threshold=0.8,
        provider="unit",
    )

    assert result.execution_status == "executed"
    assert result.passed is True
