from __future__ import annotations

import pytest

from ai_eval.config import AIEvalSettings
from ai_eval.models import EvaluationResult, FrameworkExecutionReport
from ai_eval.quality_gate import phase10_quality_gate


pytestmark = pytest.mark.ai_eval


def test_quality_gate_passes_when_baseline_passes_and_optional_frameworks_are_unavailable():
    frameworks = [
        FrameworkExecutionReport(
            framework="ragas",
            installed=False,
            enabled=True,
            status="unavailable",
            reason="not installed",
        )
    ]

    assert phase10_quality_gate(
        baseline_passed=True,
        frameworks=frameworks,
        settings=AIEvalSettings(True, False, False, "none"),
    )


def test_quality_gate_fails_executed_optional_threshold_failure():
    frameworks = [
        FrameworkExecutionReport(
            framework="ragas",
            installed=True,
            enabled=True,
            provider="unit",
            status="executed",
            results=[
                EvaluationResult(
                    framework="ragas",
                    metric="faithfulness",
                    case_id="rag-001",
                    score=0.5,
                    threshold=0.8,
                    passed=False,
                    provider="unit",
                    execution_status="executed",
                )
            ],
        )
    ]

    assert not phase10_quality_gate(
        baseline_passed=True,
        frameworks=frameworks,
        settings=AIEvalSettings(True, False, False, "unit"),
    )
