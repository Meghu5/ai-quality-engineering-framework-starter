from __future__ import annotations

import pytest

from ai_eval.models import EvaluationResult


pytestmark = pytest.mark.ai_eval


def test_evaluation_result_distinguishes_unexecuted_from_failed_score():
    result = EvaluationResult(
        framework="ragas",
        metric="faithfulness",
        case_id="rag-001",
        execution_status="provider_required",
        score=None,
        passed=None,
        reason="judge provider required",
    )

    assert result.score is None
    assert result.passed is None
    assert result.execution_status == "provider_required"


def test_evaluation_result_rejects_unknown_status():
    with pytest.raises(ValueError):
        EvaluationResult(
            framework="x",
            metric="m",
            case_id="c",
            execution_status="made_up",
        )


def test_framework_status_supports_enabled_but_not_executed():
    from ai_eval.models import FrameworkExecutionReport

    report = FrameworkExecutionReport(
        framework="promptfoo",
        installed=True,
        enabled=True,
        status="not_executed",
        reason="result artifact is absent",
    )

    assert report.status == "not_executed"
    assert report.results == []
