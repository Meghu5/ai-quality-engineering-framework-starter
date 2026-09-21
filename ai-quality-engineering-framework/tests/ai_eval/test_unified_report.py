from __future__ import annotations

import pytest

from ai_eval.config import AIEvalSettings
from ai_eval.report import build_phase10_report, write_phase10_report


pytestmark = pytest.mark.ai_eval


def test_phase10_report_contains_baseline_and_optional_framework_statuses():
    report = build_phase10_report(AIEvalSettings(False, False, False, "none"))

    assert report.baseline["overall_passed"] is True
    assert report.baseline["phase8"]["execution_status"] == "executed"
    assert report.baseline["phase9"]["execution_status"] == "executed"
    assert {framework.framework for framework in report.frameworks} == {
        "ragas",
        "deepeval",
        "promptfoo",
    }
    assert report.overall_passed is True
    assert all(framework.results == [] for framework in report.frameworks)


def test_phase10_report_writes_machine_readable_artifact(tmp_path):
    path = tmp_path / "phase10_report.json"
    report = write_phase10_report(path, AIEvalSettings(False, False, False, "none"))

    assert path.exists()
    assert report.overall_passed is True
