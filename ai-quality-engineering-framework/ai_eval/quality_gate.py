from __future__ import annotations

from ai_eval.config import AIEvalSettings
from ai_eval.models import FrameworkExecutionReport, Phase10Report


def phase10_quality_gate(
    *,
    baseline_passed: bool,
    frameworks: list[FrameworkExecutionReport],
    settings: AIEvalSettings,
) -> bool:
    if not baseline_passed:
        return False

    for report in frameworks:
        if report.status == "executed":
            if any(result.passed is False for result in report.results):
                return False
        elif report.enabled and settings.fail_on_optional_unavailable:
            return False
    return True


def report_passed(report: Phase10Report) -> bool:
    return report.overall_passed
