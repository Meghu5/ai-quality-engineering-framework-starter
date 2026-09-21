from __future__ import annotations

import json
from pathlib import Path

from ai_eval.baseline import build_baseline_report
from ai_eval.comparison import build_comparisons
from ai_eval.config import AIEvalSettings, DEFAULT_AI_EVAL_SETTINGS
from ai_eval.integrations.deepeval_adapter import DeepEvalAdapter
from ai_eval.integrations.promptfoo_adapter import PromptfooAdapter
from ai_eval.integrations.ragas_adapter import RagasAdapter
from ai_eval.models import Phase10Report
from ai_eval.quality_gate import phase10_quality_gate


REPORT_PATH = Path("reports") / "ai_eval" / "phase10_report.json"


def build_phase10_report(
    settings: AIEvalSettings = DEFAULT_AI_EVAL_SETTINGS,
) -> Phase10Report:
    baseline = build_baseline_report()
    frameworks = [
        RagasAdapter(settings=settings).evaluate(),
        DeepEvalAdapter(settings=settings).evaluate(),
        PromptfooAdapter(settings=settings).evaluate(),
    ]
    comparisons = build_comparisons(frameworks)
    overall_passed = phase10_quality_gate(
        baseline_passed=bool(baseline["overall_passed"]),
        frameworks=frameworks,
        settings=settings,
    )
    return Phase10Report(
        baseline=baseline,
        frameworks=frameworks,
        comparisons=comparisons,
        overall_passed=overall_passed,
    )


def write_phase10_report(
    path: Path = REPORT_PATH,
    settings: AIEvalSettings = DEFAULT_AI_EVAL_SETTINGS,
) -> Phase10Report:
    report = build_phase10_report(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return report
