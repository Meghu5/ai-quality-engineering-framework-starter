from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ai_eval.baseline import build_baseline_report
from ai_eval.comparison import build_comparisons
from ai_eval.config import AIEvalSettings, DEFAULT_AI_EVAL_SETTINGS
from ai_eval.integrations.deepeval_adapter import DeepEvalAdapter
from ai_eval.integrations.promptfoo_adapter import PromptfooAdapter
from ai_eval.integrations.ragas_adapter import RagasAdapter
from ai_eval.models import Phase10Report
from ai_eval.quality_gate import phase10_quality_gate
from observability.context import get_context
from observability.tracing import TracingFacade, create_tracing_facade


REPORT_PATH = Path("reports") / "ai_eval" / "phase10_report.json"


class Phase10EvidenceEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    correlation_id: str
    report: Phase10Report


def build_phase10_report(
    settings: AIEvalSettings = DEFAULT_AI_EVAL_SETTINGS,
    *,
    tracer: TracingFacade | None = None,
) -> Phase10Report:
    resolved_tracer = tracer or create_tracing_facade()
    with resolved_tracer.start_trace(
        "ai.evaluation",
        operation_type="evaluation",
        attributes={"operation": "phase10_report"},
    ) as span:
        report = _build_phase10_report(settings)
        span.set_attribute(
            "evaluation_status", "passed" if report.overall_passed else "failed"
        )
        return report


def _build_phase10_report(settings: AIEvalSettings) -> Phase10Report:
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


def build_correlated_phase10_report(
    settings: AIEvalSettings = DEFAULT_AI_EVAL_SETTINGS,
    *,
    tracer: TracingFacade,
) -> Phase10EvidenceEnvelope:
    with tracer.start_trace(
        "ai.evaluation",
        operation_type="evaluation",
        attributes={"operation": "phase10_report"},
    ) as span:
        context = get_context()
        report = _build_phase10_report(settings)
        span.set_attribute(
            "evaluation_status", "passed" if report.overall_passed else "failed"
        )
        if context is None:
            raise RuntimeError("correlated evidence requires enabled tracing")
        return Phase10EvidenceEnvelope(
            trace_id=context.trace_id,
            correlation_id=context.correlation_id,
            report=report,
        )


def write_phase10_report(
    path: Path = REPORT_PATH,
    settings: AIEvalSettings = DEFAULT_AI_EVAL_SETTINGS,
) -> Phase10Report:
    report = build_phase10_report(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return report
