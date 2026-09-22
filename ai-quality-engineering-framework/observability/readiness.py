from __future__ import annotations

import hashlib
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from observability.analysis import OperationalAnalysisReport
from observability.ci import CI_REPORT_PATH, TRACE_REPORT_PATH, CiCorrelationReport
from observability.config import ObservabilitySettings
from observability.models import TraceEnvelope
from observability.operational_report import (
    OPERATIONAL_REPORT_PATH,
    _load_optional_model,
    _load_optional_models,
    _write_json_atomic,
)
from observability.redaction import REDACTED


logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"
READINESS_REPORT_PATH = Path("reports") / "observability" / "readiness.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/#-]{0,199}")


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReadinessStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"
    UNAVAILABLE = "unavailable"


class DiagnosticFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    severity: DiagnosticSeverity
    category: str
    message: str
    remediation: str
    affected_trace_ids: list[str] = Field(default_factory=list)
    affected_span_ids: list[str] = Field(default_factory=list)
    affected_test_ids: list[str] = Field(default_factory=list)
    affected_case_ids: list[str] = Field(default_factory=list)


class ReadinessCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_evidence_available: bool
    trace_count: int = Field(ge=0)
    span_count: int = Field(ge=0)
    traces_with_spans: int = Field(ge=0)
    traces_without_spans: int = Field(ge=0)
    test_coverage_available: bool
    test_count: int | None = Field(default=None, ge=0)
    tests_with_traces: int | None = Field(default=None, ge=0)
    tests_without_traces: int | None = Field(default=None, ge=0)
    evaluation_coverage_available: bool
    evaluated_case_count: int | None = Field(default=None, ge=0)
    cases_without_trace_lineage: int | None = Field(default=None, ge=0)
    rag_coverage_available: bool
    rag_trace_count: int | None = Field(default=None, ge=0)
    complete_rag_trace_count: int | None = Field(default=None, ge=0)


class ExporterReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observability_enabled: bool
    exporter: str
    configuration_ready: bool
    runtime_health: Literal["unavailable"] = "unavailable"


class ObservabilityReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    status: ReadinessStatus
    ready: bool
    coverage: ReadinessCoverage
    exporter: ExporterReadiness
    diagnostic_counts: dict[str, int]
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    unavailable_reasons: list[str] = Field(default_factory=list)


def analyze_observability_readiness(
    analysis: OperationalAnalysisReport,
    traces: list[TraceEnvelope],
    *,
    ci_report: CiCorrelationReport | None = None,
    settings: ObservabilitySettings | None = None,
    slow_span_threshold_ms: float | None = None,
) -> ObservabilityReadinessReport:
    if slow_span_threshold_ms is not None and slow_span_threshold_ms <= 0:
        raise ValueError("slow_span_threshold_ms must be greater than zero")
    resolved_settings = settings or ObservabilitySettings.from_env()
    findings: list[DiagnosticFinding] = []

    integrity = analysis.integrity
    _add_finding(
        findings,
        present=bool(integrity.duplicate_trace_ids),
        code="duplicate_trace_envelopes",
        severity=DiagnosticSeverity.CRITICAL,
        category="trace_integrity",
        message="Duplicate trace envelopes make trace evidence ambiguous.",
        remediation="Group each trace identifier into exactly one evidence envelope.",
        trace_ids=integrity.duplicate_trace_ids,
    )
    _add_finding(
        findings,
        present=bool(integrity.duplicate_span_ids),
        code="duplicate_spans",
        severity=DiagnosticSeverity.CRITICAL,
        category="trace_integrity",
        message="Duplicate span identifiers make trace evidence ambiguous.",
        remediation="Ensure each emitted span uses a unique span identifier.",
        span_ids=integrity.duplicate_span_ids,
    )
    cycle_ids = sorted(set(integrity.cyclic_span_ids + integrity.self_parented_span_ids))
    _add_finding(
        findings,
        present=bool(cycle_ids),
        code="cyclic_parentage",
        severity=DiagnosticSeverity.CRITICAL,
        category="trace_integrity",
        message="Cyclic or self-referential span parentage prevents trustworthy traversal.",
        remediation="Emit an acyclic parent chain rooted in the current trace.",
        span_ids=cycle_ids,
    )
    _add_finding(
        findings,
        present=bool(integrity.orphan_parents),
        code="orphan_parents",
        severity=DiagnosticSeverity.ERROR,
        category="trace_integrity",
        message="One or more spans reference a parent absent from their trace.",
        remediation="Export parent spans with their children or remove invalid parent references.",
        trace_ids=[item.trace_id for item in integrity.orphan_parents],
        span_ids=[item.span_id for item in integrity.orphan_parents],
    )
    invalid_root_traces = sorted(
        set(integrity.traces_without_roots + integrity.traces_with_multiple_roots)
    )
    _add_finding(
        findings,
        present=bool(invalid_root_traces),
        code="invalid_trace_roots",
        severity=DiagnosticSeverity.ERROR,
        category="trace_integrity",
        message="One or more traces do not have exactly one root span.",
        remediation="Emit exactly one parentless root span for each trace.",
        trace_ids=invalid_root_traces,
    )

    empty_traces = sorted(item.trace_id for item in traces if not item.spans)
    _add_finding(
        findings,
        present=bool(empty_traces),
        code="empty_traces",
        severity=DiagnosticSeverity.WARNING,
        category="telemetry_coverage",
        message="Trace envelopes without spans provide no operation evidence.",
        remediation="Ensure enabled instrumentation exports at least the root span.",
        trace_ids=empty_traces,
    )
    missing_correlations = sorted(item.trace_id for item in traces if not item.correlation_id.strip())
    _add_finding(
        findings,
        present=bool(missing_correlations),
        code="missing_correlation_ids",
        severity=DiagnosticSeverity.WARNING,
        category="correlation",
        message="One or more traces have no usable correlation identifier.",
        remediation="Propagate a non-empty correlation identifier into each trace.",
        trace_ids=missing_correlations,
    )

    if ci_report is not None:
        tests_without_traces = [item.test_id for item in ci_report.tests if not item.trace_ids]
        _add_finding(
            findings,
            present=bool(tests_without_traces),
            code="tests_without_traces",
            severity=DiagnosticSeverity.WARNING,
            category="telemetry_coverage",
            message="Some correlated tests produced no trace evidence.",
            remediation="Instrument expected AI operations or mark non-AI tests outside readiness scope.",
            test_ids=tests_without_traces,
        )
    _add_finding(
        findings,
        present=bool(analysis.correlation.trace_ids_missing_from_evidence),
        code="correlation_missing_traces",
        severity=DiagnosticSeverity.ERROR,
        category="correlation",
        message="CI correlation references trace identifiers absent from trace evidence.",
        remediation="Generate correlation and trace artifacts from the same completed test run.",
        trace_ids=analysis.correlation.trace_ids_missing_from_evidence,
    )
    _add_finding(
        findings,
        present=bool(analysis.correlation.trace_ids_without_tests),
        code="traces_without_tests",
        severity=DiagnosticSeverity.INFO,
        category="correlation",
        message="Some traces are not linked to a CI test.",
        remediation="Review whether these traces are expected background or report-generation work.",
        trace_ids=analysis.correlation.trace_ids_without_tests,
    )

    _add_finding(
        findings,
        present=bool(analysis.rag_lineage.incomplete_trace_ids),
        code="incomplete_rag_lineage",
        severity=DiagnosticSeverity.WARNING,
        category="rag_completeness",
        message="Some RAG traces lack case, document, or chunk lineage.",
        remediation="Record safe case, document, and chunk identifiers on retrieval evidence.",
        trace_ids=analysis.rag_lineage.incomplete_trace_ids,
    )
    _add_rag_component_findings(findings, traces)
    _add_finding(
        findings,
        present=bool(analysis.evaluation.case_ids_without_trace_lineage),
        code="evaluation_cases_without_traces",
        severity=DiagnosticSeverity.WARNING,
        category="evaluation_coverage",
        message="Some evaluated cases have no matching trace lineage.",
        remediation="Propagate safe case identifiers through evaluation and trace evidence.",
        case_ids=analysis.evaluation.case_ids_without_trace_lineage,
    )

    for category, count in sorted(analysis.failure_categories.items()):
        if count:
            _add_finding(
                findings,
                present=True,
                code=f"observed_failure_{category}",
                severity=DiagnosticSeverity.INFO,
                category="observed_failures",
                message="Operational evidence contains a classified failure category.",
                remediation="Review the referenced trace evidence using the recorded failure category.",
            )

    if slow_span_threshold_ms is not None:
        slow = [
            span
            for envelope in traces
            for span in envelope.spans
            if span.duration_ms > slow_span_threshold_ms
        ]
        _add_finding(
            findings,
            present=bool(slow),
            code="slow_spans",
            severity=DiagnosticSeverity.WARNING,
            category="performance",
            message="One or more spans exceed the configured readiness latency threshold.",
            remediation="Review the affected operations against the explicitly configured threshold.",
            trace_ids=[item.trace_id for item in slow],
            span_ids=[item.span_id for item in slow],
        )

    configuration_ready = not (
        resolved_settings.enabled and resolved_settings.exporter == "none"
    )
    if not resolved_settings.enabled:
        _add_finding(
            findings,
            present=True,
            code="observability_disabled",
            severity=DiagnosticSeverity.INFO,
            category="configuration",
            message="Runtime observability is disabled in the evaluated configuration.",
            remediation="Enable observability where production telemetry is required.",
        )
    elif not configuration_ready:
        _add_finding(
            findings,
            present=True,
            code="exporter_not_configured",
            severity=DiagnosticSeverity.WARNING,
            category="configuration",
            message="Observability is enabled without an active exporter.",
            remediation="Configure a supported exporter for the target environment.",
        )

    findings = sorted(findings, key=_finding_sort_key)
    unavailable_reasons = [] if traces else ["trace_evidence_unavailable"]
    status = _resolve_status(findings, trace_evidence_available=bool(traces))
    counts = {
        severity.value: sum(item.severity == severity for item in findings)
        for severity in DiagnosticSeverity
    }
    return ObservabilityReadinessReport(
        status=status,
        ready=status == ReadinessStatus.READY,
        coverage=_coverage(analysis, traces),
        exporter=ExporterReadiness(
            observability_enabled=resolved_settings.enabled,
            exporter=resolved_settings.exporter,
            configuration_ready=configuration_ready,
        ),
        diagnostic_counts=counts,
        findings=findings,
        unavailable_reasons=unavailable_reasons,
    )


def build_readiness_report_from_files(
    *,
    analysis_path: Path = OPERATIONAL_REPORT_PATH,
    trace_path: Path = TRACE_REPORT_PATH,
    ci_path: Path = CI_REPORT_PATH,
    settings: ObservabilitySettings | None = None,
) -> ObservabilityReadinessReport:
    analysis = _load_optional_model(analysis_path, OperationalAnalysisReport)
    if analysis is None:
        raise ValueError("operational analysis is required for readiness evaluation")
    traces = _load_optional_models(trace_path, TraceEnvelope)
    ci_report = _load_optional_model(ci_path, CiCorrelationReport)
    return analyze_observability_readiness(
        analysis, traces, ci_report=ci_report, settings=settings
    )


def write_readiness_report(
    path: Path = READINESS_REPORT_PATH,
    **kwargs,
) -> ObservabilityReadinessReport:
    report = build_readiness_report_from_files(**kwargs)
    _write_json_atomic(path, report.model_dump(mode="json"))
    return report


def main() -> int:
    try:
        write_readiness_report()
    except Exception as exc:
        logger.error("Unable to generate observability readiness report (%s)", type(exc).__name__)
    return 0


def _add_rag_component_findings(
    findings: list[DiagnosticFinding], traces: list[TraceEnvelope]
) -> None:
    expected = {
        "rag.retrieval": "retrieval",
        "llm.generation": "generation",
        "ai.evaluation": "evaluation",
    }
    for operation_name, label in expected.items():
        trace_ids: list[str] = []
        root_span_ids: list[str] = []
        for envelope in traces:
            roots = [span for span in envelope.spans if span.operation_name == "rag.operation"]
            for root in roots:
                if not any(
                    span.operation_name == operation_name
                    and span.parent_span_id == root.span_id
                    for span in envelope.spans
                ):
                    trace_ids.append(envelope.trace_id)
                    root_span_ids.append(root.span_id)
        _add_finding(
            findings,
            present=bool(trace_ids),
            code=f"rag_missing_{label}",
            severity=DiagnosticSeverity.WARNING,
            category="rag_completeness",
            message=f"Some explicit RAG operations have no {label} child span.",
            remediation=f"Instrument the expected {label} stage beneath each RAG operation.",
            trace_ids=trace_ids,
            span_ids=root_span_ids,
        )


def _coverage(
    analysis: OperationalAnalysisReport, traces: list[TraceEnvelope]
) -> ReadinessCoverage:
    correlation = analysis.correlation
    evaluation = analysis.evaluation
    lineage = analysis.rag_lineage
    return ReadinessCoverage(
        trace_evidence_available=bool(traces),
        trace_count=analysis.trace_count,
        span_count=analysis.span_count,
        traces_with_spans=sum(bool(item.spans) for item in traces),
        traces_without_spans=sum(not item.spans for item in traces),
        test_coverage_available=correlation.available,
        test_count=correlation.test_count,
        tests_with_traces=correlation.tests_with_traces,
        tests_without_traces=correlation.tests_without_traces,
        evaluation_coverage_available=evaluation.available,
        evaluated_case_count=(len(evaluation.evaluated_case_ids) if evaluation.available else None),
        cases_without_trace_lineage=(
            len(evaluation.case_ids_without_trace_lineage) if evaluation.available else None
        ),
        rag_coverage_available=lineage.available,
        rag_trace_count=lineage.rag_trace_count,
        complete_rag_trace_count=lineage.complete_trace_count,
    )


def _add_finding(
    findings: list[DiagnosticFinding],
    *,
    present: bool,
    code: str,
    severity: DiagnosticSeverity,
    category: str,
    message: str,
    remediation: str,
    trace_ids: list[str] | None = None,
    span_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    case_ids: list[str] | None = None,
) -> None:
    if not present:
        return
    references = {
        "trace": _safe_references(trace_ids),
        "span": _safe_references(span_ids),
        "test": _safe_references(test_ids, sha256=True),
        "case": _safe_references(case_ids),
    }
    identity = json.dumps([code, references], sort_keys=True, separators=(",", ":"))
    findings.append(
        DiagnosticFinding(
            finding_id=f"{code}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}",
            severity=severity,
            category=category,
            message=message,
            remediation=remediation,
            affected_trace_ids=references["trace"],
            affected_span_ids=references["span"],
            affected_test_ids=references["test"],
            affected_case_ids=references["case"],
        )
    )


def _safe_references(values: list[str] | None, *, sha256: bool = False) -> list[str]:
    safe = set()
    pattern = _SHA256 if sha256 else _SAFE_REFERENCE
    for value in values or []:
        safe.add(value if pattern.fullmatch(value) else REDACTED)
    return sorted(safe)


def _resolve_status(
    findings: list[DiagnosticFinding], *, trace_evidence_available: bool
) -> ReadinessStatus:
    if not trace_evidence_available:
        return ReadinessStatus.UNAVAILABLE
    severities = {item.severity for item in findings}
    if severities & {DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL}:
        return ReadinessStatus.NOT_READY
    if DiagnosticSeverity.WARNING in severities:
        return ReadinessStatus.DEGRADED
    return ReadinessStatus.READY


def _finding_sort_key(finding: DiagnosticFinding) -> tuple:
    severity_order = {
        DiagnosticSeverity.CRITICAL: 0,
        DiagnosticSeverity.ERROR: 1,
        DiagnosticSeverity.WARNING: 2,
        DiagnosticSeverity.INFO: 3,
    }
    return severity_order[finding.severity], finding.category, finding.finding_id


if __name__ == "__main__":
    raise SystemExit(main())
