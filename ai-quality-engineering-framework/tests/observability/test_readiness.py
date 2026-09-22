from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ai_eval.models import EvaluationResult, FrameworkExecutionReport, Phase10Report
from observability.analysis import analyze_operational_evidence
from observability.ci import CiCorrelationReport, CiTestEvidence
from observability.config import ObservabilitySettings
from observability.models import FailureCategory, SpanEvidence, TraceEnvelope, TraceStatus
from observability.readiness import (
    ObservabilityReadinessReport,
    ReadinessStatus,
    analyze_observability_readiness,
    main,
    write_readiness_report,
)


pytestmark = pytest.mark.observability
ENABLED = ObservabilitySettings(enabled=True, exporter="json")


def _span(index: int, **overrides) -> SpanEvidence:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = {
        "trace_id": f"{index:032x}",
        "span_id": f"{index:016x}",
        "parent_span_id": None,
        "correlation_id": "correlation-1",
        "operation_name": "operation",
        "operation_type": "internal",
        "started_at": started,
        "ended_at": started + timedelta(milliseconds=index),
        "duration_ms": float(index),
        "status": TraceStatus.OK,
        "attributes": {},
        "service_name": "quality-engine",
        "environment": "test",
    }
    values.update(overrides)
    return SpanEvidence(**values)


def _envelope(*spans: SpanEvidence) -> TraceEnvelope:
    first = spans[0]
    return TraceEnvelope(
        trace_id=first.trace_id,
        correlation_id=first.correlation_id,
        service_name=first.service_name,
        environment=first.environment,
        spans=list(spans),
    )


def _rag_trace(*, omit: str | None = None) -> TraceEnvelope:
    root = _span(
        1,
        operation_name="rag.operation",
        operation_type="rag",
        attributes={
            "case_id": "case-1",
            "document_id": ["doc-1"],
            "chunk_id": ["chunk-1"],
        },
    )
    names = ["rag.retrieval", "llm.generation", "ai.evaluation"]
    children = [
        _span(
            index,
            trace_id=root.trace_id,
            correlation_id=root.correlation_id,
            parent_span_id=root.span_id,
            operation_name=name,
            operation_type={
                "rag.retrieval": "retrieval",
                "llm.generation": "llm",
                "ai.evaluation": "evaluation",
            }[name],
            attributes={
                "case_id": "case-1",
                "document_id": ["doc-1"],
                "chunk_id": ["chunk-1"],
                "evaluation_status": "passed",
            },
        )
        for index, name in enumerate(names, start=2)
        if name != omit
    ]
    return _envelope(root, *children)


def _ci(
    trace_ids: list[str],
    *,
    test_id: str = "a" * 64,
    node_id: str | None = None,
) -> CiCorrelationReport:
    test = CiTestEvidence(
        test_id=test_id,
        node_id=node_id,
        outcome="passed",
        duration_ms=1,
        correlation_id="correlation-1",
        trace_ids=trace_ids,
        case_ids=["case-1"],
    )
    return CiCorrelationReport(tests=[test], failures=[], summary={"total_tests": 1})


def _phase10(case_id: str = "case-1") -> Phase10Report:
    result = EvaluationResult(
        framework="baseline",
        metric="quality",
        case_id=case_id,
        passed=True,
        execution_status="executed",
    )
    framework = FrameworkExecutionReport(
        framework="baseline",
        installed=True,
        enabled=True,
        status="executed",
        results=[result],
    )
    return Phase10Report(baseline={}, frameworks=[framework], overall_passed=True)


def _readiness(
    traces: list[TraceEnvelope],
    *,
    ci_report: CiCorrelationReport | None = None,
    phase10_report: Phase10Report | None = None,
    settings: ObservabilitySettings = ENABLED,
    threshold: float | None = None,
) -> ObservabilityReadinessReport:
    analysis = analyze_operational_evidence(
        traces, ci_report=ci_report, phase10_report=phase10_report
    )
    return analyze_observability_readiness(
        analysis,
        traces,
        ci_report=ci_report,
        settings=settings,
        slow_span_threshold_ms=threshold,
    )


def _codes(report: ObservabilityReadinessReport) -> set[str]:
    return {item.finding_id.rsplit("-", 1)[0] for item in report.findings}


def test_healthy_correlated_rag_evidence_is_ready():
    trace = _rag_trace()
    report = _readiness(
        [trace], ci_report=_ci([trace.trace_id]), phase10_report=_phase10()
    )
    assert report.status == ReadinessStatus.READY
    assert report.ready is True
    assert report.coverage.tests_without_traces == 0
    assert report.coverage.complete_rag_trace_count == 1
    assert report.diagnostic_counts["warning"] == 0


def test_empty_evidence_is_unavailable_without_fabricated_coverage():
    report = _readiness([])
    assert report.status == ReadinessStatus.UNAVAILABLE
    assert report.ready is False
    assert report.coverage.trace_evidence_available is False
    assert report.coverage.test_count is None
    assert report.unavailable_reasons == ["trace_evidence_unavailable"]


@pytest.mark.parametrize(
    ("omit", "expected"),
    [
        ("rag.retrieval", "rag_missing_retrieval"),
        ("llm.generation", "rag_missing_generation"),
        ("ai.evaluation", "rag_missing_evaluation"),
    ],
)
def test_explicit_rag_operation_missing_expected_child_is_degraded(omit, expected):
    trace = _rag_trace(omit=omit)
    report = _readiness([trace], ci_report=_ci([trace.trace_id]))
    assert report.status == ReadinessStatus.DEGRADED
    assert expected in _codes(report)


def test_optional_non_rag_trace_does_not_require_rag_children():
    trace = _envelope(_span(1))
    report = _readiness([trace], ci_report=_ci([trace.trace_id]))
    assert report.status == ReadinessStatus.READY
    assert not any(code.startswith("rag_missing") for code in _codes(report))


def test_missing_test_trace_and_evaluation_lineage_are_warnings():
    trace = _envelope(_span(1, attributes={"case_id": "case-1"}))
    report = _readiness(
        [trace], ci_report=_ci([]), phase10_report=_phase10("case-2")
    )
    assert report.status == ReadinessStatus.DEGRADED
    assert {"tests_without_traces", "evaluation_cases_without_traces"} <= _codes(report)


def test_missing_correlated_trace_is_not_ready():
    trace = _envelope(_span(1))
    report = _readiness([trace], ci_report=_ci(["f" * 32]))
    assert report.status == ReadinessStatus.NOT_READY
    assert "correlation_missing_traces" in _codes(report)


def test_integrity_corruption_is_critical_and_deterministic():
    first = _span(1, parent_span_id=f"{2:016x}")
    second = _span(
        2,
        trace_id=first.trace_id,
        correlation_id=first.correlation_id,
        parent_span_id=first.span_id,
    )
    trace = _envelope(second, first)
    report = _readiness([trace])
    repeated = _readiness([trace])
    assert report.status == ReadinessStatus.NOT_READY
    assert "cyclic_parentage" in _codes(report)
    assert report.model_dump_json() == repeated.model_dump_json()


def test_duplicate_trace_envelopes_are_critical():
    trace = _envelope(_span(1))
    report = _readiness([trace, trace.model_copy(deep=True)])
    assert report.status == ReadinessStatus.NOT_READY
    assert "duplicate_trace_envelopes" in _codes(report)


def test_orphan_and_invalid_root_are_errors():
    span = _span(1, parent_span_id="f" * 16)
    report = _readiness([_envelope(span)])
    assert report.status == ReadinessStatus.NOT_READY
    assert {"orphan_parents", "invalid_trace_roots"} <= _codes(report)


def test_empty_trace_and_missing_correlation_are_coverage_warnings():
    trace = TraceEnvelope(
        trace_id="1" * 32,
        correlation_id=" ",
        service_name="quality-engine",
        environment="test",
        spans=[],
    )
    analysis = analyze_operational_evidence([trace])
    report = analyze_observability_readiness(analysis, [trace], settings=ENABLED)
    assert report.status == ReadinessStatus.DEGRADED
    assert {"empty_traces", "missing_correlation_ids"} <= _codes(report)


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.PROVIDER,
        FailureCategory.MODEL,
        FailureCategory.TIMEOUT,
        FailureCategory.NETWORK,
    ],
)
def test_observed_application_failures_are_informational(category):
    span = _span(1, status=TraceStatus.ERROR, failure_category=category)
    trace = _envelope(span)
    report = _readiness([trace], ci_report=_ci([trace.trace_id]))
    assert report.status == ReadinessStatus.READY
    assert f"observed_failure_{category.value}" in _codes(report)


def test_latency_finding_requires_explicit_threshold():
    trace = _envelope(_span(1, duration_ms=50))
    assert "slow_spans" not in _codes(_readiness([trace]))
    report = _readiness([trace], threshold=10)
    assert report.status == ReadinessStatus.DEGRADED
    assert "slow_spans" in _codes(report)
    with pytest.raises(ValueError):
        _readiness([trace], threshold=0)


def test_exporter_configuration_is_separate_from_runtime_health():
    trace = _envelope(_span(1))
    report = _readiness(
        [trace], settings=ObservabilitySettings(enabled=True, exporter="none")
    )
    assert report.status == ReadinessStatus.DEGRADED
    assert report.exporter.configuration_ready is False
    assert report.exporter.runtime_health == "unavailable"


@pytest.mark.parametrize(
    "unsafe",
    [
        "sk-live-1234567890ABCDEF",
        "eyJhbGciOiJIUzI1NiJ9.payload.signature",
        "Bearer secret-token",
        "password=secret",
        "person@example.com",
        "+1 (202) 555-0112",
        "4111 1111 1111 1111",
        "tests/test_api.py::test_case[raw-secret-value]",
        "aB9xQ2mN7pR4vT8zK6wY3cF5hJ1sL0dG",
    ],
)
def test_unsafe_identifiers_never_reach_diagnostics(unsafe):
    trace = _envelope(_span(1))
    report = _readiness(
        [trace], ci_report=_ci([], test_id=unsafe, node_id=unsafe)
    )
    serialized = report.model_dump_json()
    assert unsafe not in serialized
    assert "[REDACTED]" in serialized


def test_strict_contract_and_deterministic_finding_order():
    trace = _rag_trace(omit="rag.retrieval")
    report = _readiness([trace], ci_report=_ci([]), phase10_report=_phase10("case-2"))
    assert report.findings == sorted(
        report.findings,
        key=lambda item: (
            {"critical": 0, "error": 1, "warning": 2, "info": 3}[item.severity.value],
            item.category,
            item.finding_id,
        ),
    )
    with pytest.raises(ValidationError):
        ObservabilityReadinessReport.model_validate({**report.model_dump(), "extra": True})


def test_file_report_is_atomic_and_round_trips(tmp_path):
    trace = _envelope(_span(1))
    analysis = analyze_operational_evidence([trace])
    analysis_path = tmp_path / "operational.json"
    trace_path = tmp_path / "traces.json"
    output_path = tmp_path / "readiness.json"
    analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    trace_path.write_text(
        json.dumps([trace.model_dump(mode="json")]), encoding="utf-8"
    )
    report = write_readiness_report(
        output_path,
        analysis_path=analysis_path,
        trace_path=trace_path,
        ci_path=tmp_path / "missing-ci.json",
        settings=ENABLED,
    )
    assert ObservabilityReadinessReport.model_validate_json(output_path.read_text()) == report
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_is_fail_open_and_does_not_log_malformed_content(tmp_path, monkeypatch, caplog):
    secret = "private-readiness-input"
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "reports" / "observability" / "operational-analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text(f'{{"secret": "{secret}"}}', encoding="utf-8")
    assert main() == 0
    assert secret not in caplog.text


def test_phase10_contract_remains_independent():
    assert set(Phase10Report.model_fields) == {
        "baseline",
        "frameworks",
        "comparisons",
        "overall_passed",
    }
