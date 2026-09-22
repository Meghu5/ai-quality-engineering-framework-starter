from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import observability.operational_report as operational_report
from ai_eval.models import EvaluationResult, FrameworkExecutionReport, Phase10Report
from observability.analysis import OperationalAnalysisReport, analyze_operational_evidence
from observability.ci import CiCorrelationReport, CiTestEvidence
from observability.models import FailureCategory, SpanEvidence, TraceEnvelope, TraceStatus
from observability.operational_report import main, write_operational_report


pytestmark = pytest.mark.observability


def _span(index: int, **overrides) -> SpanEvidence:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = {
        "trace_id": f"{index:032x}",
        "span_id": f"{index:016x}",
        "correlation_id": f"correlation-{index}",
        "operation_name": "rag.operation",
        "operation_type": "rag",
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


def _ci_report(trace_id: str) -> CiCorrelationReport:
    test = CiTestEvidence(
        test_id="a" * 64,
        outcome="passed",
        duration_ms=2,
        correlation_id="correlation-1",
        trace_ids=[trace_id, "f" * 32],
        case_ids=["case-1"],
    )
    return CiCorrelationReport(
        tests=[test], failures=[], summary={"total_tests": 1}
    )


def _phase10_report() -> Phase10Report:
    result = EvaluationResult(
        framework="baseline",
        metric="correctness",
        case_id="case-1",
        score=0.2,
        threshold=0.8,
        passed=False,
        execution_status="executed",
    )
    framework = FrameworkExecutionReport(
        framework="baseline",
        installed=True,
        enabled=True,
        status="executed",
        results=[result],
    )
    return Phase10Report(baseline={}, frameworks=[framework], overall_passed=False)


def test_analysis_aggregates_latency_failures_usage_and_lineage():
    root = _span(
        1,
        attributes={
            "case_id": "case-1",
            "document_id": ["doc-1"],
            "chunk_id": ["chunk-1"],
        },
    )
    child = _span(
        2,
        trace_id=root.trace_id,
        correlation_id=root.correlation_id,
        parent_span_id=root.span_id,
        operation_name="llm.generation",
        operation_type="llm",
        status=TraceStatus.ERROR,
        failure_category=FailureCategory.PROVIDER,
        attributes={
            "provider_name": "openai",
            "model_name": "test-model",
            "token_count": 17,
            "evaluation_status": "failed",
        },
    )
    report = analyze_operational_evidence(
        [_envelope(root, child)],
        ci_report=_ci_report(root.trace_id),
        phase10_report=_phase10_report(),
    )

    assert report.trace_count == 1
    assert report.span_count == 2
    assert report.failure_categories == {"provider": 1}
    assert report.provider_counts == {"openai": 1}
    assert report.model_counts == {"test-model": 1}
    assert report.token_count_total == 17
    assert report.latency.average_ms == 1.5
    assert [item.span_id for item in report.slowest_operations] == [child.span_id, root.span_id]
    assert report.rag_lineage.complete_trace_count == 1
    assert report.correlation.trace_ids_missing_from_evidence == ["f" * 32]
    assert report.evaluation.failed_case_ids == ["case-1"]
    assert report.evaluation.case_ids_without_trace_lineage == []


def test_missing_sources_are_explicit_and_do_not_fabricate_metrics():
    report = analyze_operational_evidence([])
    assert report.trace_count == report.span_count == 0
    assert report.latency.available is False
    assert report.latency.average_ms is None
    assert report.token_count_total is None
    assert report.rag_lineage.available is False
    assert report.correlation.available is False
    assert report.evaluation.available is False
    assert report.warnings == [
        "ci_correlation_unavailable",
        "phase10_evidence_unavailable",
        "trace_evidence_unavailable",
    ]


def test_multiple_traces_preserve_correlation_and_report_incomplete_lineage():
    first = _span(1, attributes={"case_id": "case-1"})
    second = _span(
        2,
        correlation_id=first.correlation_id,
        status=TraceStatus.ERROR,
        failure_category=FailureCategory.TIMEOUT,
        attributes={"case_id": "case-2", "document_id": ["doc-2"]},
    )
    report = analyze_operational_evidence([_envelope(second), _envelope(first)])
    assert report.trace_count == 2
    assert report.failure_categories == {"timeout": 1}
    assert report.rag_lineage.rag_trace_count == 2
    assert report.rag_lineage.complete_trace_count == 0
    assert report.rag_lineage.incomplete_trace_ids == sorted(
        [first.trace_id, second.trace_id]
    )
    assert "incomplete_rag_lineage" in report.warnings


def test_integrity_detects_duplicates_orphans_cycles_and_multiple_roots():
    first = _span(1, parent_span_id=f"{2:016x}")
    second = _span(
        2,
        trace_id=first.trace_id,
        correlation_id=first.correlation_id,
        parent_span_id=first.span_id,
    )
    orphan = _span(
        3,
        trace_id=first.trace_id,
        correlation_id=first.correlation_id,
        parent_span_id="e" * 16,
    )
    report = analyze_operational_evidence(
        [_envelope(first, second, orphan), _envelope(_span(1))]
    )
    assert report.integrity.duplicate_trace_ids == [first.trace_id]
    assert report.integrity.duplicate_span_ids == [first.span_id]
    assert report.integrity.cyclic_span_ids == [first.span_id, second.span_id]
    assert report.integrity.traces_without_roots == [first.trace_id]
    assert report.integrity.orphan_parents[0].span_id == orphan.span_id

    roots = analyze_operational_evidence(
        [
            _envelope(
                _span(4),
                _span(5, trace_id=f"{4:032x}", correlation_id="correlation-4"),
            )
        ]
    )
    assert roots.integrity.traces_with_multiple_roots == [f"{4:032x}"]


def test_duplicate_span_ids_do_not_collapse_attribute_aggregation():
    first = _span(1, attributes={"provider_name": "first"})
    duplicate = _span(
        2,
        trace_id=first.trace_id,
        span_id=first.span_id,
        correlation_id=first.correlation_id,
        attributes={"provider_name": "second"},
    )
    report = analyze_operational_evidence([_envelope(first, duplicate)])
    assert report.provider_counts == {"first": 1, "second": 1}


def test_analysis_sanitizes_attributes_and_never_serializes_raw_content():
    secret = "Bearer secret-token user@example.com raw prompt and response"
    span = _span(
        1,
        attributes={
            "authorization": secret,
            "prompt": secret,
            "response": secret,
            "context": secret,
            "provider_name": secret,
            "model_name": secret,
        },
    )
    serialized = analyze_operational_evidence([_envelope(span)]).model_dump_json()
    assert secret not in serialized
    assert "secret-token" not in serialized
    assert "user@example.com" not in serialized
    assert "raw prompt" not in serialized


def test_report_contract_is_strict_and_output_is_deterministic():
    trace = _envelope(
        _span(2),
        _span(1, trace_id=f"{2:032x}", correlation_id="correlation-2"),
    )
    report = analyze_operational_evidence([trace])
    assert report.model_dump_json() == analyze_operational_evidence([trace]).model_dump_json()
    with pytest.raises(ValidationError):
        OperationalAnalysisReport.model_validate({**report.model_dump(), "extra": True})


def test_file_report_round_trip_and_missing_optional_inputs(tmp_path):
    trace_path = tmp_path / "traces.json"
    output_path = tmp_path / "operational.json"
    trace_path.write_text(
        json.dumps([_envelope(_span(1)).model_dump(mode="json")]), encoding="utf-8"
    )
    report = write_operational_report(
        output_path,
        trace_path=trace_path,
        ci_path=tmp_path / "missing-ci.json",
        phase10_path=tmp_path / "missing-phase10.json",
    )
    assert OperationalAnalysisReport.model_validate_json(output_path.read_text()) == report
    assert not list(tmp_path.glob("*.tmp"))


def test_malformed_input_does_not_replace_existing_output(tmp_path):
    trace_path = tmp_path / "traces.json"
    output_path = tmp_path / "operational.json"
    output_path.write_text('{"existing": true}\n', encoding="utf-8")
    trace_path.write_text('{"secret": "do-not-log"}', encoding="utf-8")
    with pytest.raises(ValueError):
        write_operational_report(
            output_path,
            trace_path=trace_path,
            ci_path=tmp_path / "missing-ci.json",
            phase10_path=tmp_path / "missing-phase10.json",
        )
    assert output_path.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_oversized_input_is_rejected_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(operational_report, "MAX_ARTIFACT_BYTES", 10)
    trace_path = tmp_path / "traces.json"
    trace_path.write_bytes(b" " * 11)
    with pytest.raises(ValueError, match="size limit"):
        write_operational_report(
            tmp_path / "operational.json",
            trace_path=trace_path,
            ci_path=tmp_path / "missing-ci.json",
            phase10_path=tmp_path / "missing-phase10.json",
        )


def test_cli_is_fail_open_and_does_not_log_artifact_content(tmp_path, monkeypatch, caplog):
    secret = "private-artifact-value"
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "reports" / "observability" / "traces.json"
    path.parent.mkdir(parents=True)
    path.write_text(f'{{"value": "{secret}"}}', encoding="utf-8")
    assert main() == 0
    assert secret not in caplog.text


def test_phase10_contract_has_no_observability_fields():
    assert set(Phase10Report.model_fields) == {
        "baseline",
        "frameworks",
        "comparisons",
        "overall_passed",
    }
