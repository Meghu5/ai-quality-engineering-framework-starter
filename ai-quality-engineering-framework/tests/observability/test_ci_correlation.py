from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ai_eval.models import Phase10Report
from observability.ci import (
    CiTestEvidence,
    TestCorrelationState as CorrelationState,
    begin_test_correlation,
    build_correlation_report,
    build_test_evidence,
    classify_test_exception,
    end_test_correlation,
    get_current_test_state,
    load_test_evidence,
    load_test_spans,
    sanitize_node_id,
    stable_test_id,
    worker_id_from_environment,
    write_ci_reports,
    write_test_evidence,
    write_test_spans,
    write_trace_report,
)
from observability.config import ObservabilitySettings
from observability.context import get_context
from observability.exporters import InMemoryExporter
from observability.models import FailureCategory, SpanEvidence, TraceStatus
from observability.tracing import TracingFacade


pytestmark = pytest.mark.observability


def _span(**overrides):
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = {
        "trace_id": "1" * 32,
        "span_id": "2" * 16,
        "correlation_id": "c" * 64,
        "operation_name": "rag.retrieval",
        "operation_type": "retrieval",
        "started_at": started,
        "ended_at": started + timedelta(milliseconds=2),
        "duration_ms": 2,
        "status": TraceStatus.OK,
        "attributes": {
            "case_id": "rag-001",
            "document_id": ["AIR-BAG-001"],
            "chunk_id": ["AIR-BAG-001#chunk-001"],
            "evaluation_status": "passed",
        },
        "service_name": "test-service",
        "environment": "test",
    }
    values.update(overrides)
    return SpanEvidence(**values)


def _state(*, outcome="passed", spans=None, category=None):
    return CorrelationState(
        test_id="a" * 64,
        node_id="tests/test_safe.py::test_case",
        correlation_id="c" * 64,
        spans=spans or [],
        phase_outcomes={"setup": "passed", "call": outcome, "teardown": "passed"},
        duration_ms=12.3456,
        test_failure_category=category,
    )


def test_stable_test_id_is_repeatable_and_collision_resistant_for_node_ids():
    first = stable_test_id("tests/test_a.py::test_case[param]")
    assert first == stable_test_id("tests/test_a.py::test_case[param]")
    assert first != stable_test_id("tests/test_a.py::test_case[other]")
    assert len(first) == 64


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        ("tests/test_safe.py::test_case", "tests/test_safe.py::test_case"),
        ("tests/test_safe.py::test_case[param-1]", "tests/test_safe.py::test_case[redacted]"),
        ("x" * 241, None),
        ("tests/test.py::test_case[user@example.com]", "tests/test.py::test_case[redacted]"),
        ("tests/test.py::test_case[Bearer token]", "tests/test.py::test_case[redacted]"),
        ("tests/test.py::test_case[4111111111111111]", "tests/test.py::test_case[redacted]"),
        ("tests/test.py::test_case[url?token=secret]", "tests/test.py::test_case[redacted]"),
    ],
)
def test_node_id_sanitization(node_id, expected):
    assert sanitize_node_id(node_id) == expected


@pytest.mark.parametrize(
    "parameter_payload",
    [
        "A" * 160,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature",
        "sk_live_51ABCDEF0123456789",
        "aB9xQ2mN7pR4vT8zK6wY3cF5hJ1sL0dG",
        "value1-value2-secret-value3",
    ],
)
def test_parametrized_node_id_never_preserves_original_payload(parameter_payload):
    original = f"tests/test_api.py::test_create_booking[{parameter_payload}]"
    sanitized = sanitize_node_id(original)
    assert sanitized == "tests/test_api.py::test_create_booking[redacted]"
    assert parameter_payload not in sanitized


def test_non_parametrized_node_id_remains_readable():
    node_id = "tests/test_api.py::test_create_booking"
    assert sanitize_node_id(node_id) == node_id


def test_full_original_node_id_still_drives_stable_test_id():
    first = "tests/test_api.py::test_create_booking[first-secret]"
    second = "tests/test_api.py::test_create_booking[second-secret]"
    assert stable_test_id(first) == stable_test_id(first)
    assert stable_test_id(first) != stable_test_id(second)


def test_context_setup_and_restoration_are_isolated():
    first, first_token = begin_test_correlation("tests/test_a.py::test_one")
    assert get_context() is None
    assert get_current_test_state().correlation_id == first.test_id
    end_test_correlation(first_token)
    assert get_context() is None

    second, second_token = begin_test_correlation("tests/test_a.py::test_two")
    assert second.correlation_id != first.correlation_id
    end_test_correlation(second_token)
    assert get_context() is None


def test_independent_traces_inherit_correlation_without_parenting():
    exporter = InMemoryExporter()
    tracer = TracingFacade(
        ObservabilitySettings(enabled=True, exporter="memory"), exporter
    )
    state, token = begin_test_correlation("tests/test_trace.py::test_independent")
    try:
        with tracer.start_trace("first"):
            pass
        with tracer.start_trace("second"):
            pass
    finally:
        end_test_correlation(token)

    first, second = exporter.spans
    assert first.trace_id != second.trace_id
    assert first.correlation_id == second.correlation_id == state.correlation_id
    assert first.parent_span_id is None
    assert second.parent_span_id is None
    assert {span.trace_id for span in state.spans} == {first.trace_id, second.trace_id}


def test_explicit_correlation_id_overrides_inherited_value():
    exporter = InMemoryExporter()
    tracer = TracingFacade(
        ObservabilitySettings(enabled=True, exporter="memory"), exporter
    )
    state, token = begin_test_correlation("tests/test_trace.py::test_override")
    try:
        with tracer.start_trace("explicit", correlation_id="explicit-correlation"):
            pass
    finally:
        end_test_correlation(token)
    assert exporter.spans[0].correlation_id == "explicit-correlation"
    assert state.spans == []


def test_case_document_chunk_and_evaluation_lineage_are_collected():
    evidence = build_test_evidence(_state(spans=[_span()]))
    assert evidence.case_ids == ["rag-001"]
    assert evidence.document_ids == ["AIR-BAG-001"]
    assert evidence.chunk_ids == ["AIR-BAG-001#chunk-001"]
    assert evidence.evaluation_statuses == ["passed"]
    assert evidence.trace_ids == ["1" * 32]
    assert evidence.span_ids == ["2" * 16]


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped", "xfailed", "xpassed"])
def test_supported_pytest_outcomes(outcome):
    state = _state(
        outcome=outcome,
        category=FailureCategory.CONTRACT if outcome == "failed" else None,
    )
    assert build_test_evidence(state).outcome == outcome


def test_failure_classification_is_conservative():
    assert classify_test_exception(AssertionError) == FailureCategory.CONTRACT
    assert classify_test_exception(TimeoutError) == FailureCategory.TIMEOUT
    assert classify_test_exception(ValidationError) == FailureCategory.VALIDATION
    assert classify_test_exception(RuntimeError) == FailureCategory.UNKNOWN
    assert classify_test_exception(None) == FailureCategory.UNKNOWN


def test_operational_failure_category_takes_precedence():
    failed_span = _span(
        status=TraceStatus.ERROR,
        failure_category=FailureCategory.RETRIEVAL,
    )
    evidence = build_test_evidence(
        _state(
            outcome="failed",
            spans=[failed_span],
            category=FailureCategory.CONTRACT,
        )
    )
    assert evidence.failure_category == FailureCategory.RETRIEVAL


def test_missing_telemetry_is_valid_and_does_not_fabricate_ids():
    evidence = build_test_evidence(_state())
    assert evidence.trace_ids == []
    assert evidence.span_ids == []
    assert evidence.artifact_refs == []


def test_report_order_and_summary_are_deterministic():
    second = CiTestEvidence(
        test_id="b" * 64,
        outcome="passed",
        duration_ms=1,
        correlation_id="2" * 64,
    )
    first = CiTestEvidence(
        test_id="a" * 64,
        outcome="failed",
        duration_ms=1,
        correlation_id="1" * 64,
        failure_category=FailureCategory.CONTRACT,
    )
    report = build_correlation_report([second, first])
    assert [item.test_id for item in report.tests] == [first.test_id, second.test_id]
    assert report.summary["total_tests"] == 2
    assert report.summary["failures_by_category"] == {"contract": 1}


def test_worker_scoped_evidence_round_trip_is_xdist_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    evidence = build_test_evidence(_state(spans=[_span()]))
    evidence_path = write_test_evidence(
        evidence, tmp_path, worker_id_from_environment()
    )
    span_path = write_test_spans(
        [_span()], evidence.test_id, tmp_path, worker_id_from_environment()
    )
    assert "gw1" in evidence_path.parts
    assert "gw1" in span_path.parts
    assert load_test_evidence(tmp_path) == [evidence]
    assert load_test_spans(tmp_path) == [_span()]


def test_artifacts_are_safe_deterministic_and_network_free(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: pytest.fail("network used"))
    evidence = build_test_evidence(_state(spans=[_span()]))
    correlation = tmp_path / "test-trace-correlation.json"
    failures = tmp_path / "failure-summary.json"
    traces = tmp_path / "traces.json"
    write_ci_reports([evidence], report_path=correlation, failure_path=failures)
    write_trace_report([_span()], traces)

    serialized = correlation.read_text(encoding="utf-8") + traces.read_text(encoding="utf-8")
    assert json.loads(correlation.read_text(encoding="utf-8"))["tests"]
    for forbidden in (
        "raw prompt",
        "raw response",
        "raw context",
        "authorization",
        "api_key",
        "password",
        "cookie",
        "person@example.com",
        "4111111111111111",
    ):
        assert forbidden not in serialized.lower()


def test_phase10_strict_model_has_no_observability_fields():
    assert "trace_id" not in Phase10Report.model_fields
    assert "correlation_id" not in Phase10Report.model_fields
