from __future__ import annotations

import pytest

from observability.config import ObservabilitySettings
from observability.context import get_context, get_span_id, get_trace_id
from observability.exporters import InMemoryExporter
from observability.models import FailureCategory, TraceStatus
from observability.tracing import TracingFacade


pytestmark = pytest.mark.observability


def test_disabled_tracing_is_a_true_noop():
    exporter = InMemoryExporter()
    facade = TracingFacade(ObservabilitySettings(), exporter)

    with facade.start_trace("disabled") as span:
        span.set_attribute("case_id", "case-1")
        assert facade.get_current_trace() is None

    assert exporter.spans == []
    assert get_context() is None


def test_enabled_tracing_exports_parent_and_child(tracing, memory_exporter):
    with tracing.start_trace("request", correlation_id="corr-1"):
        parent_span_id = get_span_id()
        trace_id = get_trace_id()
        with tracing.start_span("retrieve", operation_type="retrieval"):
            assert get_trace_id() == trace_id

    child, parent = memory_exporter.spans
    assert child.parent_span_id == parent_span_id
    assert parent.parent_span_id is None
    assert child.trace_id == parent.trace_id
    assert child.correlation_id == "corr-1"
    assert child.status == TraceStatus.OK
    assert get_context() is None


def test_exception_is_recorded_without_replacing_original(tracing, memory_exporter):
    original = ValueError("business failure")

    with pytest.raises(ValueError) as captured:
        with tracing.start_trace("validate") as span:
            span.record_exception(original, FailureCategory.VALIDATION)
            raise original

    assert captured.value is original
    evidence = memory_exporter.spans[0]
    assert evidence.status == TraceStatus.ERROR
    assert evidence.failure_category == FailureCategory.VALIDATION
    assert evidence.attributes["exception_type"] == "ValueError"
    assert "business failure" not in str(evidence.model_dump())


def test_exporter_failure_cannot_break_application_execution():
    class BrokenExporter:
        def export(self, span):
            raise RuntimeError("telemetry unavailable")

    facade = TracingFacade(
        ObservabilitySettings(enabled=True, exporter="broken"), BrokenExporter()
    )
    with facade.start_trace("still-works"):
        result = 40 + 2
    assert result == 42
