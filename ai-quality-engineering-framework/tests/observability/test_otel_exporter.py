from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

import pytest

from observability.config import ObservabilitySettings
from observability.exporters import InMemoryExporter, JsonEvidenceExporter
from observability.models import FailureCategory, SpanEvidence, TraceStatus
from observability.otel import OtlpSpanExporterAdapter, normalize_otel_attributes
from observability.tracing import create_tracing_facade


pytestmark = pytest.mark.observability


class FakeSpanProcessor:
    def __init__(self, *, fail_export=False, fail_flush=False, fail_shutdown=False):
        self.spans = []
        self.fail_export = fail_export
        self.fail_flush = fail_flush
        self.fail_shutdown = fail_shutdown

    def on_end(self, span):
        if self.fail_export:
            raise TimeoutError("synthetic timeout")
        self.spans.append(span)

    def force_flush(self, timeout_millis=None):
        if self.fail_flush:
            raise RuntimeError("synthetic flush failure")
        return True

    def shutdown(self):
        if self.fail_shutdown:
            raise RuntimeError("synthetic shutdown failure")


def _settings(**overrides):
    values = {
        "enabled": True,
        "exporter": "otlp",
        "otlp_endpoint": "http://localhost:4318/v1/traces",
    }
    values.update(overrides)
    return ObservabilitySettings(**values)


def _evidence(**overrides):
    started_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    values = {
        "trace_id": "1" * 32,
        "span_id": "2" * 16,
        "parent_span_id": "3" * 16,
        "correlation_id": "correlation-123",
        "operation_name": "rag.retrieval",
        "operation_type": "retrieval",
        "started_at": started_at,
        "ended_at": started_at + timedelta(milliseconds=25),
        "duration_ms": 25,
        "status": TraceStatus.OK,
        "attributes": {"case_id": "case-1", "retrieval_count": 2},
        "service_name": "quality-service",
        "environment": "staging",
    }
    values.update(overrides)
    return SpanEvidence(**values)


def test_disabled_mode_does_not_import_or_export_otel(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "observability.otel", None)
    facade = create_tracing_facade(ObservabilitySettings())
    assert facade.enabled is False
    with facade.start_trace("disabled"):
        pass


@pytest.mark.parametrize(
    ("choice", "expected_type"),
    [("none", type(None)), ("memory", InMemoryExporter), ("json", JsonEvidenceExporter)],
)
def test_exporter_selection_for_non_otlp_modes(choice, expected_type):
    facade = create_tracing_facade(ObservabilitySettings(enabled=True, exporter=choice))
    assert isinstance(facade.exporter, expected_type)


def test_otlp_exporter_selection(monkeypatch):
    marker = object()
    monkeypatch.setattr("observability.otel.OtlpSpanExporterAdapter", lambda settings: marker)
    assert create_tracing_facade(_settings()).exporter is marker


def test_otel_initialization_failure_is_fail_open(monkeypatch):
    def fail(_settings):
        raise RuntimeError("synthetic initialization failure")

    monkeypatch.setattr("observability.otel.OtlpSpanExporterAdapter", fail)
    assert create_tracing_facade(_settings()).enabled is False


def test_trace_resource_metadata_and_timestamps_are_mapped():
    processor = FakeSpanProcessor()
    adapter = OtlpSpanExporterAdapter(_settings(), span_processor=processor)
    evidence = _evidence()
    adapter.export(evidence)

    exported = processor.spans[0]
    assert exported.context.trace_id == int(evidence.trace_id, 16)
    assert exported.context.span_id == int(evidence.span_id, 16)
    assert exported.parent.trace_id == int(evidence.trace_id, 16)
    assert exported.parent.span_id == int(evidence.parent_span_id, 16)
    assert exported.name == evidence.operation_name
    assert exported.start_time == int(evidence.started_at.timestamp() * 1_000_000_000)
    assert exported.end_time == int(evidence.ended_at.timestamp() * 1_000_000_000)
    assert exported.resource.attributes["service.name"] == "quality-service"
    assert exported.resource.attributes["deployment.environment.name"] == "staging"
    assert exported.attributes["ai.observability.operation_type"] == "retrieval"
    assert exported.attributes["ai.observability.correlation_id"] == "correlation-123"
    assert exported.attributes["case_id"] == "case-1"


def test_sdk_batch_export_preserves_framework_ids_end_to_end():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter as OtelInMemorySpanExporter,
    )

    otel_exporter = OtelInMemorySpanExporter()
    adapter = OtlpSpanExporterAdapter(_settings(), otel_exporter=otel_exporter)
    evidence = _evidence()
    adapter.export(evidence)
    assert adapter.force_flush()

    exported = otel_exporter.get_finished_spans()[0]
    assert exported.context.trace_id == int(evidence.trace_id, 16)
    assert exported.context.span_id == int(evidence.span_id, 16)
    assert exported.parent.span_id == int(evidence.parent_span_id, 16)
    adapter.shutdown()


@pytest.mark.parametrize(
    ("framework_status", "otel_status"),
    [(TraceStatus.OK, "OK"), (TraceStatus.ERROR, "ERROR"), (TraceStatus.UNSET, "UNSET")],
)
def test_status_mapping(framework_status, otel_status):
    processor = FakeSpanProcessor()
    adapter = OtlpSpanExporterAdapter(_settings(), span_processor=processor)
    adapter.export(_evidence(status=framework_status))
    assert processor.spans[0].status.status_code.name == otel_status


def test_failure_category_and_safe_exception_type_are_mapped():
    processor = FakeSpanProcessor()
    adapter = OtlpSpanExporterAdapter(_settings(), span_processor=processor)
    adapter.export(_evidence(status=TraceStatus.ERROR, failure_category=FailureCategory.TIMEOUT, attributes={"exception_type": "TimeoutError"}))
    exported = processor.spans[0]
    assert exported.attributes["ai.observability.failure_category"] == "timeout"
    assert "exception_type" not in exported.attributes
    assert exported.events[0].name == "exception"
    assert exported.events[0].attributes == {"exception.type": "TimeoutError"}


def test_attribute_normalization_accepts_only_supported_safe_values():
    class Unsafe:
        def __str__(self):
            return "password=do-not-export"

    normalized = normalize_otel_attributes({
        "case_id": "case-1", "retrieval_count": 2, "latency_ms": 1.5,
        "operation": True, "chunk_id": ["a", "b"],
        "document_id": [1, "mixed"], "model_name": {"nested": "secret"},
        "provider_name": b"bytes", "exception_type": Unsafe(),
        "authorization": "Bearer token",
    })
    assert normalized == {
        "case_id": "case-1", "retrieval_count": 2, "latency_ms": 1.5,
        "operation": True, "chunk_id": ("a", "b"),
    }


def test_sensitive_content_and_pii_are_absent_from_export():
    processor = FakeSpanProcessor()
    adapter = OtlpSpanExporterAdapter(_settings(), span_processor=processor)
    adapter.export(_evidence(attributes={
        "authorization": "Bearer abc123", "api_key": "secret-key",
        "password": "password=hidden", "cookie": "session=hidden",
        "prompt": "raw prompt", "context": "raw rag context",
        "response": "raw model response", "content": "request body",
        "email": "person@example.com", "phone": "+1 202 555 0123",
        "passport": "PASSPORT-A1234567", "card": "4111 1111 1111 1111",
        "case_id": "person@example.com Bearer abc123",
    }))
    serialized = str(dict(processor.spans[0].attributes)).lower()
    for forbidden in (
        "bearer", "abc123", "secret-key", "password=", "session=", "raw prompt",
        "raw rag context", "raw model response", "person@example.com", "+1 202",
        "a1234567", "4111",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("failure", ["export", "flush", "shutdown"])
def test_processor_failures_are_fail_open(failure):
    processor = FakeSpanProcessor(
        fail_export=failure == "export", fail_flush=failure == "flush",
        fail_shutdown=failure == "shutdown",
    )
    adapter = OtlpSpanExporterAdapter(_settings(), span_processor=processor)
    adapter.export(_evidence())
    assert adapter.force_flush() is (failure != "flush")
    adapter.shutdown()


def test_shutdown_is_bounded_when_processor_hangs():
    class HangingProcessor(FakeSpanProcessor):
        def shutdown(self):
            time.sleep(1)

    adapter = OtlpSpanExporterAdapter(
        _settings(otlp_timeout_seconds=0.01), span_processor=HangingProcessor()
    )
    started = time.perf_counter()
    adapter.shutdown()
    assert time.perf_counter() - started < 0.5


def test_existing_facade_behavior_is_preserved_with_otel_adapter():
    processor = FakeSpanProcessor()
    adapter = OtlpSpanExporterAdapter(_settings(), span_processor=processor)
    facade = create_tracing_facade(ObservabilitySettings(enabled=False))
    facade.settings = _settings()
    facade.exporter = adapter

    with facade.start_trace("rag.operation", operation_type="rag") as root:
        with facade.start_span("llm.generation", operation_type="llm"):
            pass

    child, parent = processor.spans
    assert child.context.trace_id == parent.context.trace_id
    assert child.parent.span_id == parent.context.span_id
    assert root.evidence is not None
