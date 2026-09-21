from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from observability.exporters import JsonEvidenceExporter
from observability.models import FailureCategory, SpanEvidence, TraceEnvelope, TraceStatus


pytestmark = pytest.mark.observability


def make_span(**updates):
    values = {
        "trace_id": "1" * 32,
        "span_id": "2" * 16,
        "parent_span_id": None,
        "correlation_id": "correlation-1",
        "operation_name": "evaluate",
        "operation_type": "evaluation",
        "started_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        "duration_ms": 1000.0,
        "status": TraceStatus.ERROR,
        "failure_category": FailureCategory.MODEL,
        "attributes": {"case_id": "case-1"},
        "service_name": "test-service",
        "environment": "test",
    }
    values.update(updates)
    return SpanEvidence(**values)


def test_evidence_models_are_strict_and_serialize_failure_category():
    span = make_span()
    payload = span.model_dump(mode="json")
    assert payload["failure_category"] == "model"
    assert payload["status"] == "error"

    with pytest.raises(ValidationError):
        SpanEvidence(**span.model_dump(), unexpected=True)


def test_trace_envelope_validates_relationships():
    parent = make_span(status=TraceStatus.OK, failure_category=None)
    child = make_span(
        span_id="3" * 16,
        parent_span_id=parent.span_id,
        status=TraceStatus.OK,
        failure_category=None,
    )
    envelope = TraceEnvelope(
        trace_id=parent.trace_id,
        correlation_id=parent.correlation_id,
        service_name=parent.service_name,
        environment=parent.environment,
        spans=[parent, child],
    )
    assert envelope.spans[1].parent_span_id == parent.span_id


def test_json_export_is_deterministic_and_machine_readable(tmp_path):
    path = tmp_path / "traces.json"
    exporter = JsonEvidenceExporter(path)
    exporter.export(make_span())
    first = path.read_text(encoding="utf-8")
    exporter = JsonEvidenceExporter(path)
    exporter.export(make_span())
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first)[0]["spans"][0]["failure_category"] == "model"
