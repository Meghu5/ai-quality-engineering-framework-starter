from __future__ import annotations

import httpx
import pytest

from ai_eval.models import Phase10Report
from ai_eval.report import build_correlated_phase10_report
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.providers import DeterministicLLMProvider
from clients.api_client import ApiClient
from observability.config import ObservabilitySettings
from observability.exporters import InMemoryExporter
from observability.models import FailureCategory
from observability.tracing import TracingFacade
from rag_quality.pipeline import DeterministicRagPipeline, load_rag_cases


pytestmark = pytest.mark.observability


def test_http_request_exports_safe_span_and_propagates_headers(
    observability_tracer, observability_exporter
):
    captured_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return httpx.Response(201, json={"ok": True})

    client = ApiClient(
        "https://api.example.test",
        auth_token="bearer-secret",
        transport=httpx.MockTransport(handler),
        tracer=observability_tracer,
    )
    try:
        response = client.post("/v1/bookings?private=value", json={"password": "hidden"})
    finally:
        client.close()

    assert response.status_code == 201
    span = observability_exporter.spans[0]
    assert span.operation_name == "http.post"
    assert span.attributes["http_method"] == "POST"
    assert span.attributes["http_host"] == "api.example.test"
    assert span.attributes["http_path"] == "/v1/bookings"
    assert span.attributes["status_code"] == 201
    assert span.duration_ms >= 0
    assert span.trace_id == captured_headers["x-trace-id"]
    assert span.correlation_id == captured_headers["x-correlation-id"]
    serialized = str(span.model_dump()).lower()
    assert "bearer-secret" not in serialized
    assert "password" not in serialized
    assert "private=value" not in serialized


def test_http_preserves_explicit_trace_headers(
    observability_tracer, observability_exporter
):
    captured_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return httpx.Response(200)

    client = ApiClient(
        "https://api.example.test",
        transport=httpx.MockTransport(handler),
        tracer=observability_tracer,
    )
    try:
        client.get(
            "/health",
            headers={
                "X-Trace-ID": "caller-trace",
                "X-Correlation-ID": "caller-correlation",
            },
        )
    finally:
        client.close()

    assert captured_headers["x-trace-id"] == "caller-trace"
    assert captured_headers["x-correlation-id"] == "caller-correlation"
    assert observability_exporter.spans[0].correlation_id == "caller-correlation"


def test_http_exception_is_preserved_and_classified(
    observability_tracer, observability_exporter
):
    original = httpx.TimeoutException("timeout")

    def handler(request: httpx.Request) -> httpx.Response:
        original.request = request
        raise original

    client = ApiClient(
        "https://api.example.test",
        transport=httpx.MockTransport(handler),
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(httpx.TimeoutException) as captured:
            client.get("/slow")
    finally:
        client.close()

    assert captured.value is original
    assert observability_exporter.spans[0].failure_category == FailureCategory.TIMEOUT


def test_http_disabled_mode_exports_nothing():
    exporter = InMemoryExporter()
    tracer = TracingFacade(ObservabilitySettings(enabled=False), exporter)
    client = ApiClient(
        "https://api.example.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        tracer=tracer,
    )
    try:
        assert client.get("/health").status_code == 200
    finally:
        client.close()
    assert exporter.spans == []


def test_llm_invocation_exports_only_safe_metadata(
    observability_tracer, observability_exporter
):
    raw_prompt = "Contact private.person@example.com about booking ABC123"
    provider = DeterministicLLMProvider(tracer=observability_tracer)
    prompt = PromptRegistry().get("airline_assistant", "v1")
    response = provider.generate_structured(raw_prompt, prompt=prompt)

    span = observability_exporter.spans[0]
    assert span.operation_name == "llm.generation"
    assert span.attributes["provider_name"] == "deterministic"
    assert span.attributes["model_name"] == "rule-based-airline-assistant"
    assert span.attributes["prompt_length"] == len(raw_prompt)
    assert span.attributes["response_length"] == len(response.response)
    serialized = str(span.model_dump())
    assert raw_prompt not in serialized
    assert response.response not in serialized
    assert "private.person@example.com" not in serialized


def test_llm_exception_is_preserved_and_classified(
    observability_tracer, observability_exporter
):
    original = RuntimeError("model unavailable")

    class FailingProvider(DeterministicLLMProvider):
        def _generate_structured(self, *args, **kwargs):
            raise original

    provider = FailingProvider(tracer=observability_tracer)
    prompt = PromptRegistry().get("airline_assistant", "v1")
    with pytest.raises(RuntimeError) as captured:
        provider.generate_structured("safe test input", prompt=prompt)

    assert captured.value is original
    assert observability_exporter.spans[0].failure_category == FailureCategory.MODEL
    assert "model unavailable" not in str(observability_exporter.spans[0].model_dump())


def test_llm_disabled_mode_exports_nothing():
    exporter = InMemoryExporter()
    tracer = TracingFacade(ObservabilitySettings(enabled=False), exporter)
    provider = DeterministicLLMProvider(tracer=tracer)
    prompt = PromptRegistry().get("airline_assistant", "v1")
    assert provider.generate("Find a flight", prompt=prompt)
    assert exporter.spans == []


def test_rag_pipeline_exports_expected_span_tree(
    observability_tracer, observability_exporter
):
    case = next(item for item in load_rag_cases() if item.case_id == "rag-001")
    answer = DeterministicRagPipeline(tracer=observability_tracer).answer_case(case)

    spans = observability_exporter.spans
    by_name = {span.operation_name: span for span in spans}
    assert set(by_name) == {
        "rag.operation",
        "rag.retrieval",
        "llm.generation",
        "ai.evaluation",
    }
    parent = by_name["rag.operation"]
    for name in ("rag.retrieval", "llm.generation", "ai.evaluation"):
        assert by_name[name].trace_id == parent.trace_id
        assert by_name[name].parent_span_id == parent.span_id

    retrieval = by_name["rag.retrieval"]
    assert retrieval.attributes["case_id"] == case.case_id
    assert retrieval.attributes["retrieval_count"] > 0
    assert "AIR-BAG-001" in retrieval.attributes["document_id"]
    assert retrieval.attributes["chunk_id"]
    assert by_name["llm.generation"].attributes["answer_length"] == len(answer.answer)
    assert by_name["ai.evaluation"].attributes["evaluation_status"] == "passed"
    serialized = str([span.model_dump() for span in spans])
    assert case.question not in serialized
    assert answer.answer not in serialized
    assert not any(item.chunk.content in serialized for item in answer.retrieved_context)


def test_rag_disabled_mode_preserves_behavior_without_spans():
    exporter = InMemoryExporter()
    tracer = TracingFacade(ObservabilitySettings(enabled=False), exporter)
    case = next(item for item in load_rag_cases() if item.case_id == "rag-001")
    answer = DeterministicRagPipeline(tracer=tracer).answer_case(case)
    assert answer.case_id == "rag-001"
    assert exporter.spans == []


def test_phase10_correlation_is_outside_existing_contract(
    observability_tracer, observability_exporter
):
    evidence = build_correlated_phase10_report(tracer=observability_tracer)

    assert isinstance(evidence.report, Phase10Report)
    assert evidence.trace_id == observability_exporter.spans[0].trace_id
    assert evidence.correlation_id == observability_exporter.spans[0].correlation_id
    assert "trace_id" not in Phase10Report.model_fields
    assert "correlation_id" not in Phase10Report.model_fields
    assert all(
        framework.status
        in {
            "executed",
            "not_executed",
            "skipped",
            "unavailable",
            "provider_required",
            "failed",
        }
        for framework in evidence.report.frameworks
    )
