from __future__ import annotations

import json

import httpx
import pytest

from ai_quality.models import AirlineAssistantResponse
from ai_quality.prompt_registry import Prompt
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.providers import (
    OptionalRealLLMProvider,
    RealProviderDisabledError,
    RealProviderHTTPError,
    RealProviderResponseError,
)
from observability.config import ObservabilitySettings
from observability.context import use_context
from observability.exporters import InMemoryExporter
from observability.models import FailureCategory
from observability.tracing import TracingFacade


pytestmark = pytest.mark.real_llm

API_KEY = "test-key-must-never-appear"
RAW_PROMPT = "private.person@example.test needs flight help"
RAW_RESPONSE_SECRET = "response-secret-must-never-appear"
TRACE_ID = "1" * 32
CORRELATION_ID = "safe-correlation"


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(
        name="airline_assistant",
        version="v1",
        purpose="test",
        text="Return the airline response contract.",
    )


def _settings(**overrides) -> RealLLMProviderSettings:
    values = {
        "enabled": True,
        "base_url": "https://llm.example.test/v1/chat/completions",
        "model": "airline-model-v1",
        "timeout_seconds": 10.0,
        "max_attempts": 3,
        "require_structured_output": True,
        "required": False,
    }
    values.update(overrides)
    return RealLLMProviderSettings(**values)


def _assistant_payload(response: str = "I can help with flights.") -> dict:
    return {
        "intent": "flight_search",
        "response": response,
        "entities": {"destination": "LHR"},
        "actions": [],
        "grounding": {"context_ids": [], "grounded": True},
        "safety": {
            "safe": True,
            "refusal": False,
            "pii_detected": False,
            "pii_redacted": True,
            "prompt_injection_detected": False,
        },
        "prompt_version": "v1",
    }


def _provider_response(content=None, *, token_count: int = 12) -> httpx.Response:
    resolved = json.dumps(_assistant_payload()) if content is None else content
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": resolved}}],
            "usage": {"total_tokens": token_count},
        },
    )


def _provider(handler, *, tracer: TracingFacade | None = None):
    return OptionalRealLLMProvider(
        settings=_settings(),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        tracer=tracer,
        sleeper=lambda delay: None,
    )


def test_provider_is_disabled_by_default(prompt):
    provider = OptionalRealLLMProvider(
        settings=RealLLMProviderSettings(),
        transport=httpx.MockTransport(
            lambda request: pytest.fail("disabled provider attempted network access")
        ),
    )
    try:
        assert provider.health_check() is False
        with pytest.raises(RealProviderDisabledError, match="disabled"):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()


def test_enabled_provider_accepts_valid_configuration():
    provider = _provider(lambda request: _provider_response())
    try:
        assert provider.health_check() is True
        assert provider.settings.safe_dict() == {
            "enabled": True,
            "base_url": "https://llm.example.test/v1/chat/completions",
            "model": "airline-model-v1",
            "timeout_seconds": 10.0,
            "max_attempts": 3,
            "require_structured_output": True,
            "required": False,
        }
    finally:
        provider.close()


@pytest.mark.parametrize(
    ("settings", "api_key", "message"),
    [
        (_settings(), "", "AI_REAL_PROVIDER_API_KEY"),
        (_settings(model=""), API_KEY, "AI_REAL_PROVIDER_MODEL"),
        (_settings(base_url=""), API_KEY, "AI_REAL_PROVIDER_BASE_URL"),
        (_settings(timeout_seconds=0), API_KEY, "AI_REAL_PROVIDER_TIMEOUT_SECONDS"),
        (_settings(timeout_seconds=121), API_KEY, "AI_REAL_PROVIDER_TIMEOUT_SECONDS"),
    ],
    ids=["missing-key", "missing-model", "missing-url", "zero-timeout", "large-timeout"],
)
def test_invalid_configuration_is_rejected_before_execution(
    settings, api_key, message
):
    with pytest.raises(ValueError, match=message):
        OptionalRealLLMProvider(settings=settings, api_key=api_key)


def test_environment_configuration_does_not_retain_api_key():
    environment = {
        "AI_REAL_PROVIDER_ENABLED": "true",
        "AI_REAL_PROVIDER_BASE_URL": "https://llm.example.test/v1/chat/completions",
        "AI_REAL_PROVIDER_API_KEY": API_KEY,
        "AI_REAL_PROVIDER_MODEL": "airline-model-v1",
        "AI_REAL_PROVIDER_TIMEOUT_SECONDS": "15",
        "AI_REAL_PROVIDER_MAX_ATTEMPTS": "3",
        "AI_REAL_PROVIDER_REQUIRE_STRUCTURED_OUTPUT": "true",
        "AI_REAL_PROVIDER_REQUIRED": "false",
    }
    provider = OptionalRealLLMProvider(
        environment=environment,
        transport=httpx.MockTransport(lambda request: _provider_response()),
    )
    try:
        serialized = json.dumps(provider.settings.safe_dict(), sort_keys=True)
        assert API_KEY not in serialized
        assert "api_key" not in serialized.lower()
    finally:
        provider.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("false", False)],
)
def test_required_environment_flag_accepts_explicit_booleans(value, expected):
    settings = RealLLMProviderSettings.from_env(
        {"AI_REAL_PROVIDER_REQUIRED": value}
    )
    assert settings.required is expected


@pytest.mark.parametrize("value", ["tru", "yesplease", "enabled", "", "1", "0"])
def test_required_environment_flag_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="AI_REAL_PROVIDER_REQUIRED"):
        RealLLMProviderSettings.from_env({"AI_REAL_PROVIDER_REQUIRED": value})


def test_successful_mocked_provider_response(prompt):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return _provider_response()

    provider = _provider(handler)
    try:
        result = provider.generate_structured(
            "Find a flight to London", prompt=prompt, context="Safe policy context"
        )
    finally:
        provider.close()

    assert isinstance(result, AirlineAssistantResponse)
    assert result.intent == "flight_search"
    assert captured["payload"]["model"] == "airline-model-v1"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["authorization"] == f"Bearer {API_KEY}"


def test_generate_returns_contract_response_text(prompt):
    provider = _provider(lambda request: _provider_response())
    try:
        assert provider.generate("Find a flight", prompt=prompt) == "I can help with flights."
    finally:
        provider.close()


def test_malformed_provider_json_is_classified_as_provider_failure(
    prompt, observability_tracer, observability_exporter
):
    provider = _provider(
        lambda request: httpx.Response(
            200, content=b"not-json", headers={"content-type": "application/json"}
        ),
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(RealProviderResponseError, match="envelope is malformed"):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert observability_exporter.spans[0].failure_category == FailureCategory.PROVIDER


@pytest.mark.parametrize("content", ["not-json", json.dumps({"intent": "missing-fields"})])
def test_invalid_structured_response_is_classified_as_contract_failure(
    content, prompt, observability_tracer, observability_exporter
):
    provider = _provider(
        lambda request: _provider_response(content), tracer=observability_tracer
    )
    try:
        with pytest.raises(RealProviderResponseError, match="contract validation"):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert observability_exporter.spans[0].failure_category == FailureCategory.CONTRACT


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, FailureCategory.AUTHENTICATION),
        (403, FailureCategory.AUTHORIZATION),
        (429, FailureCategory.RATE_LIMIT),
        (500, FailureCategory.PROVIDER),
    ],
)
def test_http_errors_have_stable_safe_mapping(
    status_code, category, prompt, observability_tracer, observability_exporter
):
    provider = _provider(
        lambda request: httpx.Response(status_code, text=RAW_RESPONSE_SECRET),
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(RealProviderHTTPError) as captured:
            provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()
    assert captured.value.status_code == status_code
    assert API_KEY not in str(captured.value)
    assert RAW_PROMPT not in str(captured.value)
    assert RAW_RESPONSE_SECRET not in str(captured.value)
    assert observability_exporter.spans[0].failure_category == category


def test_api_key_prompt_and_response_are_absent_from_trace_evidence(
    prompt, observability_tracer, observability_exporter
):
    provider = _provider(
        lambda request: _provider_response(
            json.dumps(_assistant_payload(RAW_RESPONSE_SECRET))
        ),
        tracer=observability_tracer,
    )
    try:
        result = provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()

    span = observability_exporter.spans[0]
    serialized = json.dumps(span.model_dump(mode="json"), sort_keys=True)
    assert span.attributes == {
        "latency_ms": span.attributes["latency_ms"],
        "model_name": "airline-model-v1",
        "operation": "generate_structured",
        "prompt_length": len(RAW_PROMPT),
        "provider_name": "http-json",
        "response_length": len(json.dumps(_assistant_payload(RAW_RESPONSE_SECRET))),
        "status_code": 200,
        "timeout_seconds": 10.0,
        "token_count": 12,
    }
    for sensitive in (API_KEY, RAW_PROMPT, RAW_RESPONSE_SECRET, "private.person"):
        assert sensitive not in serialized
    assert result.response == RAW_RESPONSE_SECRET


def test_trace_context_is_propagated_to_provider_request(
    prompt, observability_tracer, observability_exporter
):
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return _provider_response()

    provider = _provider(handler, tracer=observability_tracer)
    try:
        with use_context(trace_id=TRACE_ID, correlation_id=CORRELATION_ID):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()

    span = observability_exporter.spans[0]
    assert captured_headers["x-trace-id"] == TRACE_ID
    assert captured_headers["x-correlation-id"] == CORRELATION_ID
    assert span.trace_id == TRACE_ID
    assert span.correlation_id == CORRELATION_ID


def test_safe_configuration_and_response_serialization_are_deterministic(prompt):
    first = json.dumps(_settings().safe_dict(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(_settings().safe_dict(), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert API_KEY not in first

    provider = _provider(lambda request: _provider_response())
    try:
        first_response = provider.generate_structured("hello", prompt=prompt)
        second_response = provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert first_response.model_dump_json() == second_response.model_dump_json()


def test_injected_client_is_not_closed_by_provider(prompt):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: _provider_response()))
    provider = OptionalRealLLMProvider(
        settings=_settings(), api_key=API_KEY, client=client
    )
    provider.close()
    try:
        assert provider.generate_structured("hello", prompt=prompt).intent == "flight_search"
    finally:
        client.close()


def test_disabled_provider_exports_no_sensitive_metadata(prompt):
    exporter = InMemoryExporter()
    tracer = TracingFacade(ObservabilitySettings(enabled=True, exporter="memory"), exporter)
    provider = OptionalRealLLMProvider(
        settings=RealLLMProviderSettings(),
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: _provider_response()),
        tracer=tracer,
    )
    try:
        with pytest.raises(RealProviderDisabledError):
            provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()
    serialized = json.dumps(exporter.spans[0].model_dump(mode="json"), sort_keys=True)
    assert API_KEY not in serialized
    assert RAW_PROMPT not in serialized
