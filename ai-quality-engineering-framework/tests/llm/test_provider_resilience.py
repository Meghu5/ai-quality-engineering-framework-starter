from __future__ import annotations

import json
import traceback

import httpx
import pytest

from ai_quality.prompt_registry import Prompt
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.provider_resilience import (
    ProviderFailureMetadata,
    TRANSIENT_HTTP_STATUSES,
    classify_http_failure,
    classify_transport_failure,
)
from ai_quality.providers import (
    DeterministicLLMProvider,
    LLMProvider,
    OptionalRealLLMProvider,
    RealProviderError,
    RealProviderHTTPError,
    RealProviderResponseError,
    RealProviderTransportError,
)
from observability.context import use_context
from observability.models import FailureCategory


pytestmark = pytest.mark.real_llm

API_KEY = "resilience-key-must-never-appear"
RAW_PROMPT = "private.retry@example.test retry this"
RAW_RESPONSE = "raw-provider-secret"


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(
        name="airline_assistant",
        version="v1",
        purpose="test",
        text="Return the airline response contract.",
    )


def _settings(*, max_attempts: int = 3) -> RealLLMProviderSettings:
    return RealLLMProviderSettings(
        enabled=True,
        base_url="https://llm.example.test/v1/chat/completions",
        model="airline-model-v1",
        timeout_seconds=10.0,
        max_attempts=max_attempts,
        require_structured_output=True,
    )


def _success_response() -> httpx.Response:
    content = {
        "intent": "flight_search",
        "response": "I can help with flights.",
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
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"total_tokens": 12},
        },
    )


def _scripted_handler(outcomes, calls, headers=None):
    scripted = iter(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if headers is not None:
            headers.append(dict(request.headers))
        outcome = next(scripted)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return handler


def _provider(
    outcomes,
    *,
    max_attempts=3,
    calls=None,
    delays=None,
    tracer=None,
    headers=None,
):
    recorded_calls = calls if calls is not None else []
    recorded_delays = delays if delays is not None else []
    return OptionalRealLLMProvider(
        settings=_settings(max_attempts=max_attempts),
        api_key=API_KEY,
        transport=httpx.MockTransport(
            _scripted_handler(outcomes, recorded_calls, headers=headers)
        ),
        tracer=tracer,
        sleeper=recorded_delays.append,
    )


def test_max_attempts_one_performs_no_retry(prompt):
    calls = []
    provider = _provider([httpx.Response(429)], max_attempts=1, calls=calls)
    try:
        with pytest.raises(RealProviderHTTPError) as captured:
            provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()
    assert len(calls) == 1
    assert captured.value.failure == ProviderFailureMetadata(
        category=FailureCategory.RATE_LIMIT,
        retryable=True,
        status_code=429,
        attempt=1,
        max_attempts=1,
    )


def test_429_then_success_retries_once(prompt):
    calls = []
    delays = []
    provider = _provider(
        [httpx.Response(429), _success_response()], calls=calls, delays=delays
    )
    try:
        result = provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()
    assert result.intent == "flight_search"
    assert len(calls) == 2
    assert delays == [0.25]


def test_repeated_429_stops_at_max_attempts(prompt, observability_tracer, observability_exporter):
    calls = []
    provider = _provider(
        [httpx.Response(429), httpx.Response(429), httpx.Response(429)],
        calls=calls,
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(RealProviderHTTPError) as captured:
            provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()
    assert len(calls) == 3
    assert captured.value.failure.attempt == 3
    assert captured.value.failure.category == FailureCategory.RATE_LIMIT
    assert observability_exporter.spans[0].failure_category == FailureCategory.RATE_LIMIT


@pytest.mark.parametrize("status_code", sorted(TRANSIENT_HTTP_STATUSES))
def test_transient_5xx_then_success_retries(status_code, prompt):
    calls = []
    provider = _provider(
        [httpx.Response(status_code), _success_response()], calls=calls
    )
    try:
        assert provider.generate_structured("hello", prompt=prompt).intent == "flight_search"
    finally:
        provider.close()
    assert len(calls) == 2


def test_repeated_transient_5xx_is_bounded(prompt):
    calls = []
    provider = _provider(
        [httpx.Response(503), httpx.Response(503)], max_attempts=2, calls=calls
    )
    try:
        with pytest.raises(RealProviderHTTPError) as captured:
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert len(calls) == 2
    assert captured.value.failure == ProviderFailureMetadata(
        category=FailureCategory.PROVIDER,
        retryable=True,
        status_code=503,
        attempt=2,
        max_attempts=2,
    )


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, FailureCategory.AUTHENTICATION),
        (403, FailureCategory.AUTHORIZATION),
        (400, FailureCategory.CONTRACT),
        (404, FailureCategory.CONTRACT),
        (501, FailureCategory.PROVIDER),
    ],
)
def test_permanent_http_failures_are_not_retried(status_code, category, prompt):
    calls = []
    provider = _provider([httpx.Response(status_code)], calls=calls)
    try:
        with pytest.raises(RealProviderHTTPError) as captured:
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert len(calls) == 1
    assert captured.value.failure.category == category
    assert captured.value.failure.retryable is False


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}]},
        ),
    ],
)
def test_response_and_contract_failures_are_not_retried(response, prompt):
    calls = []
    provider = _provider([response], calls=calls)
    try:
        with pytest.raises(RealProviderResponseError) as captured:
            provider.generate_structured(RAW_PROMPT, prompt=prompt)
    finally:
        provider.close()
    assert len(calls) == 1
    assert captured.value.failure.retryable is False


def test_timeout_then_success_preserves_retry_behavior(prompt):
    calls = []
    timeout = httpx.ReadTimeout("sensitive text is ignored")
    provider = _provider([timeout, _success_response()], calls=calls)
    try:
        assert provider.generate_structured("hello", prompt=prompt).intent == "flight_search"
    finally:
        provider.close()
    assert len(calls) == 2


def test_repeated_timeout_is_bounded_and_returns_safe_provider_error(
    prompt, observability_tracer, observability_exporter
):
    calls = []
    final_timeout = httpx.ReadTimeout("must-not-drive-classification")
    provider = _provider(
        [httpx.ReadTimeout("first"), final_timeout],
        max_attempts=2,
        calls=calls,
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(RealProviderTransportError) as captured:
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert captured.value.failure.category == FailureCategory.TIMEOUT
    assert captured.value.failure.attempt == 2
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not hasattr(captured.value, "request")
    assert len(calls) == 2
    assert observability_exporter.spans[0].failure_category == FailureCategory.TIMEOUT


def test_transient_network_failure_then_success_retries(prompt):
    calls = []
    provider = _provider(
        [httpx.ConnectError("message is not parsed"), _success_response()], calls=calls
    )
    try:
        assert provider.generate_structured("hello", prompt=prompt).intent == "flight_search"
    finally:
        provider.close()
    assert len(calls) == 2


def test_unknown_exception_is_not_retried_and_keeps_identity(prompt):
    calls = []
    original = RuntimeError("unknown failure")
    provider = _provider([original], calls=calls)
    try:
        with pytest.raises(RuntimeError) as captured:
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert captured.value is original
    assert len(calls) == 1


def test_non_transient_request_error_fails_closed(prompt):
    calls = []
    original = httpx.UnsupportedProtocol("unsupported")
    provider = _provider([original], calls=calls)
    try:
        with pytest.raises(RealProviderTransportError) as captured:
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert captured.value.failure.retryable is False
    assert captured.value.failure.category == FailureCategory.NETWORK
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not hasattr(captured.value, "request")
    assert len(calls) == 1


def test_injected_backoff_and_sleeper_are_deterministic(prompt):
    calls = []
    delays = []
    backoff_attempts = []

    def backoff(attempt):
        backoff_attempts.append(attempt)
        return attempt * 0.1

    provider = OptionalRealLLMProvider(
        settings=_settings(max_attempts=3),
        api_key=API_KEY,
        transport=httpx.MockTransport(
            _scripted_handler(
                [httpx.Response(503), httpx.Response(503), _success_response()], calls
            )
        ),
        sleeper=delays.append,
        backoff=backoff,
    )
    try:
        provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert backoff_attempts == [1, 2]
    assert delays == [0.1, 0.2]


def test_failure_metadata_is_safe_stable_and_message_independent():
    first = classify_http_failure(429, attempt=2, max_attempts=3)
    second = classify_http_failure(429, attempt=2, max_attempts=3)
    timeout_a = classify_transport_failure(
        httpx.ReadTimeout("retry please"), attempt=1, max_attempts=3
    )
    timeout_b = classify_transport_failure(
        httpx.ReadTimeout("do not retry"), attempt=1, max_attempts=3
    )
    serialized = json.dumps(first.safe_dict(), sort_keys=True)
    assert first == second
    assert timeout_a == timeout_b
    assert serialized == (
        '{"attempt": 2, "category": "rate_limit", "max_attempts": 3, '
        '"retryable": true, "status_code": 429}'
    )
    for sensitive in (API_KEY, RAW_PROMPT, RAW_RESPONSE, "llm.example.test"):
        assert sensitive not in serialized


def test_retry_telemetry_and_exceptions_exclude_sensitive_data(
    prompt, observability_tracer, observability_exporter
):
    provider = _provider(
        [
            httpx.Response(429, text=RAW_RESPONSE),
            httpx.Response(429, text=RAW_RESPONSE),
            httpx.Response(429, text=RAW_RESPONSE),
        ],
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(RealProviderHTTPError) as captured:
            provider.generate_structured(RAW_PROMPT, prompt=prompt, context="raw context")
    finally:
        provider.close()
    evidence = json.dumps(
        observability_exporter.spans[0].model_dump(mode="json"), sort_keys=True
    )
    diagnostic = json.dumps(captured.value.failure.safe_dict(), sort_keys=True)
    combined = f"{captured.value} {diagnostic} {evidence}"
    for sensitive in (
        API_KEY,
        RAW_PROMPT,
        RAW_RESPONSE,
        "raw context",
        "private.retry",
        "Authorization",
        "cookie",
        "llm.example.test",
    ):
        assert sensitive not in combined


def test_trace_and_correlation_context_survive_retries(prompt):
    headers = []
    provider = _provider(
        [httpx.Response(503), _success_response()], headers=headers
    )
    try:
        with use_context(trace_id="1" * 32, correlation_id="retry-correlation"):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert [header["x-trace-id"] for header in headers] == ["1" * 32, "1" * 32]
    assert [header["x-correlation-id"] for header in headers] == [
        "retry-correlation",
        "retry-correlation",
    ]


@pytest.mark.parametrize("value", ["0", "-1", "6", "1.5", "many"])
def test_invalid_retry_configuration_is_rejected(value):
    environment = {
        "AI_REAL_PROVIDER_ENABLED": "false",
        "AI_REAL_PROVIDER_MAX_ATTEMPTS": value,
    }
    with pytest.raises(ValueError, match="AI_REAL_PROVIDER_MAX_ATTEMPTS"):
        RealLLMProviderSettings.from_env(environment)


@pytest.mark.parametrize("value", [1, 3, 5])
def test_direct_retry_configuration_accepts_valid_integers(value):
    settings = RealLLMProviderSettings(max_attempts=value)
    settings.validate()
    assert settings.max_attempts == value


@pytest.mark.parametrize("value", [0, 6, -1, 1.5, True, False])
def test_direct_retry_configuration_rejects_invalid_values(value):
    settings = RealLLMProviderSettings(max_attempts=value)
    with pytest.raises(ValueError, match="AI_REAL_PROVIDER_MAX_ATTEMPTS"):
        settings.validate()


def test_retry_configuration_is_secret_free():
    settings = _settings(max_attempts=4)
    serialized = json.dumps(settings.safe_dict(), sort_keys=True)
    assert settings.max_attempts == 4
    assert API_KEY not in serialized
    assert "api_key" not in serialized.lower()


def test_provider_contract_and_deterministic_provider_remain_unchanged(prompt):
    assert isinstance(_provider([_success_response()]), LLMProvider)
    deterministic = DeterministicLLMProvider()
    assert deterministic.health_check() is True
    assert deterministic.generate("Find a flight", prompt=prompt)


def test_disabled_provider_never_calls_transport(prompt):
    provider = OptionalRealLLMProvider(
        settings=RealLLMProviderSettings(),
        transport=httpx.MockTransport(
            lambda request: pytest.fail("disabled provider attempted network access")
        ),
        sleeper=lambda delay: pytest.fail("disabled provider attempted retry sleep"),
    )
    try:
        with pytest.raises(RuntimeError, match="disabled"):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()


def test_safety_failure_is_not_retried_and_preserves_exception_identity(prompt):
    calls = []
    failure = ProviderFailureMetadata(
        category=FailureCategory.SAFETY,
        retryable=False,
        attempt=1,
        max_attempts=3,
    )
    original = RealProviderError("Safe provider failure", failure=failure)
    provider = _provider([original], calls=calls)
    try:
        with pytest.raises(RealProviderError) as captured:
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    assert captured.value is original
    assert captured.value.failure.category == FailureCategory.SAFETY
    assert len(calls) == 1


def test_final_transport_failure_clears_prior_http_status(
    prompt, observability_tracer, observability_exporter
):
    provider = _provider(
        [httpx.Response(503), httpx.ReadTimeout("final")],
        max_attempts=2,
        tracer=observability_tracer,
    )
    try:
        with pytest.raises(RealProviderTransportError):
            provider.generate_structured("hello", prompt=prompt)
    finally:
        provider.close()
    span = observability_exporter.spans[0]
    assert span.failure_category == FailureCategory.TIMEOUT
    assert span.attributes["status_code"] is None


@pytest.mark.parametrize(
    ("transport_error", "category"),
    [
        (httpx.ReadTimeout("sensitive timeout text"), FailureCategory.TIMEOUT),
        (httpx.ConnectError("sensitive network text"), FailureCategory.NETWORK),
    ],
)
def test_final_transport_exception_exposes_no_sensitive_request_data(
    transport_error, category, prompt, observability_tracer, observability_exporter
):
    endpoint_with_sensitive_path = "https://llm.example.test/private/provider/path"
    settings = RealLLMProviderSettings(
        enabled=True,
        base_url=endpoint_with_sensitive_path,
        model="airline-model-v1",
        timeout_seconds=10.0,
        max_attempts=1,
        require_structured_output=True,
    )

    def handler(request):
        raise transport_error

    provider = OptionalRealLLMProvider(
        settings=settings,
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        tracer=observability_tracer,
        sleeper=lambda delay: None,
    )
    raw_context = "private context value"
    try:
        with pytest.raises(RealProviderTransportError) as captured:
            provider.generate_structured(
                RAW_PROMPT, prompt=prompt, context=raw_context
            )
    finally:
        provider.close()

    error = captured.value
    assert error.failure.category == category
    assert error.failure.retryable is True
    assert error.failure.attempt == 1
    assert error.failure.max_attempts == 1
    assert error.failure.status_code is None
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not hasattr(error, "request")
    assert vars(error) == {"failure": error.failure}
    assert not any(
        isinstance(value, (httpx.Request, httpx.RequestError))
        for value in vars(error).values()
    )
    traceback_frames = []
    current_traceback = error.__traceback__
    while current_traceback is not None:
        traceback_frames.append(current_traceback.tb_frame)
        current_traceback = current_traceback.tb_next
    assert traceback_frames
    assert all(
        frame.f_globals.get("__name__") != "ai_quality.providers"
        for frame in traceback_frames
    )
    assert all(
        not isinstance(value, OptionalRealLLMProvider)
        for frame in traceback_frames
        if frame.f_globals.get("__name__", "").startswith("ai_quality")
        for value in frame.f_locals.values()
    )
    assert all(
        not isinstance(value, (httpx.Request, httpx.RequestError))
        for frame in traceback_frames
        if frame.f_globals.get("__name__", "").startswith("ai_quality")
        for value in frame.f_locals.values()
    )
    visible = " ".join(
        [
            str(error),
            repr(error),
            repr(error.args),
            "".join(traceback.format_exception(error)),
            json.dumps(error.failure.safe_dict(), sort_keys=True),
            json.dumps(
                observability_exporter.spans[0].model_dump(mode="json"),
                sort_keys=True,
            ),
        ]
    )
    for sensitive in (
        API_KEY,
        endpoint_with_sensitive_path,
        RAW_PROMPT,
        raw_context,
        "sensitive timeout text",
        "sensitive network text",
        "Authorization",
    ):
        assert sensitive not in visible
