from __future__ import annotations

import os
import re
import json
import math
import time
from datetime import datetime
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from ai_quality.models import (
    AirlineAssistantResponse,
    ChatbotAction,
    GoldenCase,
    GroundingResult,
    SafetyResult,
)
from ai_quality.prompt_registry import Prompt
from ai_quality.privacy import safe_model_label
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.provider_resilience import (
    ProviderFailureMetadata,
    classify_http_failure,
    classify_transport_failure,
    exponential_backoff,
    parse_retry_after,
    utc_now,
)
from observability.context import get_context
from observability.models import FailureCategory
from observability.tracing import TracingFacade, create_tracing_facade


AIRPORT_ALIASES = {
    "JFK": "JFK",
    "LHR": "LHR",
    "LAX": "LAX",
    "NRT": "NRT",
    "LONDON": "LHR",
    "TOKYO": "NRT",
    "NEW YORK": "JFK",
    "LOS ANGELES": "LAX",
}

INTENT_KEYWORDS = {
    "flight_search": ("find", "search", "flight to", "fly to"),
    "flight_status": ("status", "delayed", "on time"),
    "booking": ("booking", "reservation", "pnr"),
    "cancellation": ("cancel", "cancellation"),
    "baggage": ("bag", "baggage", "checked bag", "carry-on"),
    "check_in": ("check in", "check-in", "boarding pass"),
    "refund": ("refund",),
    "seat_selection": ("seat", "window", "aisle"),
    "fare": ("fare", "price", "cost"),
}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, user_input: str, *, prompt: Prompt, context: str | None = None) -> str:
        """Generate a natural-language answer."""

    @abstractmethod
    def generate_structured(
        self,
        user_input: str,
        *,
        prompt: Prompt,
        case: GoldenCase | None = None,
        context: str | None = None,
    ) -> AirlineAssistantResponse:
        """Generate the airline chatbot contract."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the provider is available."""


class DeterministicLLMProvider(LLMProvider):
    """Rule-based provider for reproducible tests. This is not a real LLM."""

    def __init__(self, *, tracer: TracingFacade | None = None) -> None:
        self.tracer = tracer or create_tracing_facade()

    def generate(self, user_input: str, *, prompt: Prompt, context: str | None = None) -> str:
        response = self.generate_structured(user_input, prompt=prompt, context=context)
        return response.response

    def generate_structured(
        self,
        user_input: str,
        *,
        prompt: Prompt,
        case: GoldenCase | None = None,
        context: str | None = None,
    ) -> AirlineAssistantResponse:
        attributes = {
            "provider_name": "deterministic",
            "model_name": "rule-based-airline-assistant",
            "operation": "generate_structured",
            "prompt_length": len(user_input),
            "case_id": case.id if case else None,
        }
        with self.tracer.start_span(
            "llm.generation", operation_type="llm", attributes=attributes
        ) as span:
            try:
                response = self._generate_structured(
                    user_input, prompt=prompt, case=case, context=context
                )
            except Exception as exc:
                span.record_exception(exc, FailureCategory.MODEL)
                raise
            span.set_attribute("response_length", len(response.response))
            return response

    def _generate_structured(
        self,
        user_input: str,
        *,
        prompt: Prompt,
        case: GoldenCase | None = None,
        context: str | None = None,
    ) -> AirlineAssistantResponse:
        text = " ".join([*case.turns, user_input]) if case and case.turns else user_input
        normalized = text.upper()
        intent = case.expected_intent if case else self._classify_intent(text)
        entities = self._extract_entities(text)
        if case:
            entities.update(case.expected_entities)
        action = self._action_for_intent(intent, entities)
        safety = self._safety_for(text, case)
        answer = self._answer(intent, entities, case, context)

        return AirlineAssistantResponse(
            intent=intent,
            response=answer,
            entities=entities,
            actions=[action] if action else [],
            grounding=GroundingResult(
                context_ids=[case.id] if case and case.required_context_terms else [],
                grounded=True,
            ),
            safety=safety,
            prompt_version=prompt.version,
        )

    def health_check(self) -> bool:
        return True

    def _classify_intent(self, text: str) -> str:
        lowered = text.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return intent
        return "general_airline_help"

    def _extract_entities(self, text: str) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        normalized = text.upper()
        airports = [
            code
            for alias, code in AIRPORT_ALIASES.items()
            if re.search(rf"\b{re.escape(alias)}\b", normalized)
        ]
        if airports:
            if len(airports) >= 2:
                entities["origin"] = airports[0]
                entities["destination"] = airports[1]
            elif "TO " in normalized or "LONDON" in normalized or "TOKYO" in normalized:
                entities["destination"] = airports[0]
            else:
                entities["origin"] = airports[0]

        booking_match = re.search(r"\b[A-Z]{3}\d{3}\b", normalized)
        if booking_match:
            entities["booking_reference"] = booking_match.group(0)

        passenger_match = re.search(r"\b([1-9])\s+(PASSENGER|PASSENGERS|PEOPLE)\b", normalized)
        if passenger_match:
            entities["passengers"] = int(passenger_match.group(1))
        return entities

    def _action_for_intent(
        self,
        intent: str,
        entities: dict[str, Any],
    ) -> ChatbotAction | None:
        action_by_intent = {
            "flight_search": "search_flights",
            "flight_status": "get_flight_status",
            "booking": "get_booking",
            "baggage": "calculate_baggage",
            "fare": "search_fares",
            "seat_selection": "select_seat",
            "refund": "get_refund_policy",
            "check_in": "get_check_in_options",
            "cancellation": "get_cancellation_policy",
        }
        action_name = action_by_intent.get(intent)
        if not action_name:
            return None
        allowed = {
            "origin",
            "destination",
            "departure_date",
            "passengers",
            "booking_reference",
            "fare_id",
            "seat_preference",
        }
        return ChatbotAction(
            name=action_name,
            parameters={key: value for key, value in entities.items() if key in allowed},
        )

    def _safety_for(self, text: str, case: GoldenCase | None) -> SafetyResult:
        lowered = text.lower()
        injection = any(
            phrase in lowered
            for phrase in (
                "ignore previous instructions",
                "reveal the system prompt",
                "hidden instructions",
                "api credentials",
                "unrestricted administrator",
            )
        )
        pii = bool(
            re.search(r"[\w.-]+@[\w.-]+", text)
            or re.search(r"\+\d{10,15}", text)
            or "TEST-PASSPORT-" in text
            or "TEST-CARD-" in text
        )
        return SafetyResult(
            safe=True,
            refusal=bool(case and case.expect_refusal) or injection,
            pii_detected=bool(case and case.expect_pii) or pii,
            pii_redacted=True,
            prompt_injection_detected=injection,
        )

    def _answer(
        self,
        intent: str,
        entities: dict[str, Any],
        case: GoldenCase | None,
        context: str | None,
    ) -> str:
        if case and case.expect_refusal:
            return "I cannot help with that request, but I can help with normal airline questions."
        if case and case.expected_missing_info:
            missing = ", ".join(case.expected_missing_info)
            return f"I need {missing} before I can help with that airline request."
        if intent == "flight_search":
            return "I can search flights using the provided origin, destination, date, and passenger count."
        if intent == "flight_status":
            return "I can check flight status using the booking reference or flight number."
        if intent == "baggage" and context:
            return "Economy passengers receive 1 checked bag based on the supplied airline policy."
        if intent == "refund":
            return "Refund eligibility depends on the fare rules supplied by the airline policy."
        if intent == "unsupported_request":
            return "I cannot complete that request, but I can help with airline travel questions."
        return "I can help with airline search, booking, baggage, check-in, fares, seats, and refunds."


class RealProviderError(RuntimeError):
    """Safe base exception for provider-boundary failures."""

    def __init__(
        self, message: str, *, failure: ProviderFailureMetadata | None = None
    ) -> None:
        self.failure = failure
        super().__init__(message)


class RealProviderDisabledError(RealProviderError):
    pass


class RealProviderHTTPError(RealProviderError):
    def __init__(
        self, status_code: int, *, failure: ProviderFailureMetadata | None = None
    ) -> None:
        self.status_code = status_code
        super().__init__(
            f"Real LLM provider returned HTTP {status_code}", failure=failure
        )


class RealProviderTransportError(RealProviderError):
    def __init__(self, *, failure: ProviderFailureMetadata) -> None:
        super().__init__(
            f"Real LLM provider transport failure ({failure.category.value})",
            failure=failure,
        )


class RealProviderResponseError(RealProviderError):
    pass


class _ProviderExceptionPrivacyBoundary:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if isinstance(exc, RealProviderError):
            exc.__traceback__ = None
            exc.__cause__ = None
            exc.__context__ = None
        return False


class OptionalRealLLMProvider(LLMProvider):
    """Opt-in provider-neutral HTTP adapter for structured LLM responses."""

    def __init__(
        self,
        *,
        settings: RealLLMProviderSettings | None = None,
        api_key: str | None = None,
        environment: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        tracer: TracingFacade | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        backoff: Callable[[int], float] = exponential_backoff,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("client and transport cannot both be provided")
        env = environment if environment is not None else os.environ
        self.settings = settings or RealLLMProviderSettings.from_env(env)
        resolved_api_key = api_key if api_key is not None else env.get(
            "AI_REAL_PROVIDER_API_KEY", ""
        )
        self.settings.validate(api_key_present=bool(resolved_api_key))
        self._api_key = resolved_api_key
        self.tracer = tracer or create_tracing_facade()
        self._sleeper = sleeper
        self._backoff = backoff
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self.settings.timeout_seconds,
            transport=transport,
        )

    def health_check(self) -> bool:
        return self.settings.enabled and bool(self._api_key)

    def generate(self, user_input: str, *, prompt: Prompt, context: str | None = None) -> str:
        return self.generate_structured(
            user_input, prompt=prompt, context=context
        ).response

    def generate_structured(
        self,
        user_input: str,
        *,
        prompt: Prompt,
        case: GoldenCase | None = None,
        context: str | None = None,
    ) -> AirlineAssistantResponse:
        attributes = {
            "provider_name": "http-json",
            "model_name": safe_model_label(
                self.settings.model or "unconfigured"
            ).value,
            "operation": "generate_structured",
            "prompt_length": len(user_input),
            "timeout_seconds": self.settings.timeout_seconds,
        }
        with _ProviderExceptionPrivacyBoundary(), self.tracer.start_span(
            "llm.generation", operation_type="llm", attributes=attributes
        ) as span:
            if not self.settings.enabled:
                error = RealProviderDisabledError("Real LLM provider is disabled")
                span.record_exception(error, FailureCategory.PROVIDER)
                raise error

            started_at = time.perf_counter()
            response: httpx.Response | None = None
            completed_attempt = 0
            remaining_retry_delay = self.settings.max_retry_delay_seconds
            for attempt in range(1, self.settings.max_attempts + 1):
                completed_attempt = attempt
                transport_error: RealProviderTransportError | None = None
                try:
                    response = self._client.post(
                        self.settings.base_url,
                        headers=self._headers(),
                        json=self._request_payload(user_input, prompt, context),
                    )
                except httpx.RequestError as exc:
                    span.set_attribute("status_code", None)
                    failure = classify_transport_failure(
                        exc,
                        attempt=attempt,
                        max_attempts=self.settings.max_attempts,
                    )
                    if failure.retryable and attempt < self.settings.max_attempts:
                        remaining_retry_delay = self._sleep_before_retry(
                            attempt, remaining_budget=remaining_retry_delay
                        )
                        continue
                    transport_error = RealProviderTransportError(failure=failure)

                if transport_error is not None:
                    span.record_exception(
                        transport_error, transport_error.failure.category
                    )
                    raise transport_error

                span.set_attribute("status_code", response.status_code)
                if not response.is_error:
                    break
                failure = classify_http_failure(
                    response.status_code,
                    attempt=attempt,
                    max_attempts=self.settings.max_attempts,
                )
                if failure.retryable and attempt < self.settings.max_attempts:
                    retry_after_value = response.headers.get("Retry-After")
                    retry_after = (
                        parse_retry_after(
                            retry_after_value,
                            current_time=self._clock(),
                        )
                        if retry_after_value is not None
                        else None
                    )
                    remaining_retry_delay = self._sleep_before_retry(
                        attempt,
                        retry_after_seconds=retry_after,
                        remaining_budget=remaining_retry_delay,
                    )
                    continue
                error = RealProviderHTTPError(
                    response.status_code, failure=failure
                )
                span.record_exception(error, failure.category)
                raise error

            span.set_attribute("latency_ms", (time.perf_counter() - started_at) * 1000)
            if response is None:
                raise RuntimeError("Provider request completed without a response")

            try:
                payload = response.json()
                content, token_count = _parse_provider_envelope(payload)
            except (json.JSONDecodeError, TypeError, KeyError, IndexError, ValueError):
                failure = ProviderFailureMetadata(
                    category=FailureCategory.PROVIDER,
                    retryable=False,
                    attempt=completed_attempt,
                    max_attempts=self.settings.max_attempts,
                )
                error = RealProviderResponseError(
                    "Provider response envelope is malformed", failure=failure
                )
                span.record_exception(error, FailureCategory.PROVIDER)
                raise error from None

            span.set_attribute("response_length", len(content))
            if token_count is not None:
                span.set_attribute("token_count", token_count)
            try:
                structured = json.loads(content) if isinstance(content, str) else content
                result = AirlineAssistantResponse.model_validate(structured)
            except (json.JSONDecodeError, ValidationError, TypeError):
                failure = ProviderFailureMetadata(
                    category=FailureCategory.CONTRACT,
                    retryable=False,
                    attempt=completed_attempt,
                    max_attempts=self.settings.max_attempts,
                )
                error = RealProviderResponseError(
                    "Provider structured response failed contract validation",
                    failure=failure,
                )
                span.record_exception(error, FailureCategory.CONTRACT)
                raise error from None
            return result

    def _sleep_before_retry(
        self,
        attempt: int,
        *,
        remaining_budget: float,
        retry_after_seconds: float | None = None,
    ) -> float:
        base_delay = self._backoff(attempt)
        if (
            isinstance(base_delay, bool)
            or not isinstance(base_delay, (int, float))
            or not math.isfinite(base_delay)
            or base_delay < 0
        ):
            raise ValueError("Retry backoff must be a finite non-negative number")
        selected_delay = (
            retry_after_seconds
            if retry_after_seconds is not None
            else float(base_delay)
        )
        per_attempt_delay = min(
            selected_delay, self.settings.max_retry_delay_seconds
        )
        bounded_delay = min(per_attempt_delay, remaining_budget)
        if bounded_delay > 0:
            self._sleeper(bounded_delay)
        return remaining_budget - bounded_delay

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        context = get_context()
        if context is not None:
            headers["X-Trace-ID"] = context.trace_id
            headers["X-Correlation-ID"] = context.correlation_id
        return headers

    def _request_payload(
        self, user_input: str, prompt: Prompt, context: str | None
    ) -> dict[str, Any]:
        messages = [{"role": "system", "content": prompt.text}]
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": user_input})
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
        }
        if self.settings.require_structured_output:
            payload["response_format"] = {"type": "json_object"}
        return payload


def _parse_provider_envelope(payload: Any) -> tuple[str | dict[str, Any], int | None]:
    if not isinstance(payload, dict):
        raise TypeError("provider response must be an object")
    choices = payload["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response must contain choices")
    content = choices[0]["message"]["content"]
    if not isinstance(content, (str, dict)):
        raise TypeError("provider content must be text or an object")
    usage = payload.get("usage")
    token_count = usage.get("total_tokens") if isinstance(usage, dict) else None
    if token_count is not None and (not isinstance(token_count, int) or token_count < 0):
        token_count = None
    return content, token_count
