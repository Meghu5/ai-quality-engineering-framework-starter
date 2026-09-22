from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

from ai_quality.models import (
    AirlineAssistantResponse,
    ChatbotAction,
    GoldenCase,
    GroundingResult,
    SafetyResult,
)
from ai_quality.prompt_registry import Prompt
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


class OptionalRealLLMProvider(LLMProvider):
    """Disabled-by-default adapter seam for future real LLM integrations."""

    def __init__(
        self,
        *,
        api_key_env: str = "REAL_LLM_API_KEY",
        tracer: TracingFacade | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.tracer = tracer or create_tracing_facade()

    def health_check(self) -> bool:
        return bool(os.getenv(self.api_key_env))

    def generate(self, user_input: str, *, prompt: Prompt, context: str | None = None) -> str:
        return self._raise_unavailable(user_input, operation="generate")

    def generate_structured(
        self,
        user_input: str,
        *,
        prompt: Prompt,
        case: GoldenCase | None = None,
        context: str | None = None,
    ) -> AirlineAssistantResponse:
        return self._raise_unavailable(user_input, operation="generate_structured")

    def _raise_unavailable(self, user_input: str, *, operation: str):
        with self.tracer.start_span(
            "llm.generation",
            operation_type="llm",
            attributes={
                "provider_name": "optional_real",
                "model_name": "unconfigured",
                "operation": operation,
                "prompt_length": len(user_input),
            },
        ) as span:
            error = NotImplementedError(
                "Real LLM providers are intentionally not wired in Phase 8."
            )
            span.record_exception(error, FailureCategory.MODEL)
            raise error
