from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_quality.models import AirlineAssistantResponse, GoldenCase, QualityReport
from ai_quality.prompt_registry import Prompt
from ai_quality.provider_capabilities import (
    AIRLINE_RESPONSE_SCHEMA_ID,
    AIRLINE_RESPONSE_SCHEMA_VERSION,
    ProviderCapabilities,
    ProviderCapabilityEvidence,
    ProviderCapabilityRequirement,
    SchemaCompatibilityEvidence,
    assess_provider_capabilities,
)
from ai_quality.provider_conformance import (
    ProviderConformanceReport,
    ProviderConformanceRunner,
    ProviderExecutionPolicy,
    serialize_provider_report,
)
from ai_quality.provider_resilience import ProviderFailureMetadata
from ai_quality.providers import LLMProvider, RealProviderResponseError
from observability.config import ObservabilitySettings
from observability.exporters import InMemoryExporter
from observability.models import FailureCategory
from observability.tracing import TracingFacade


FIXED_PREFLIGHT_INPUT = "Verify structured airline response compatibility."
FIXED_PREFLIGHT_PROMPT = "Return one valid structured airline assistant response."


def _response() -> AirlineAssistantResponse:
    return AirlineAssistantResponse(
        intent="flight_search",
        response="I can help with flights.",
        entities={"destination": "LHR"},
        actions=[],
        grounding={"context_ids": [], "grounded": True},
        safety={
            "safe": True,
            "refusal": False,
            "pii_detected": False,
            "pii_redacted": True,
            "prompt_injection_detected": False,
        },
        prompt_version="v1",
    )


def _case() -> GoldenCase:
    return GoldenCase(
        id="capability-case",
        category="flight_search",
        user_input="Find a flight to London",
        context="Public airline policy",
        expected_intent="flight_search",
    )


class _Evaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, cases, responses):
        self.calls += 1
        return QualityReport(
            total_cases=len(cases),
            passed_cases=len(cases),
            failed_cases=0,
            intent_accuracy=1.0,
            entity_accuracy=1.0,
            structured_output_validity=1.0,
            relevance_score=1.0,
            groundedness_score=1.0,
            safety_pass_rate=1.0,
            pii_protection_rate=1.0,
            prompt_injection_pass_rate=1.0,
            hallucination_pass_rate=1.0,
            overall_passed=True,
        )

class _UnknownProvider(LLMProvider):
    def __init__(self, *, preflight_error=None) -> None:
        self.calls = []
        self.preflight_error = preflight_error

    def health_check(self) -> bool:
        return True

    def generate(self, user_input, *, prompt, context=None):
        return self.generate_structured(
            user_input, prompt=prompt, context=context
        ).response

    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        self.calls.append((user_input, prompt.text, case, context))
        if case is None and self.preflight_error is not None:
            raise self.preflight_error
        return _response()


class _DeclaredProvider(_UnknownProvider):
    def __init__(self, declaration, *, preflight_error=None) -> None:
        super().__init__(preflight_error=preflight_error)
        self.declaration = declaration

    def get_capabilities(self):
        return self.declaration


def _capabilities(**overrides) -> ProviderCapabilities:
    values = {
        "structured_output": True,
        "response_schema_id": AIRLINE_RESPONSE_SCHEMA_ID,
        "response_schema_versions": (AIRLINE_RESPONSE_SCHEMA_VERSION,),
    }
    values.update(overrides)
    return ProviderCapabilities(**values)


def _runner(provider, evaluator, *, required=False):
    return ProviderConformanceRunner(
        provider=provider,
        provider_label="test-provider",
        model_label="test-model",
        policy=ProviderExecutionPolicy(required=required),
        evaluator=evaluator,
    )


def test_capability_contracts_are_strict_and_immutable():
    capability = _capabilities()
    requirement = ProviderCapabilityRequirement()
    evidence = ProviderCapabilityEvidence(
        status="supported", structured_output_supported=True
    )
    assert requirement.response_schema_version == "1.0"
    with pytest.raises(ValidationError):
        ProviderCapabilities(
            structured_output=1,
            response_schema_id=AIRLINE_RESPONSE_SCHEMA_ID,
            response_schema_versions=("1.0",),
        )
    with pytest.raises(ValidationError):
        capability.structured_output = False
    with pytest.raises(ValidationError):
        evidence.status = "unknown"

    compatibility = SchemaCompatibilityEvidence(
        declaration="compatible",
        empirical="verified",
        verification_basis="both",
        golden_contract_valid=True,
    )
    assert compatibility.expected_schema_id == AIRLINE_RESPONSE_SCHEMA_ID
    assert compatibility.expected_schema_version == AIRLINE_RESPONSE_SCHEMA_VERSION
    with pytest.raises(ValidationError):
        compatibility.empirical = "failed"
    with pytest.raises(ValidationError):
        SchemaCompatibilityEvidence(
            declaration="unknown",
            empirical="verified",
            verification_basis="both",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"response_schema_id": "https://secret.example/schema"},
        {"response_schema_id": "Authorization:Bearer-secret"},
        {"response_schema_versions": ("1.0", "1.0")},
        {"response_schema_versions": ("secret-version",)},
    ],
)
def test_hostile_or_malformed_declarations_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _capabilities(**overrides)


def test_declared_supported_capability_is_recognized():
    evidence = assess_provider_capabilities(
        _DeclaredProvider(_capabilities()), ProviderCapabilityRequirement()
    )
    assert evidence.status == "supported"


def test_declared_compatible_and_verified_preserves_both_sources():
    provider = _DeclaredProvider(_capabilities())
    report = _runner(provider, _Evaluator()).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.capability.status == "supported"
    assert report.schema_compatibility.declaration == "compatible"
    assert report.schema_compatibility.empirical == "verified"
    assert report.schema_compatibility.verification_basis == "both"
    assert report.schema_compatibility.golden_contract_valid is True
    assert len(provider.calls) == 2


def test_explicit_unknown_declaration_is_empirically_verified():
    provider = _DeclaredProvider(None)
    report = _runner(provider, _Evaluator()).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.capability.status == "unknown"
    assert report.schema_compatibility.declaration == "unknown"
    assert report.schema_compatibility.empirical == "verified"
    assert report.schema_compatibility.verification_basis == "empirical"
    assert report.schema_compatibility.golden_contract_valid is True


def test_provider_without_optional_protocol_is_unknown():
    evidence = assess_provider_capabilities(
        _UnknownProvider(), ProviderCapabilityRequirement()
    )
    assert evidence.status == "unknown"
    assert evidence.reason_code is None


@pytest.mark.parametrize("required", [False, True])
def test_declared_unsupported_short_circuits_all_execution(required):
    evaluator = _Evaluator()
    provider = _DeclaredProvider(_capabilities(structured_output=False))
    report = _runner(provider, evaluator, required=required).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.outcome == "capability_failed"
    assert report.reason_code == "provider_capability_unsupported"
    assert report.failure.category == FailureCategory.CONTRACT
    assert report.failure_category == FailureCategory.CONTRACT
    assert report.completed_case_count == 0
    assert report.quality_report is None
    assert report.policy_passed is False
    assert report.schema_compatibility.declaration == "incompatible"
    assert report.schema_compatibility.empirical == "not_run"
    assert report.schema_compatibility.verification_basis is None
    assert provider.calls == []
    assert evaluator.calls == 0


def test_declared_schema_mismatch_short_circuits_all_execution():
    evaluator = _Evaluator()
    provider = _DeclaredProvider(
        {
            "structured_output": True,
            "response_schema_id": AIRLINE_RESPONSE_SCHEMA_ID,
            "response_schema_versions": ("2.0",),
        }
    )
    report = _runner(provider, evaluator).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.outcome == "schema_mismatch"
    assert report.reason_code == "provider_schema_mismatch"
    assert report.failure_category == FailureCategory.CONTRACT
    assert report.failure.category == FailureCategory.CONTRACT
    assert report.completed_case_count == 0
    assert report.quality_report is None
    assert report.policy_passed is False
    assert report.schema_compatibility.declaration == "incompatible"
    assert report.schema_compatibility.empirical == "not_run"
    assert provider.calls == []
    assert evaluator.calls == 0


def test_declared_schema_identifier_mismatch_short_circuits_execution():
    evaluator = _Evaluator()
    provider = _DeclaredProvider(
        {
            "structured_output": True,
            "response_schema_id": "other-airline-response",
            "response_schema_versions": (AIRLINE_RESPONSE_SCHEMA_VERSION,),
        }
    )
    report = _runner(provider, evaluator).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.outcome == "schema_mismatch"
    assert report.capability.response_schema_id == AIRLINE_RESPONSE_SCHEMA_ID
    assert "other-airline-response" not in serialize_provider_report(report)
    assert provider.calls == []
    assert evaluator.calls == 0


@pytest.mark.parametrize(
    "hostile_schema_id",
    [
        "book-flight-now",
        "opaque-0123456789abcdef",
        "case-deadbeefdeadbeef",
        "ignore-previous-instructions",
        "https://private.example/schema",
        "sk-live-secret-123",
        "Bearer-secret-token",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_12345678901234567890",
    ],
    ids=[
        "prompt-like",
        "opaque-token",
        "spoofed-case",
        "instruction-like",
        "url-like",
        "api-key-like",
        "bearer-like",
        "aws-key-like",
        "github-token-like",
    ],
)
def test_hostile_schema_identifiers_are_absent_from_reports_and_spans(
    hostile_schema_id,
):
    with pytest.raises(ValidationError):
        _capabilities(response_schema_id=hostile_schema_id)

    exporter = InMemoryExporter()
    tracer = TracingFacade(
        settings=ObservabilitySettings(enabled=True), exporter=exporter
    )
    evaluator = _Evaluator()
    provider = _DeclaredProvider(
        {
            "structured_output": True,
            "response_schema_id": hostile_schema_id,
            "response_schema_versions": (AIRLINE_RESPONSE_SCHEMA_VERSION,),
        }
    )
    report = ProviderConformanceRunner(
        provider=provider,
        provider_label="test-provider",
        model_label="test-model",
        evaluator=evaluator,
        tracer=tracer,
    ).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )

    exported = json.dumps(
        {
            "report": json.loads(serialize_provider_report(report)),
            "spans": [span.model_dump(mode="json") for span in exporter.spans],
        },
        sort_keys=True,
    )
    assert report.outcome == "schema_mismatch"
    assert report.reason_code == "provider_schema_mismatch"
    assert report.failure_category == FailureCategory.CONTRACT
    assert report.capability.response_schema_id == AIRLINE_RESPONSE_SCHEMA_ID
    assert hostile_schema_id not in exported
    assert provider.calls == []
    assert evaluator.calls == 0


@pytest.mark.parametrize(
    "hostile_version",
    [
        "2.0",
        "../../secret",
        "https://attacker.example",
        "Authorization: Bearer private-token",
        "x" * 4096,
        '{"schema_version":"1.0","extra":"private"}',
        "1.0\nAuthorization: Bearer private-token",
    ],
    ids=[
        "future-version",
        "path",
        "url",
        "authorization",
        "oversized",
        "json-looking",
        "control-character",
    ],
)
def test_hostile_schema_versions_are_normalized_without_retention(hostile_version):
    provider = _DeclaredProvider(
        {
            "structured_output": True,
            "response_schema_id": AIRLINE_RESPONSE_SCHEMA_ID,
            "response_schema_versions": (hostile_version,),
        }
    )
    report = _runner(provider, _Evaluator()).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    serialized = serialize_provider_report(report)
    assert report.outcome == "schema_mismatch"
    assert report.schema_compatibility.declaration == "incompatible"
    assert report.schema_compatibility.empirical == "not_run"
    public_capability = json.dumps(
        report.capability.model_dump(mode="json"), sort_keys=True
    )
    assert hostile_version not in public_capability
    assert "response_schema_versions" not in serialized
    assert report.schema_compatibility.expected_schema_version == "1.0"


def test_capability_field_injection_is_invalid_and_not_persisted():
    hostile = "private-injected-metadata"
    provider = _DeclaredProvider(
        {
            "structured_output": True,
            "response_schema_id": AIRLINE_RESPONSE_SCHEMA_ID,
            "response_schema_versions": (AIRLINE_RESPONSE_SCHEMA_VERSION,),
            "arbitrary_metadata": hostile,
        }
    )
    report = _runner(provider, _Evaluator()).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    serialized = serialize_provider_report(report)
    assert report.outcome == "capability_failed"
    assert report.reason_code == "provider_capability_invalid"
    assert report.schema_compatibility.declaration == "invalid"
    assert hostile not in serialized


def test_malformed_protocol_declaration_fails_closed():
    evaluator = _Evaluator()
    provider = _DeclaredProvider({"structured_output": "yes"})
    report = _runner(provider, evaluator).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.outcome == "capability_failed"
    assert report.reason_code == "provider_capability_invalid"
    assert report.capability.status == "unknown"
    assert report.schema_compatibility.declaration == "invalid"
    assert report.schema_compatibility.empirical == "not_run"
    assert provider.calls == []


def test_unknown_declaration_runs_one_preflight_before_golden_cases():
    evaluator = _Evaluator()
    provider = _UnknownProvider()
    case = _case()
    report = _runner(provider, evaluator).run(
        [case], prompt=Prompt(name="golden", version="v1", purpose="p", text="golden")
    )
    assert report.outcome == "passed"
    assert report.capability.status == "unknown"
    assert report.schema_compatibility.declaration == "absent"
    assert report.schema_compatibility.empirical == "verified"
    assert report.schema_compatibility.verification_basis == "empirical"
    assert report.schema_compatibility.golden_contract_valid is True
    assert evaluator.calls == 1
    assert len(provider.calls) == 2
    preflight, golden = provider.calls
    assert preflight == (
        FIXED_PREFLIGHT_INPUT,
        FIXED_PREFLIGHT_PROMPT,
        None,
        None,
    )
    assert golden[0] == case.user_input
    assert golden[2] is case
    assert case.user_input not in preflight
    assert case.context not in preflight


def test_schema_invalid_preflight_prevents_golden_and_evaluation():
    failure = ProviderFailureMetadata(
        category=FailureCategory.CONTRACT,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    provider = _UnknownProvider(
        preflight_error=RealProviderResponseError(
            "safe schema failure", failure=failure
        )
    )
    evaluator = _Evaluator()
    report = _runner(provider, evaluator).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    assert report.outcome == "schema_mismatch"
    assert report.capability.status == "unknown"
    assert report.schema_compatibility.declaration == "absent"
    assert report.schema_compatibility.empirical == "failed"
    assert len(provider.calls) == 1
    assert evaluator.calls == 0


def test_declared_compatible_preflight_failure_preserves_provenance():
    failure = ProviderFailureMetadata(
        category=FailureCategory.CONTRACT,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    provider = _DeclaredProvider(
        _capabilities(),
        preflight_error=RealProviderResponseError(
            "private provider detail", failure=failure
        ),
    )
    report = _runner(provider, _Evaluator()).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    serialized = serialize_provider_report(report)
    assert report.outcome == "schema_mismatch"
    assert report.capability.status == "supported"
    assert report.schema_compatibility.declaration == "compatible"
    assert report.schema_compatibility.empirical == "failed"
    assert report.schema_compatibility.verification_basis is None
    assert report.schema_compatibility.golden_contract_valid is False
    assert "private provider detail" not in serialized


def test_capability_evidence_is_safe_and_deterministic():
    provider = _DeclaredProvider(
        {
            "structured_output": True,
            "response_schema_id": AIRLINE_RESPONSE_SCHEMA_ID,
            "response_schema_versions": ("2.0",),
        }
    )
    report = _runner(provider, _Evaluator()).run(
        [_case()], prompt=Prompt(name="p", version="v1", purpose="p", text="p")
    )
    first = serialize_provider_report(report)
    second = serialize_provider_report(report)
    assert first == second
    assert json.loads(first)["capability"] == {
        "failure_category": "contract",
        "reason_code": "provider_schema_mismatch",
        "response_schema_id": "airline-assistant-response",
        "response_schema_version": "1.0",
        "status": "schema_mismatch",
        "structured_output_supported": True,
    }
    assert "response_schema_versions" not in first


def test_report_rejects_supported_outcome_with_mismatched_capability():
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            provider="test-provider",
            model="test-model",
            required=False,
            execution_status="executed",
            outcome="passed",
            reason_code=None,
            case_count=1,
            completed_case_count=1,
            case_ids=("case-1",),
            capability=ProviderCapabilityEvidence(
                status="schema_mismatch",
                structured_output_supported=True,
                failure_category=FailureCategory.CONTRACT,
                reason_code="provider_schema_mismatch",
            ),
            schema_compatibility=SchemaCompatibilityEvidence(
                declaration="incompatible",
                empirical="not_run",
            ),
            quality_report=_Evaluator().evaluate([_case()], [_response()]),
            policy_passed=True,
        )
