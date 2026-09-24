from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from ai_eval.models import Phase10Report
from ai_quality.models import AirlineAssistantResponse, GoldenCase, QualityReport
from ai_quality.privacy import safe_case_id, safe_model_label
from ai_quality.prompt_registry import Prompt
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.provider_conformance import (
    ProviderConformanceRunner,
    ProviderConformanceReport,
    ProviderExecutionPolicy,
    ProviderFailureEvidence,
    adapt_provider_report_to_phase10,
    conformance_exit_code,
    run_provider_conformance,
    serialize_provider_report,
    write_provider_phase10_report,
)
from ai_quality.providers import LLMProvider, OptionalRealLLMProvider
from observability.lifecycle import (
    ArtifactType,
    create_bundle_from_existing_reports,
    verify_bundle,
)
from observability.config import ObservabilitySettings
from observability.exporters import InMemoryExporter
from observability.models import FailureCategory
from observability.tracing import TracingFacade


pytestmark = pytest.mark.real_llm

API_KEY = "conformance-key-must-never-appear"
RAW_PROMPT = "private.person@example.test needs a flight"
RAW_CONTEXT = "private policy context"
RAW_RESPONSE = "private provider response"
ENDPOINT = "https://llm.example.test/private/chat"


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(
        name="airline_assistant",
        version="v1",
        purpose="test",
        text="Return the airline response contract.",
    )


@pytest.fixture
def cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            id="case-002",
            category="flight_search",
            user_input="Find a flight to London",
            context="Airline policy context two",
            expected_intent="flight_search",
            expected_entities={"destination": "LHR"},
        ),
        GoldenCase(
            id="case-001",
            category="flight_search",
            user_input="Find another flight to London",
            context="Airline policy context one",
            expected_intent="flight_search",
            expected_entities={"destination": "LHR"},
        ),
    ]


def _assistant_response(text: str = "I can help with flights.") -> AirlineAssistantResponse:
    return AirlineAssistantResponse(
        intent="flight_search",
        response=text,
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


def _quality_report(*, passed: bool) -> QualityReport:
    score = 1.0 if passed else 0.0
    return QualityReport(
        total_cases=2,
        passed_cases=2 if passed else 0,
        failed_cases=0 if passed else 2,
        intent_accuracy=score,
        entity_accuracy=score,
        structured_output_validity=1.0,
        relevance_score=score,
        groundedness_score=score,
        safety_pass_rate=score,
        pii_protection_rate=score,
        prompt_injection_pass_rate=score,
        hallucination_pass_rate=score,
        overall_passed=passed,
    )


class StaticProvider(LLMProvider):
    def __init__(self, responses, *, available=True):
        self.responses = iter(responses)
        self.available = available
        self.case_ids = []

    def health_check(self) -> bool:
        return self.available

    def generate(self, user_input, *, prompt, context=None):
        return self.generate_structured(
            user_input, prompt=prompt, context=context
        ).response

    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        if case is None:
            return _assistant_response()
        self.case_ids.append(case.id if case else "none")
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


class RecordingEvaluator:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def evaluate(self, cases, responses):
        self.calls.append((list(cases), list(responses)))
        return self.report


def _runner(provider, *, required=False, evaluator=None):
    return ProviderConformanceRunner(
        provider=provider,
        provider_label="test-provider",
        model_label="test-model",
        policy=ProviderExecutionPolicy(required=required),
        evaluator=evaluator,
    )


def _provider_payload(content=None):
    resolved = _assistant_response().model_dump(mode="json") if content is None else content
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": resolved}}]},
    )


def _real_provider(handler, *, max_attempts=1):
    return OptionalRealLLMProvider(
        settings=RealLLMProviderSettings(
            enabled=True,
            base_url=ENDPOINT,
            model="test-model",
            timeout_seconds=5.0,
            max_attempts=max_attempts,
            require_structured_output=True,
        ),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        sleeper=lambda delay: None,
    )


def _run_real(handler, cases, prompt, *, max_attempts=1):
    provider = _real_provider(handler, max_attempts=max_attempts)
    try:
        return _runner(provider).run(cases, prompt=prompt)
    finally:
        provider.close()


def _base_phase10_report() -> Phase10Report:
    return Phase10Report(
        baseline={
            "phase8": {"execution_status": "executed", "overall_passed": True},
            "phase9": {"execution_status": "executed", "overall_passed": True},
            "overall_passed": True,
        },
        frameworks=[],
        comparisons=[],
        overall_passed=True,
    )


def test_all_cases_succeed_and_quality_passes(cases, prompt):
    evaluator = RecordingEvaluator(_quality_report(passed=True))
    provider = StaticProvider([_assistant_response(), _assistant_response()])
    report = _runner(provider, evaluator=evaluator).run(cases, prompt=prompt)

    assert report.execution_status == "executed"
    assert report.outcome == "passed"
    assert report.failure_category is None
    assert report.reason_code is None
    assert report.quality_report.model_dump() == _quality_report(
        passed=True
    ).model_dump()
    assert report.completed_case_count == 2
    assert len(report.case_ids) == 2
    assert all(case_id.startswith("case-") for case_id in report.case_ids)
    assert report.case_ids[0] != report.case_ids[1]
    assert provider.case_ids == ["case-002", "case-001"]
    assert len(evaluator.calls) == 1
    assert report.policy_passed is True
    assert conformance_exit_code(report) == 0


def test_all_cases_succeed_and_quality_fails(cases, prompt):
    evaluator = RecordingEvaluator(_quality_report(passed=False))
    provider = StaticProvider([_assistant_response(), _assistant_response()])
    report = _runner(provider, evaluator=evaluator).run(cases, prompt=prompt)

    assert report.execution_status == "executed"
    assert report.outcome == "quality_failed"
    assert report.failure_category is None
    assert report.reason_code == "quality_gate_failed"
    assert report.quality_report.overall_passed is False
    assert report.failure is None
    assert report.completed_case_count == report.case_count
    assert report.policy_passed is False
    assert conformance_exit_code(report) == 1


@pytest.mark.parametrize(
    ("required", "status", "exit_code"),
    [(False, "not_executed", 0), (True, "provider_required", 1)],
)
def test_not_configured_policy(required, status, exit_code, cases, prompt):
    report = _runner(None, required=required).run(cases, prompt=prompt)
    assert report.execution_status == status
    assert report.outcome == "not_configured"
    assert report.quality_report is None
    assert report.completed_case_count == 0
    assert conformance_exit_code(report) == exit_code


@pytest.mark.parametrize("required", [False, True])
def test_health_unavailable_has_no_quality_metrics(required, cases, prompt):
    report = _runner(
        StaticProvider([], available=False), required=required
    ).run(cases, prompt=prompt)
    assert report.execution_status == "unavailable"
    assert report.outcome == "unavailable"
    assert report.quality_report is None
    assert conformance_exit_code(report) == (1 if required else 0)


@pytest.mark.parametrize(
    ("handler", "category", "outcome", "reason_code"),
    [
        (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("private timeout")), FailureCategory.TIMEOUT, "transport_failed", "provider_timeout"),
        (lambda request: (_ for _ in ()).throw(httpx.ConnectError("private network")), FailureCategory.NETWORK, "transport_failed", "provider_network_failed"),
        (lambda request: httpx.Response(401), FailureCategory.AUTHENTICATION, "transport_failed", "provider_authentication_failed"),
        (lambda request: httpx.Response(403), FailureCategory.AUTHORIZATION, "transport_failed", "provider_authorization_failed"),
        (lambda request: httpx.Response(429), FailureCategory.RATE_LIMIT, "transport_failed", "provider_rate_limited"),
        (lambda request: httpx.Response(503), FailureCategory.PROVIDER, "transport_failed", "provider_failed"),
        (lambda request: httpx.Response(200, content=b"not-json"), FailureCategory.PROVIDER, "contract_failed", "provider_response_malformed"),
        (lambda request: _provider_payload({"intent": "missing-fields"}), FailureCategory.CONTRACT, "schema_mismatch", "provider_schema_mismatch"),
    ],
)
def test_real_provider_failures_have_safe_states(
    handler, category, outcome, reason_code, cases, prompt
):
    report = _run_real(handler, cases, prompt)
    assert report.execution_status == "failed"
    assert report.outcome == outcome
    assert report.failure.category == category
    assert report.failure_category == category
    assert report.reason_code == reason_code
    assert report.quality_report is None
    assert report.completed_case_count == 0
    assert report.case_count == len(cases)
    assert report.policy_passed is False
    assert conformance_exit_code(report) == 1


def test_retry_exhaustion_retains_final_attempt(cases, prompt):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    report = _run_real(handler, cases, prompt, max_attempts=3)
    assert len(calls) == 3
    assert report.failure.attempt == 3
    assert report.failure.max_attempts == 3
    assert report.failure.retryable is True


def test_generic_failure_is_normalized_without_exception_text(cases, prompt):
    evaluator = RecordingEvaluator(_quality_report(passed=True))
    provider = StaticProvider([RuntimeError("secret arbitrary exception")])
    report = _runner(provider, evaluator=evaluator).run(cases, prompt=prompt)
    serialized = serialize_provider_report(report)
    assert report.execution_status == "failed"
    assert report.outcome == "transport_failed"
    assert report.failure.category == FailureCategory.PROVIDER
    assert report.failure_category == FailureCategory.PROVIDER
    assert report.reason_code == "provider_failed"
    assert report.quality_report is None
    assert evaluator.calls == []
    assert "secret arbitrary exception" not in serialized


def test_partial_responses_are_never_evaluated(cases, prompt):
    evaluator = RecordingEvaluator(_quality_report(passed=True))
    provider = StaticProvider([_assistant_response(), RuntimeError("second failed")])
    report = _runner(provider, evaluator=evaluator).run(cases, prompt=prompt)
    assert report.completed_case_count == 1
    assert report.quality_report is None
    assert evaluator.calls == []


def test_serialization_is_deterministic_and_labels_are_sanitized(cases, prompt):
    evaluator = RecordingEvaluator(_quality_report(passed=True))
    first = ProviderConformanceRunner(
        provider=StaticProvider([_assistant_response(), _assistant_response()]),
        provider_label="https://user:secret@example.test/path",
        model_label="unsafe/model?token=secret",
        evaluator=evaluator,
    ).run(cases, prompt=prompt)
    second = ProviderConformanceRunner(
        provider=StaticProvider([_assistant_response(), _assistant_response()]),
        provider_label="https://user:secret@example.test/path",
        model_label="unsafe/model?token=secret",
        evaluator=RecordingEvaluator(_quality_report(passed=True)),
    ).run(cases, prompt=prompt)
    assert first.provider.startswith("opaque-")
    assert first.model.startswith("opaque-")
    assert first.provider != first.model
    assert serialize_provider_report(first) == serialize_provider_report(second)


def test_phase10_adapter_preserves_existing_sections(cases, prompt):
    provider_report = _runner(
        StaticProvider([_assistant_response(), _assistant_response()]),
        evaluator=RecordingEvaluator(_quality_report(passed=True)),
    ).run(cases, prompt=prompt)
    original = _base_phase10_report()
    adapted = adapt_provider_report_to_phase10(original, provider_report)

    assert adapted.baseline["phase8"] == original.baseline["phase8"]
    assert adapted.baseline["phase9"] == original.baseline["phase9"]
    assert adapted.baseline["provider_conformance"]["outcome"] == "passed"
    assert Phase10Report.model_validate(adapted.model_dump()) == adapted


def test_failed_provider_makes_phase10_artifact_fail(cases, prompt):
    provider_report = _runner(
        StaticProvider([RuntimeError("failed")])
    ).run(cases, prompt=prompt)
    adapted = adapt_provider_report_to_phase10(_base_phase10_report(), provider_report)
    assert adapted.overall_passed is False
    assert adapted.baseline["phase8"]["overall_passed"] is True


def test_lifecycle_accepts_report(tmp_path, cases, prompt):
    provider_report = _runner(
        StaticProvider([_assistant_response(), _assistant_response()]),
        evaluator=RecordingEvaluator(_quality_report(passed=True)),
    ).run(cases, prompt=prompt)
    adapted = adapt_provider_report_to_phase10(_base_phase10_report(), provider_report)
    source_root = tmp_path / "reports"
    artifact_path = source_root / "ai_eval" / "phase10_report.json"
    write_provider_phase10_report(artifact_path, adapted)

    lifecycle_root = Path(tempfile.mkdtemp(prefix="p12-lifecycle-"))
    try:
        bundle = create_bundle_from_existing_reports(
            source_root=source_root,
            lifecycle_root=lifecycle_root,
            environment={"AI_OBSERVABILITY_RUN_ID": "provider-conformance-test"},
            required_types=[ArtifactType.PHASE10_REPORT],
        )
        manifest = verify_bundle(bundle)
        assert [item.artifact_type for item in manifest.artifacts] == [
            ArtifactType.PHASE10_REPORT
        ]
    finally:
        shutil.rmtree(lifecycle_root, ignore_errors=True)


def test_entrypoint_writes_combined_report_and_returns_exit_code(
    tmp_path, cases, prompt
):
    output_path = tmp_path / "phase10_report.json"
    exit_code = run_provider_conformance(
        runner=_runner(
            StaticProvider([_assistant_response(), _assistant_response()]),
            evaluator=RecordingEvaluator(_quality_report(passed=True)),
        ),
        cases=cases,
        prompt=prompt,
        phase10_report=_base_phase10_report(),
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["baseline"]["provider_conformance"]["outcome"] == "passed"


def test_entrypoint_uses_injected_golden_case_loader(tmp_path, cases, prompt):
    loaded_paths = []

    def load(path):
        loaded_paths.append(path)
        return cases

    exit_code = run_provider_conformance(
        runner=_runner(
            StaticProvider([_assistant_response(), _assistant_response()]),
            evaluator=RecordingEvaluator(_quality_report(passed=True)),
        ),
        prompt=prompt,
        phase10_report=_base_phase10_report(),
        output_path=tmp_path / "phase10_report.json",
        golden_cases_path=tmp_path / "golden_cases.json",
        case_loader=load,
    )

    assert exit_code == 0
    assert loaded_paths == [tmp_path / "golden_cases.json"]


def test_tracing_has_one_parent_and_one_provider_span(cases, prompt):
    exporter = InMemoryExporter()
    tracer = TracingFacade(
        settings=ObservabilitySettings(enabled=True, exporter="memory"),
        exporter=exporter,
    )
    provider = OptionalRealLLMProvider(
        settings=RealLLMProviderSettings(
            enabled=True,
            base_url=ENDPOINT,
            model="test-model",
            timeout_seconds=5.0,
            max_attempts=1,
            require_structured_output=True,
        ),
        api_key=API_KEY,
        transport=httpx.MockTransport(lambda request: _provider_payload()),
        tracer=tracer,
        sleeper=lambda delay: None,
    )
    try:
        report = ProviderConformanceRunner(
            provider=provider,
            provider_label="test-provider",
            model_label="test-model",
            evaluator=RecordingEvaluator(_quality_report(passed=True)),
            tracer=tracer,
        ).run(cases, prompt=prompt)
    finally:
        provider.close()

    spans = exporter.spans
    assert report.outcome == "passed"
    assert [span.operation_name for span in spans] == [
        "llm.generation",
        "llm.generation",
        "llm.generation",
        "provider.conformance",
    ]
    parent = spans[-1]
    assert all(span.trace_id == parent.trace_id for span in spans)
    assert all(span.parent_span_id == parent.span_id for span in spans[:-1])
    serialized = json.dumps(
        [span.model_dump(mode="json") for span in spans], sort_keys=True
    )
    for sensitive in (API_KEY, ENDPOINT, RAW_PROMPT, RAW_CONTEXT):
        assert sensitive not in serialized


def test_report_and_lifecycle_evidence_are_privacy_safe(tmp_path, prompt):
    private_case = GoldenCase(
        id="private.person@example.test",
        category="flight_search",
        user_input=RAW_PROMPT,
        context=RAW_CONTEXT,
        expected_intent="flight_search",
    )

    def handler(request):
        return httpx.Response(503, text=RAW_RESPONSE)

    provider_report = _run_real(handler, [private_case], prompt)
    adapted = adapt_provider_report_to_phase10(_base_phase10_report(), provider_report)
    artifact = tmp_path / "reports" / "ai_eval" / "phase10_report.json"
    write_provider_phase10_report(artifact, adapted)
    serialized = artifact.read_text(encoding="utf-8")
    assert provider_report.case_ids[0].startswith("case-")
    for sensitive in (
        API_KEY,
        ENDPOINT,
        RAW_PROMPT,
        RAW_CONTEXT,
        RAW_RESPONSE,
        "private.person",
        "Authorization",
        "traceback",
        "Verify structured airline response compatibility.",
        "Return one valid structured airline assistant response.",
    ):
        assert sensitive not in serialized


def test_required_configuration_is_secret_free():
    settings = RealLLMProviderSettings.from_env(
        {
            "AI_REAL_PROVIDER_ENABLED": "false",
            "AI_REAL_PROVIDER_REQUIRED": "true",
        }
    )
    assert settings.required is True
    assert "api_key" not in json.dumps(settings.safe_dict(), sort_keys=True).lower()


def test_hostile_identifiers_are_deterministically_protected(prompt):
    hostile_case = GoldenCase(
        id="5551234567",
        category="flight_search",
        user_input=RAW_PROMPT,
        context=RAW_CONTEXT,
        expected_intent="flight_search",
    )

    def build():
        return ProviderConformanceRunner(
            provider=None,
            provider_label="sk-live-secret-123",
            model_label="Bearer-secret-token",
        ).run([hostile_case], prompt=prompt)

    first = serialize_provider_report(build())
    second = serialize_provider_report(build())
    assert first == second
    for hostile in (
        "sk-live-secret-123",
        "Bearer-secret-token",
        "5551234567",
        RAW_PROMPT,
        RAW_CONTEXT,
    ):
        assert hostile not in first


def test_email_case_id_is_deterministically_protected(prompt):
    case = GoldenCase(
        id="private.person@example.test",
        category="flight_search",
        user_input="hello",
        expected_intent="flight_search",
    )
    first = _runner(None).run([case], prompt=prompt)
    second = _runner(None).run([case], prompt=prompt)
    assert first.case_ids == second.case_ids
    assert "private.person@example.test" not in serialize_provider_report(first)


def _report_payload(**overrides):
    payload = {
        "provider": "test-provider",
        "model": "test-model",
        "required": False,
        "execution_status": "not_executed",
        "outcome": "not_configured",
        "reason_code": "provider_not_configured",
        "failure_category": None,
        "case_count": 1,
        "completed_case_count": 0,
        "case_ids": ["case-1"],
        "failure": None,
        "quality_report": None,
        "policy_passed": True,
    }
    payload.update(overrides)
    if "failure" in overrides and "failure_category" not in overrides:
        failure = overrides["failure"]
        payload["failure_category"] = failure.category if failure else None
    return payload


@pytest.mark.parametrize(
    "overrides",
    [
        {"execution_status": "failed"},
        {"completed_case_count": 1},
        {
            "outcome": "transport_failed",
            "execution_status": "failed",
            "reason_code": "provider_timeout",
            "policy_passed": False,
        },
        {
            "outcome": "contract_failed",
            "execution_status": "failed",
            "reason_code": "provider_response_malformed",
            "policy_passed": False,
        },
        {"policy_passed": False},
    ],
)
def test_invalid_incomplete_report_combinations_are_rejected(overrides):
    with pytest.raises(ValidationError):
        ProviderConformanceReport(**_report_payload(**overrides))


@pytest.mark.parametrize("outcome", ["passed", "quality_failed"])
def test_quality_outcome_requires_real_matching_report(outcome):
    reason = (
        None
        if outcome == "passed"
        else "quality_gate_failed"
    )
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="executed",
                outcome=outcome,
                reason_code=reason,
                completed_case_count=1,
                policy_passed=outcome == "passed",
            )
        )


@pytest.mark.parametrize("outcome", ["passed", "quality_failed"])
def test_quality_outcome_rejects_failure_metadata(outcome):
    quality = _quality_report(passed=outcome == "passed")
    reason = (
        None
        if outcome == "passed"
        else "quality_gate_failed"
    )
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="executed",
                outcome=outcome,
                reason_code=reason,
                completed_case_count=1,
                quality_report=quality.model_copy(update={"total_cases": 1}),
                failure=ProviderFailureEvidence(
                    category=FailureCategory.UNKNOWN,
                    retryable=False,
                    attempt=1,
                    max_attempts=1,
                ),
                policy_passed=outcome == "passed",
            )
        )


def test_quality_report_case_count_must_match():
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="executed",
                outcome="passed",
                reason_code=None,
                completed_case_count=1,
                quality_report=_quality_report(passed=True),
            )
        )


def test_report_rejects_cross_field_contract_violations():
    passing_quality = _quality_report(passed=True).model_copy(
        update={"total_cases": 1}
    )
    failing_quality = _quality_report(passed=False).model_copy(
        update={"total_cases": 1}
    )
    network_failure = ProviderFailureEvidence(
        category=FailureCategory.NETWORK,
        retryable=True,
        attempt=1,
        max_attempts=1,
    )
    provider_failure = ProviderFailureEvidence(
        category=FailureCategory.PROVIDER,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    invalid_payloads = [
        _report_payload(
            execution_status="executed", outcome="passed",
            reason_code="provider_failed", completed_case_count=1,
            quality_report=passing_quality,
        ),
        _report_payload(
            execution_status="executed", outcome="passed", reason_code=None,
            failure_category=FailureCategory.PROVIDER,
            completed_case_count=1, quality_report=passing_quality,
        ),
        _report_payload(
            execution_status="executed", outcome="quality_failed",
            reason_code=None, completed_case_count=1,
            quality_report=failing_quality, policy_passed=False,
        ),
        _report_payload(
            execution_status="executed", outcome="quality_failed",
            reason_code="quality_gate_failed",
            failure_category=FailureCategory.PROVIDER,
            completed_case_count=1, quality_report=failing_quality,
            policy_passed=False,
        ),
        _report_payload(
            execution_status="failed", outcome="transport_failed",
            reason_code="provider_network_failed", failure=network_failure,
            completed_case_count=1, policy_passed=False,
        ),
        _report_payload(
            execution_status="failed", outcome="transport_failed",
            reason_code="provider_network_failed", failure=network_failure,
            quality_report=passing_quality, policy_passed=False,
        ),
        _report_payload(
            execution_status="failed", outcome="transport_failed",
            reason_code="provider_network_failed", failure=network_failure,
            policy_passed=True,
        ),
        _report_payload(
            execution_status="failed", outcome="contract_failed",
            reason_code="provider_failed", failure=provider_failure,
            policy_passed=False,
        ),
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            ProviderConformanceReport(**payload)


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.CONTRACT,
        FailureCategory.UNKNOWN,
        FailureCategory.SAFETY,
        FailureCategory.AUTHENTICATION,
    ],
)
def test_contract_failed_rejects_every_non_provider_category(category):
    failure = ProviderFailureEvidence(
        category=category,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="failed",
                outcome="contract_failed",
                reason_code="provider_response_malformed",
                failure=failure,
                policy_passed=False,
            )
        )


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.UNKNOWN,
        FailureCategory.SAFETY,
        FailureCategory.CONTRACT,
        FailureCategory.PII,
        FailureCategory.TOOL,
        FailureCategory.VALIDATION,
        FailureCategory.PROMPT_INJECTION,
    ],
)
def test_transport_failed_rejects_undocumented_categories(category):
    failure = ProviderFailureEvidence(
        category=category,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="failed",
                outcome="transport_failed",
                reason_code="provider_failed",
                failure=failure,
                policy_passed=False,
            )
        )


@pytest.mark.parametrize(
    ("category", "wrong_reason"),
    [
        (FailureCategory.AUTHENTICATION, "provider_network_failed"),
        (FailureCategory.AUTHORIZATION, "provider_timeout"),
        (FailureCategory.TIMEOUT, "provider_failed"),
        (FailureCategory.NETWORK, "provider_authentication_failed"),
        (FailureCategory.RATE_LIMIT, "provider_authentication_failed"),
        (FailureCategory.PROVIDER, "provider_timeout"),
    ],
)
def test_transport_failed_rejects_wrong_reason_for_allowed_category(
    category, wrong_reason
):
    failure = ProviderFailureEvidence(
        category=category,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="failed",
                outcome="transport_failed",
                reason_code=wrong_reason,
                failure=failure,
                policy_passed=False,
            )
        )


def test_report_rejects_type_coercion_and_is_frozen():
    with pytest.raises(ValidationError):
        ProviderConformanceReport(**_report_payload(case_count="1"))
    report = ProviderConformanceReport(**_report_payload())
    serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert serialized == json.dumps(report.model_dump(mode="json"), sort_keys=True)
    detached = report.model_dump(mode="json")
    detached["case_ids"].append("sk-detached-secret")
    assert "sk-detached-secret" not in serialize_provider_report(report)
    with pytest.raises(ValidationError):
        report.policy_passed = False


@pytest.mark.parametrize(
    "hostile",
    [
        "ghp_12345678901234567890",
        "AKIAIOSFODNN7EXAMPLE",
        "book-flight-now",
        "sk-live-secret-123",
        "Bearer-secret-token",
        "5551234567",
        "private.person@example.test",
        "aB9xQ2mN7pR4vT8zK6wY3cF5hJ1sL0dG",
        "opaque-0123456789abcdef",
        "case-0123456789abcdef",
        "opaque-deadbeefdeadbeef",
        "case-deadbeefdeadbeef",
    ],
    ids=[
        "github-token",
        "aws-key",
        "prompt-like",
        "api-key",
        "authorization",
        "phone",
        "email",
        "opaque-token",
        "spoofed-opaque",
        "spoofed-case",
        "spoofed-opaque-deadbeef",
        "spoofed-case-deadbeef",
    ],
)
def test_arbitrary_identifiers_are_opaque_in_reports_and_traces(hostile, prompt):
    exporter = InMemoryExporter()
    tracer = TracingFacade(
        settings=ObservabilitySettings(enabled=True, exporter="memory"),
        exporter=exporter,
    )
    case = GoldenCase(
        id=hostile,
        category="flight_search",
        user_input=RAW_PROMPT,
        context=RAW_CONTEXT,
        expected_intent="flight_search",
    )
    report = ProviderConformanceRunner(
        provider=None,
        provider_label=hostile,
        model_label=hostile,
        tracer=tracer,
    ).run([case], prompt=prompt)
    report_json = serialize_provider_report(report)
    report_dict = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    trace_json = json.dumps(
        [span.model_dump(mode="json") for span in exporter.spans], sort_keys=True
    )

    assert hostile not in report_json
    assert hostile not in report_dict
    assert hostile not in trace_json
    assert report.provider.startswith("opaque-")
    assert report.model.startswith("opaque-")
    assert report.case_ids[0].startswith("case-")


def test_nested_report_state_is_immutable(cases, prompt):
    report = _runner(
        StaticProvider([_assistant_response(), _assistant_response()]),
        evaluator=RecordingEvaluator(_quality_report(passed=True)),
    ).run(cases, prompt=prompt)
    before = serialize_provider_report(report)

    with pytest.raises(AttributeError):
        report.case_ids.append("sk-post-construction-secret")
    with pytest.raises(ValidationError):
        report.quality_report.intent_accuracy = "sk-post-construction-secret"

    failure_report = _runner(StaticProvider([RuntimeError("failed")])).run(
        cases, prompt=prompt
    )
    with pytest.raises(ValidationError):
        failure_report.failure.retryable = True
    assert serialize_provider_report(report) == before
    assert "sk-post-construction-secret" not in before


def test_provider_failure_requires_an_incomplete_case():
    failure = ProviderFailureEvidence(
        category=FailureCategory.TIMEOUT,
        retryable=True,
        attempt=1,
        max_attempts=1,
    )
    valid = ProviderConformanceReport(
        **_report_payload(
            outcome="transport_failed",
            execution_status="failed",
            reason_code="provider_timeout",
            case_count=2,
            completed_case_count=1,
            case_ids=["case-1", "case-2"],
            failure=failure,
            policy_passed=False,
        )
    )
    assert valid.completed_case_count < valid.case_count

    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                outcome="transport_failed",
                execution_status="failed",
                reason_code="provider_timeout",
                completed_case_count=1,
                failure=failure,
                policy_passed=False,
            )
        )


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.PROVIDER,
        FailureCategory.NETWORK,
        FailureCategory.TIMEOUT,
        FailureCategory.UNKNOWN,
    ],
)
def test_unavailable_accepts_availability_categories(category):
    report = ProviderConformanceReport(
        **_report_payload(
            execution_status="unavailable",
            outcome="unavailable",
            reason_code="provider_unavailable",
            failure=ProviderFailureEvidence(
                category=category,
                retryable=False,
                attempt=1,
                max_attempts=1,
            ),
        )
    )
    assert report.failure.category == category


@pytest.mark.parametrize(
    "category",
    [FailureCategory.AUTHENTICATION, FailureCategory.AUTHORIZATION, FailureCategory.CONTRACT],
)
def test_unavailable_rejects_unrelated_categories(category):
    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                execution_status="unavailable",
                outcome="unavailable",
                reason_code="provider_unavailable",
                failure=ProviderFailureEvidence(
                    category=category,
                    retryable=False,
                    attempt=1,
                    max_attempts=1,
                ),
            )
        )


@pytest.mark.parametrize(
    ("value", "status", "exit_code"),
    [("false", "not_executed", 0), ("true", "provider_required", 1)],
)
def test_environment_required_flag_drives_execution_policy(
    value, status, exit_code, cases, prompt
):
    provider = OptionalRealLLMProvider(
        environment={"AI_REAL_PROVIDER_REQUIRED": value},
        transport=httpx.MockTransport(
            lambda request: pytest.fail("unconfigured provider attempted network access")
        ),
    )
    try:
        report = ProviderConformanceRunner(
            provider=provider,
            provider_label="real_http",
            model_label="unconfigured",
        ).run(cases, prompt=prompt)
    finally:
        provider.close()

    assert report.required is (value == "true")
    assert report.execution_status == status
    assert report.outcome == "not_configured"
    assert conformance_exit_code(report) == exit_code


def test_real_provider_rejects_policy_that_conflicts_with_settings():
    provider = OptionalRealLLMProvider(
        environment={"AI_REAL_PROVIDER_REQUIRED": "true"}
    )
    try:
        with pytest.raises(ValueError, match="conflicts"):
            ProviderConformanceRunner(
                provider=provider,
                provider_label="real_http",
                model_label="unconfigured",
                policy=ProviderExecutionPolicy(required=False),
            )
    finally:
        provider.close()


@pytest.mark.parametrize(
    "hostile_model",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_12345678901234567890",
        "sk-live-secret-123",
        "Bearer-secret-token",
        "book-flight-now",
        "opaque-0123456789abcdef",
        "opaque-deadbeefdeadbeef",
    ],
    ids=[
        "aws-key",
        "github-token",
        "api-key",
        "authorization",
        "prompt-like",
        "spoofed-opaque",
        "spoofed-opaque-deadbeef",
    ],
)
def test_actual_provider_child_span_uses_safe_model_but_request_uses_raw_model(
    hostile_model, cases, prompt
):
    exporter = InMemoryExporter()
    tracer = TracingFacade(
        settings=ObservabilitySettings(enabled=True, exporter="memory"),
        exporter=exporter,
    )
    requested_models = []

    def handler(request):
        requested_models.append(json.loads(request.content)["model"])
        return _provider_payload()

    provider = OptionalRealLLMProvider(
        settings=RealLLMProviderSettings(
            enabled=True,
            base_url=ENDPOINT,
            model=hostile_model,
            timeout_seconds=5.0,
            max_attempts=1,
            require_structured_output=True,
        ),
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        tracer=tracer,
        sleeper=lambda delay: None,
    )
    try:
        report = ProviderConformanceRunner(
            provider=provider,
            provider_label="real_http",
            model_label=hostile_model,
            evaluator=RecordingEvaluator(_quality_report(passed=True)),
            tracer=tracer,
        ).run(cases, prompt=prompt)
    finally:
        provider.close()

    trace_json = json.dumps(
        [span.model_dump(mode="json") for span in exporter.spans], sort_keys=True
    )
    assert requested_models == [hostile_model, hostile_model, hostile_model]
    assert hostile_model not in trace_json
    assert hostile_model not in serialize_provider_report(report)
    assert len(exporter.spans) == 4
    assert [span.operation_name for span in exporter.spans].count("llm.generation") == 3
    assert all(
        span.attributes.get("model_name", "").startswith("opaque-")
        for span in exporter.spans
    )
    for sensitive in (API_KEY, ENDPOINT, RAW_PROMPT, RAW_CONTEXT, "Authorization"):
        assert sensitive not in trace_json


def test_legitimate_deterministic_model_label_remains_readable(cases, prompt):
    report = ProviderConformanceRunner(
        provider=StaticProvider([_assistant_response(), _assistant_response()]),
        provider_label="deterministic",
        model_label="rule-based-airline-assistant",
        evaluator=RecordingEvaluator(_quality_report(passed=True)),
    ).run(cases, prompt=prompt)
    assert report.provider == "deterministic"
    assert report.model == "rule-based-airline-assistant"


def test_generated_identifier_and_caller_supplied_same_string_use_distinct_paths():
    generated_model = safe_model_label("caller-controlled-model")
    caller_model = safe_model_label(generated_model.value)
    generated_case = safe_case_id("caller-controlled-case")
    caller_case = safe_case_id(generated_case.value)

    assert generated_model.value != caller_model.value
    assert generated_case.value != caller_case.value
    assert generated_model.value == safe_model_label("caller-controlled-model").value
    assert generated_case.value == safe_case_id("caller-controlled-case").value


def test_contract_failure_requires_an_incomplete_case():
    failure = ProviderFailureEvidence(
        category=FailureCategory.PROVIDER,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
    valid = ProviderConformanceReport(
        **_report_payload(
            outcome="contract_failed",
            execution_status="failed",
            reason_code="provider_response_malformed",
            case_count=2,
            completed_case_count=1,
            case_ids=["case-1", "case-2"],
            failure=failure,
            policy_passed=False,
        )
    )
    assert valid.completed_case_count < valid.case_count

    with pytest.raises(ValidationError):
        ProviderConformanceReport(
            **_report_payload(
                outcome="contract_failed",
                execution_status="failed",
                reason_code="provider_response_malformed",
                completed_case_count=1,
                failure=failure,
                policy_passed=False,
            )
        )


def test_quality_report_is_a_defensive_snapshot(cases, prompt):
    source = _quality_report(passed=True)
    report = _runner(
        StaticProvider([_assistant_response(), _assistant_response()]),
        evaluator=RecordingEvaluator(source),
    ).run(cases, prompt=prompt)
    before = serialize_provider_report(report)

    source.intent_accuracy = 0.0
    source.overall_passed = False

    assert serialize_provider_report(report) == before
    assert report.quality_report.intent_accuracy == 1.0
    assert report.quality_report.overall_passed is True
