from __future__ import annotations

import copy
import json
from types import FunctionType

import pytest
from pydantic import ValidationError

import ai_quality.trusted_provider_result as trusted_results
from ai_eval.models import Phase10Report
from ai_quality import provider_conformance
from ai_quality.dataset import (
    REGISTERED_DATASET_ID,
    REGISTERED_DATASET_VERSION,
    DatasetRegistry,
)
from ai_quality.evaluator_registry import (
    REGISTERED_EVALUATOR_ID,
    REGISTERED_EVALUATOR_VERSION,
    EvaluatorRegistry,
)
from ai_quality.execution_descriptor import TrustedExecutionDescriptor
from ai_quality.models import AirlineAssistantResponse
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.provider_capabilities import SchemaCompatibilityEvidence
from ai_quality.provider_conformance import (
    ProviderConformanceReport,
    ProviderConformanceRunner,
    adapt_trusted_provider_result_to_phase10,
)
from ai_quality.providers import LLMProvider
from ai_quality.trusted_provider_result import (
    TrustedProviderConformanceResult,
    _is_issued_trusted_result,
)


class _Provider(LLMProvider):
    def __init__(self):
        self.calls = 0

    def health_check(self) -> bool:
        return True

    def generate(self, user_input, *, prompt, context=None):
        return "unused"

    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        self.calls += 1
        return AirlineAssistantResponse(
            intent=case.expected_intent if case is not None else "flight_search",
            response="A bounded airline response.",
            prompt_version=prompt.version,
        )


class _FailingProvider(_Provider):
    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        self.calls += 1
        raise RuntimeError("controlled provider failure")


def _descriptor(*, prompt_version: str = "v1") -> TrustedExecutionDescriptor:
    return TrustedExecutionDescriptor.create(
        registered_prompt=PromptRegistry().resolve(
            "airline_assistant",
            prompt_version,
        ),
        registered_dataset=DatasetRegistry().resolve(
            REGISTERED_DATASET_ID,
            REGISTERED_DATASET_VERSION,
        ),
        registered_evaluator=EvaluatorRegistry().resolve(
            REGISTERED_EVALUATOR_ID,
            REGISTERED_EVALUATOR_VERSION,
        ),
        provider_settings=None,
        provider_required=False,
        provider_id="test-provider",
        model_id="test-model",
    )


def _runner() -> ProviderConformanceRunner:
    return ProviderConformanceRunner(
        provider=_Provider(),
        provider_label="test-provider",
        model_label="test-model",
    )


def _failing_runner() -> ProviderConformanceRunner:
    return ProviderConformanceRunner(
        provider=_FailingProvider(),
        provider_label="test-provider",
        model_label="test-model",
    )


def _public_not_configured_report() -> ProviderConformanceReport:
    descriptor = _descriptor()
    runner = ProviderConformanceRunner(
        provider=None,
        provider_label="test-provider",
        model_label="test-model",
    )
    return runner.run_trusted(descriptor).report.model_copy(deep=True)


def _trusted_result(*, prompt_version: str = "v1"):
    descriptor = _descriptor(prompt_version=prompt_version)
    return descriptor, _runner().run_trusted(descriptor)


def _phase10() -> Phase10Report:
    return Phase10Report(
        baseline={}, frameworks=[], comparisons=[], overall_passed=True
    )


def _nested_closure_values(function):
    pending = [function]
    seen = set()
    values = []
    while pending:
        current = pending.pop()
        if id(current) in seen or not isinstance(current, FunctionType):
            continue
        seen.add(id(current))
        for cell in current.__closure__ or ():
            value = cell.cell_contents
            values.append(value)
            if isinstance(value, FunctionType):
                pending.append(value)
    return values


def test_runner_mints_descriptor_bound_trusted_result_after_execution():
    descriptor, result = _trusted_result()

    assert isinstance(result, TrustedProviderConformanceResult)
    assert _is_issued_trusted_result(result)
    assert result.descriptor is descriptor
    assert result.provenance == result.report.provenance
    assert result.execution_evidence_id == result.provenance.execution_evidence_id
    assert result.outcome == result.report.outcome
    assert result.execution_status == result.report.execution_status


def test_direct_construction_and_public_evidence_promotion_are_rejected():
    _, result = _trusted_result()
    report = result.report.model_copy(deep=True)
    provenance = result.provenance.model_copy(deep=True)

    for value in (
        report,
        provenance,
        report.model_dump(mode="json"),
        report.model_dump_json(),
    ):
        with pytest.raises(TypeError, match="only be minted"):
            TrustedProviderConformanceResult(value)

    assert not hasattr(TrustedProviderConformanceResult, "from_report")
    assert not hasattr(TrustedProviderConformanceResult, "model_validate")
    assert not hasattr(TrustedProviderConformanceResult, "from_json")


def test_json_round_trip_remains_public_evidence_only():
    _, result = _trusted_result()
    parsed = ProviderConformanceReport.model_validate_json(
        result.report.model_dump_json()
    )

    assert parsed == result.report
    assert parsed is not result.report
    assert not _is_issued_trusted_result(parsed)


def test_forged_result_and_copied_public_metadata_are_not_enrolled():
    _, result = _trusted_result()
    forged = object.__new__(TrustedProviderConformanceResult)
    object.__setattr__(
        forged,
        "_TrustedProviderConformanceResult__descriptor",
        result.descriptor,
    )
    object.__setattr__(
        forged,
        "_TrustedProviderConformanceResult__report",
        result.report,
    )
    object.__setattr__(
        forged,
        "_TrustedProviderConformanceResult__execution_validator",
        lambda candidate: False,
    )

    assert not _is_issued_trusted_result(forged)
    with pytest.raises(TypeError, match="only be minted"):
        copy.copy(result)


def test_import_only_caller_has_no_mutable_authority_registry_or_mint_helper():
    for name in (
        "_CompletedExecution",
        "_PENDING_EXECUTIONS",
        "_ISSUED_RESULTS",
        "_RunnerResultCapability",
        "_mint_trusted_provider_result",
        "_enroll_provider_runner",
        "_bind_framework_trusted_execution",
    ):
        assert not hasattr(trusted_results, name)
    assert not hasattr(provider_conformance, "_bind_framework_trusted_execution")


@pytest.mark.parametrize("source", ["report", "copy", "serialized", "provenance"])
def test_caller_supplied_evidence_cannot_create_trusted_result(source):
    _, result = _trusted_result()
    values = {
        "report": result.report,
        "copy": result.report.model_copy(deep=True),
        "serialized": ProviderConformanceReport.model_validate_json(
            result.report.model_dump_json()
        ),
        "provenance": result.provenance,
    }

    with pytest.raises(TypeError, match="only be minted"):
        TrustedProviderConformanceResult(values[source])


def test_same_descriptor_matching_provenance_and_evidence_id_are_not_authority():
    descriptor, result = _trusted_result()
    forged = result.report.model_copy(deep=True)

    assert forged.provenance.execution_evidence_id == result.execution_evidence_id
    assert forged.provenance == result.provenance
    assert result.descriptor is descriptor
    with pytest.raises(TypeError, match="only be minted"):
        TrustedProviderConformanceResult(descriptor, forged)


def test_each_trusted_result_requires_a_distinct_real_runner_execution():
    descriptor = _descriptor()
    runner = _runner()
    first = runner.run_trusted(descriptor)
    first_execution_calls = runner.provider.calls
    second = runner.run_trusted(descriptor)

    assert first is not second
    assert _is_issued_trusted_result(first)
    assert _is_issued_trusted_result(second)
    assert first_execution_calls > 0
    assert runner.provider.calls == first_execution_calls * 2
    assert not hasattr(first, "issue_again")
    with pytest.raises(TypeError, match="only be minted"):
        copy.copy(first)


def test_instance_execution_override_cannot_supply_trusted_report():
    descriptor, source = _trusted_result()
    runner = _runner()
    runner._execute_trusted = lambda supplied: source.report.model_copy(deep=True)

    result = runner.run_trusted(descriptor)

    assert runner.provider.calls > 0
    assert _is_issued_trusted_result(result)
    assert result is not source


def test_alternate_instance_execution_hook_is_ignored():
    descriptor, source = _trusted_result()
    runner = _runner()
    runner.execute_trusted = lambda supplied: source.report.model_copy(deep=True)

    result = runner.run_trusted(descriptor)

    assert runner.provider.calls > 0
    assert _is_issued_trusted_result(result)


def test_instance_run_entrypoint_replacement_cannot_mint_trusted_result():
    descriptor, source = _trusted_result()
    runner = _runner()
    runner.run_trusted = lambda supplied: source.report.model_copy(deep=True)

    returned = runner.run_trusted(descriptor)

    assert isinstance(returned, ProviderConformanceReport)
    assert not _is_issued_trusted_result(returned)
    with pytest.raises(TypeError, match="runner-issued trusted result"):
        adapt_trusted_provider_result_to_phase10(_phase10(), returned)


def test_subclass_execution_overrides_cannot_supply_trusted_report():
    descriptor, source = _trusted_result()

    class FakeRunner(ProviderConformanceRunner):
        def _execute_trusted(self, supplied):
            return source.report.model_copy(deep=True)

        def _execute(self, *args, **kwargs):
            return source.report.model_copy(deep=True)

    runner = FakeRunner(
        provider=_Provider(),
        provider_label="test-provider",
        model_label="test-model",
    )
    result = runner.run_trusted(descriptor)

    assert runner.provider.calls > 0
    assert _is_issued_trusted_result(result)


def test_class_execution_replacement_does_not_redirect_bound_entrypoint(monkeypatch):
    descriptor, source = _trusted_result()
    runner = _runner()
    monkeypatch.setattr(
        ProviderConformanceRunner,
        "_execute_trusted",
        lambda self, supplied: source.report.model_copy(deep=True),
    )
    monkeypatch.setattr(
        ProviderConformanceRunner,
        "_execute",
        lambda self, *args, **kwargs: source.report.model_copy(deep=True),
    )

    result = runner.run_trusted(descriptor)

    assert runner.provider.calls > 0
    assert _is_issued_trusted_result(result)


@pytest.mark.parametrize("method_name", ["_run", "_report", "_failure_report"])
def test_instance_report_dispatch_replacement_cannot_mint_trusted_result(method_name):
    descriptor = _descriptor()
    copied_report = _public_not_configured_report()
    runner = _failing_runner() if method_name == "_failure_report" else _runner()
    calls = 0

    def fake(*args, **kwargs):
        nonlocal calls
        calls += 1
        return copied_report.model_copy(deep=True)

    setattr(runner, method_name, fake)
    result = runner.run_trusted(descriptor)
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)

    assert calls == 0
    assert _is_issued_trusted_result(result)
    assert result.outcome != copied_report.outcome
    assert adapted.baseline["provider_conformance"]["outcome"] == result.outcome


def test_subclass_report_dispatch_overrides_cannot_mint_trusted_result():
    descriptor = _descriptor()
    copied_report = _public_not_configured_report()

    class FakeRunner(ProviderConformanceRunner):
        def _run(self, *args, **kwargs):
            return copied_report.model_copy(deep=True)

        def _report(self, **kwargs):
            return copied_report.model_copy(deep=True)

        def _failure_report(self, *args, **kwargs):
            return copied_report.model_copy(deep=True)

    for provider in (_Provider(), _FailingProvider()):
        runner = FakeRunner(
            provider=provider,
            provider_label="test-provider",
            model_label="test-model",
        )
        result = runner.run_trusted(descriptor)
        adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)

        assert result.outcome != copied_report.outcome
        assert adapted.baseline["provider_conformance"]["outcome"] == result.outcome


@pytest.mark.parametrize("method_name", ["_run", "_report", "_failure_report"])
def test_class_report_dispatch_replacement_cannot_mint_trusted_result(
    monkeypatch, method_name
):
    descriptor = _descriptor()
    copied_report = _public_not_configured_report()
    runner = _failing_runner() if method_name == "_failure_report" else _runner()
    calls = 0

    def fake(*args, **kwargs):
        nonlocal calls
        calls += 1
        return copied_report.model_copy(deep=True)

    monkeypatch.setattr(ProviderConformanceRunner, method_name, fake)
    result = runner.run_trusted(descriptor)
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)

    assert calls == 0
    assert result.outcome != copied_report.outcome
    assert adapted.baseline["provider_conformance"]["outcome"] == result.outcome


@pytest.mark.parametrize("source", ["copy", "deserialized"])
def test_public_report_returned_through_run_cannot_be_promoted(source):
    descriptor = _descriptor()
    copied_report = _public_not_configured_report()
    evidence = {
        "copy": copied_report.model_copy(deep=True),
        "deserialized": ProviderConformanceReport.model_validate_json(
            copied_report.model_dump_json()
        ),
    }[source]
    runner = _runner()
    runner._run = lambda *args, **kwargs: evidence

    result = runner.run_trusted(descriptor)
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)

    assert result.outcome != evidence.outcome
    assert adapted.baseline["provider_conformance"]["outcome"] == result.outcome


def test_repeated_copied_report_issuance_attempts_execute_independently():
    descriptor = _descriptor()
    copied_report = _public_not_configured_report()
    runner = _runner()
    runner._run = lambda *args, **kwargs: copied_report.model_copy(deep=True)

    first = runner.run_trusted(descriptor)
    first_calls = runner.provider.calls
    second = runner.run_trusted(descriptor)

    assert first is not second
    assert first.outcome != copied_report.outcome
    assert second.outcome != copied_report.outcome
    assert runner.provider.calls == first_calls * 2
    adapt_trusted_provider_result_to_phase10(_phase10(), first)
    adapt_trusted_provider_result_to_phase10(_phase10(), second)


def test_trusted_closures_expose_no_authority_bearing_state_class():
    values = _nested_closure_values(ProviderConformanceRunner.run_trusted)

    assert not any(
        isinstance(value, type)
        and (
            hasattr(value, "_report")
            or hasattr(value, "_failure_report")
            or value.__name__ == "FrameworkExecutionState"
        )
        for value in values
    )


def test_closure_reachable_function_attributes_cannot_redirect_report_authority(
    monkeypatch,
):
    descriptor = _descriptor()
    copied_report = _public_not_configured_report()
    calls = 0

    def malicious_report(*args, **kwargs):
        nonlocal calls
        calls += 1
        return copied_report.model_copy(deep=True)

    values = _nested_closure_values(ProviderConformanceRunner.run_trusted)
    for value in values:
        if isinstance(value, FunctionType):
            monkeypatch.setattr(value, "_report", malicious_report, raising=False)
            monkeypatch.setattr(
                value,
                "_failure_report",
                malicious_report,
                raising=False,
            )

    runner = _runner()
    result = runner.run_trusted(descriptor)
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)

    assert calls == 0
    assert result.outcome != copied_report.outcome
    assert adapted.baseline["provider_conformance"]["outcome"] == result.outcome


@pytest.mark.parametrize(
    ("runner_factory", "expected_outcome"),
    [
        (_runner, "quality_failed"),
        (_failing_runner, "transport_failed"),
        (
            lambda: ProviderConformanceRunner(
                provider=None,
                provider_label="test-provider",
                model_label="test-model",
            ),
            "not_configured",
        ),
    ],
)
def test_module_schema_resolver_replacement_cannot_redirect_trusted_execution(
    monkeypatch,
    runner_factory,
    expected_outcome,
):
    calls = 0

    def malicious_resolver(provider, capability):
        nonlocal calls
        calls += 1
        return SchemaCompatibilityEvidence(
            declaration="unknown",
            empirical="not_run",
        )

    monkeypatch.setattr(
        provider_conformance,
        "schema_compatibility_for_declaration",
        malicious_resolver,
    )
    runner = runner_factory()
    result = runner.run_trusted(_descriptor())

    assert calls == 0
    assert result.outcome == expected_outcome
    assert result.report.schema_compatibility.declaration == "absent"
    assert result.report.provenance == result.provenance
    assert _is_issued_trusted_result(result)


def test_runner_schema_resolver_overrides_cannot_redirect_trusted_execution(
    monkeypatch,
):
    calls = 0

    def malicious_resolver(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SchemaCompatibilityEvidence(
            declaration="unknown",
            empirical="not_run",
        )

    class FakeRunner(ProviderConformanceRunner):
        schema_compatibility_for_declaration = staticmethod(malicious_resolver)

    monkeypatch.setattr(
        ProviderConformanceRunner,
        "schema_compatibility_for_declaration",
        staticmethod(malicious_resolver),
        raising=False,
    )
    runner = FakeRunner(
        provider=_Provider(),
        provider_label="test-provider",
        model_label="test-model",
    )
    runner.schema_compatibility_for_declaration = malicious_resolver
    result = runner.run_trusted(_descriptor())

    assert calls == 0
    assert result.outcome == "quality_failed"
    assert result.report.schema_compatibility.declaration == "absent"
    assert result.report.provenance == result.provenance
    assert _is_issued_trusted_result(result)


@pytest.mark.parametrize(
    ("runner_factory", "expected_outcome"),
    [
        (_runner, "quality_failed"),
        (_failing_runner, "transport_failed"),
        (
            lambda: ProviderConformanceRunner(
                provider=None,
                provider_label="test-provider",
                model_label="test-model",
            ),
            "not_configured",
        ),
    ],
)
def test_pydantic_validator_replacement_cannot_redirect_trusted_validation(
    monkeypatch,
    runner_factory,
    expected_outcome,
):
    _, source = _trusted_result()
    calls = 0

    class MaliciousValidator:
        def validate_python(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return source.report

    def malicious_model_validate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return source.report

    monkeypatch.setattr(
        ProviderConformanceReport,
        "__pydantic_validator__",
        MaliciousValidator(),
    )
    monkeypatch.setattr(
        ProviderConformanceReport,
        "model_validate",
        malicious_model_validate,
    )

    runner = runner_factory()
    result = runner.run_trusted(_descriptor())

    assert calls == 0
    assert result.outcome == expected_outcome
    assert result.report.provenance == result.provenance
    assert _is_issued_trusted_result(result)
    if runner.provider is None:
        assert result.execution_status == "not_executed"
        assert result.report.completed_case_count == 0


def test_trusted_result_is_consumed_once_by_phase10():
    _, result = _trusted_result()

    adapt_trusted_provider_result_to_phase10(_phase10(), result)

    with pytest.raises(TypeError, match="already been consumed"):
        adapt_trusted_provider_result_to_phase10(_phase10(), result)


def test_result_and_report_snapshot_are_immutable_and_independent():
    descriptor = _descriptor()
    runner = _runner()
    result = runner.run_trusted(descriptor)
    source = result.report.model_copy(deep=True)

    with pytest.raises(AttributeError, match="immutable"):
        result.report = result.report
    with pytest.raises(ValidationError):
        result.report.outcome = "passed"
    with pytest.raises(AttributeError, match="immutable"):
        descriptor._model_id = "substituted"
    object.__setattr__(source, "outcome", "passed")

    assert result.descriptor is descriptor
    assert result.report is not result.report.model_copy(deep=True)
    assert result.report.outcome != source.outcome
    assert _is_issued_trusted_result(result)


def test_trusted_result_has_no_serialization_or_sensitive_projection():
    descriptor, result = _trusted_result()
    representation = repr(result)
    public_json = result.report.model_dump_json()

    with pytest.raises(TypeError):
        json.dumps(result)
    with pytest.raises(TypeError):
        vars(result)
    assert not hasattr(result, "model_dump")
    assert not hasattr(result, "capability")

    forbidden = (
        descriptor.prompt.prompt.text,
        descriptor.prompt.prompt.purpose,
        descriptor.dataset._canonical_source,
        "_capability",
        "completed_execution",
        "trusted_execution",
        "Authorization",
        "Bearer",
        "api_key",
        "cookie",
        "https://",
        "request",
        "response text",
    )
    for value in forbidden:
        assert value not in representation
        assert value not in public_json


def test_registered_public_run_remains_a_public_report():
    descriptor = _descriptor()
    report = _runner().run(
        descriptor.dataset,
        prompt=descriptor.prompt,
    )

    assert isinstance(report, ProviderConformanceReport)
    assert not isinstance(report, TrustedProviderConformanceResult)
    assert not _is_issued_trusted_result(report)


def test_runner_issues_trusted_result_with_actual_failure_outcome():
    descriptor = _descriptor()
    runner = ProviderConformanceRunner(
        provider=None,
        provider_label="test-provider",
        model_label="test-model",
    )

    result = runner.run_trusted(descriptor)

    assert _is_issued_trusted_result(result)
    assert result.outcome == "not_configured"
    assert result.execution_status == "not_executed"
    assert result.report.policy_passed
