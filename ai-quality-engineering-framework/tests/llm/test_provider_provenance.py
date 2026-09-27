from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_quality.dataset import (
    REGISTERED_DATASET_ID,
    REGISTERED_DATASET_VERSION,
    DatasetRegistry,
    GoldenCaseRevision,
    GoldenDatasetIdentity,
    load_golden_cases,
    load_golden_dataset_identity,
    resolve_golden_dataset_identity,
)
from ai_quality.execution_descriptor import TrustedExecutionDescriptor
from ai_quality.evaluators import QualityGateEvaluator
from ai_quality.evaluator_registry import (
    REGISTERED_EVALUATOR_ID,
    REGISTERED_EVALUATOR_VERSION,
    EvaluatorRegistry,
)
from ai_quality.models import GoldenCase
from ai_quality.privacy import safe_case_id, safe_model_label, safe_provider_label
from ai_quality.prompt_registry import Prompt, PromptRegistry, registered_prompt_identity
from ai_quality.provider_capabilities import ProviderCapabilityRequirement
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.provider_provenance import (
    ExecutionPolicyProvenance,
    ProviderConformanceProvenance,
    ProviderIdentityProvenance,
    _ConformanceExecutionDescriptor,
    _build_provider_conformance_provenance,
    _build_trusted_provider_conformance_provenance,
    _evaluator_declaration,
)
from ai_quality.thresholds import AIQualityThresholds

pytestmark = pytest.mark.real_llm


def _settings(**overrides):
    values = dict(enabled=True, base_url="https://provider.example/v1/chat",
                  model="test-model", timeout_seconds=10.0, max_attempts=3,
                  max_retry_delay_seconds=5.0, require_structured_output=True,
                  required=False)
    values.update(overrides)
    return RealLLMProviderSettings(**values)


def _dataset(*, version="1.0", revisions=("1", "1"), ids=("case-001", "case-002")):
    return GoldenDatasetIdentity(
        dataset_id="airline-quality-golden-cases",
        dataset_version=version,
        cases=tuple(GoldenCaseRevision(case_id=case_id, revision=revision)
                    for case_id, revision in zip(ids, revisions)),
    )


def _descriptor(**overrides):
    settings = overrides.pop("settings", _settings())
    provider_id = overrides.pop("provider_id", safe_provider_label("real_http").value)
    model_id = overrides.pop("model_id", safe_model_label("test-model").value)
    if overrides:
        raise TypeError(f"unsupported trusted descriptor overrides: {sorted(overrides)}")
    return TrustedExecutionDescriptor.create(
        registered_prompt=PromptRegistry().resolve("airline_assistant", "v1"),
        registered_dataset=DatasetRegistry().resolve(
            REGISTERED_DATASET_ID,
            REGISTERED_DATASET_VERSION,
        ),
        registered_evaluator=EvaluatorRegistry().resolve(
            REGISTERED_EVALUATOR_ID,
            REGISTERED_EVALUATOR_VERSION,
        ),
        provider_settings=settings,
        provider_required=False,
        provider_id=provider_id,
        model_id=model_id,
    )


def _projection(descriptor, **overrides):
    values = dict(
        prompt_id=descriptor.prompt.prompt.name,
        prompt_version=descriptor.prompt.prompt.version,
        dataset=descriptor.dataset._identity,
        safe_case_ids=tuple(
            safe_case_id(item).value for item in descriptor.dataset.case_ids
        ),
        evaluator=_evaluator_declaration(descriptor.evaluator),
        provider_id=descriptor.provider_id,
        model_id=descriptor.model_id,
        settings=descriptor.provider_settings,
        provider_required=descriptor.provider_settings.provider_required,
        capability_requirement=descriptor.capability_requirement,
        trusted_source=descriptor,
    )
    values.update(overrides)
    return _ConformanceExecutionDescriptor(**values)


def _build(**overrides):
    return _build_trusted_provider_conformance_provenance(_descriptor(**overrides))


def _serialized(value):
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def test_contract_is_frozen_strict_deterministic_and_self_verifying():
    first = _build()
    second = _build()
    assert first == second
    assert _serialized(first) == _serialized(second)
    with pytest.raises(ValidationError):
        first.provenance_id = "provenance-" + ("0" * 64)
    payload = first.model_dump(mode="python")
    payload["provenance_id"] = "provenance-" + ("0" * 64)
    with pytest.raises(ValidationError, match="does not match"):
        ProviderConformanceProvenance.model_validate(payload)


@pytest.mark.parametrize("name", [
    "ignore_previous_instructions", "john_doe", "AKIAIOSFODNN7EXAMPLE",
    "ghp_12345678901234567890", "private.person@example.test", "secret-like-value",
])
def test_arbitrary_prompt_names_use_fixed_unregistered_identity(name):
    prompt = Prompt(name=name, version="v1", purpose=name, text="raw secret prompt")
    assert registered_prompt_identity(prompt) == ("unregistered", "0")


def test_only_exact_registered_prompt_pairs_are_trusted():
    registered = PromptRegistry().resolve("airline_assistant", "v1")
    assert registered_prompt_identity(registered) == ("airline_assistant", "v1")
    assert registered_prompt_identity(Prompt("airline_assistant", "v1", "x", "x")) == ("unregistered", "0")
    assert registered_prompt_identity(Prompt("AIRLINE_ASSISTANT", "v1", "x", "x")) == ("unregistered", "0")
    assert registered_prompt_identity(Prompt("airline_assistant", "v3", "x", "x")) == ("unregistered", "0")


def test_manifest_is_privacy_safe_and_matches_default_dataset():
    manifest = load_golden_dataset_identity()
    cases = load_golden_cases()
    assert tuple(case.id for case in cases) == tuple(
        item.case_id for item in manifest.cases
    )
    assert resolve_golden_dataset_identity(cases).dataset_id == "unregistered"
    serialized = json.dumps({"dataset_id": manifest.dataset_id, "dataset_version": manifest.dataset_version,
                             "cases": [item.__dict__ for item in manifest.cases]}, sort_keys=True)
    for value in (cases[0].user_input, cases[1].expected_entities.get("booking_reference"), "context", "expected_response"):
        if value:
            assert str(value) not in serialized


@pytest.mark.parametrize("changed", [
    _dataset(version="1.1"),
    _dataset(revisions=("2", "1")),
    _dataset(ids=("case-002", "case-001")),
    _dataset(ids=("case-001", "case-001")),
])
def test_dataset_version_revision_order_and_duplicates_change_identity(changed):
    with pytest.raises(ValueError, match="must match its trusted source"):
        _projection(_descriptor(), dataset=changed)


def test_unregistered_cases_receive_fixed_identity_without_hashing_content():
    first = [GoldenCase(id="x", category="a", user_input="secret one", expected_intent="a")]
    second = [GoldenCase(id="x", category="b", user_input="secret two", context="private", expected_intent="b")]
    assert resolve_golden_dataset_identity(first) == resolve_golden_dataset_identity(second)
    assert resolve_golden_dataset_identity(first).dataset_id == "unregistered"


def test_builtin_thresholds_and_versions_drive_evaluator_identity():
    baseline = _evaluator_declaration(
        EvaluatorRegistry().resolve(
            REGISTERED_EVALUATOR_ID,
            REGISTERED_EVALUATOR_VERSION,
        )
    )
    plain_default = _evaluator_declaration(QualityGateEvaluator())
    plain_changed = _evaluator_declaration(
        QualityGateEvaluator(AIQualityThresholds(intent_accuracy=0.90))
    )
    assert plain_default == plain_changed
    assert plain_default.evaluator_id == "unregistered"
    assert baseline.evaluator_id == REGISTERED_EVALUATOR_ID
    declaration = baseline.model_copy(update={"evaluator_version": "2.0"})
    with pytest.raises(ValueError, match="must match its trusted source"):
        _projection(_descriptor(), evaluator=declaration)
    policy = baseline.model_copy(update={"policy_version": "2.0"})
    with pytest.raises(ValueError, match="must match its trusted source"):
        _projection(_descriptor(), evaluator=policy)


def test_custom_evaluator_self_declaration_is_ignored():
    class Custom:
        def provenance_declaration(self):
            return {
                "evaluator_id": REGISTERED_EVALUATOR_ID,
                "evaluator_version": REGISTERED_EVALUATOR_VERSION,
                "policy_id": "quality-gate-thresholds",
                "policy_version": "1.0",
                "policy_fingerprint": "evaluator-policy-" + ("0" * 64),
            }

    declaration = _evaluator_declaration(Custom())
    assert declaration.evaluator_id == "unregistered"
    assert declaration.evaluator_version == "0"


def test_safe_policy_and_provider_changes_drift_only_relevant_components():
    baseline = _build()
    provider = _build(provider_id=safe_provider_label("provider-a").value)
    policy = _build(settings=_settings(max_attempts=4))
    assert provider.provider.provider_identity_id != baseline.provider.provider_identity_id
    assert provider.evaluation == baseline.evaluation
    assert policy.execution_policy.execution_policy_id != baseline.execution_policy.execution_policy_id
    assert policy.provider == baseline.provider


def test_urls_enablement_and_raw_model_do_not_affect_provenance():
    baseline = _build(settings=_settings())
    changed = _build(settings=_settings(enabled=False,
        base_url="https://user:password@private.example/path?api_key=secret",
        model="raw-secret-model"))
    assert changed == baseline


def test_sensitive_content_is_absent_and_cannot_influence_identity():
    hostile = ["raw prompt ABC123", "raw response P1234567", "private context",
               "Authorization: Bearer secret", "cookie=session", "https://private.example/path"]
    evidence = _build()
    serialized = _serialized(evidence)
    for value in hostile:
        assert value not in serialized
    assert _build().provenance_id == evidence.provenance_id


@pytest.mark.parametrize("component", ["provider", "evaluation", "contract", "execution_policy", "aggregate"])
def test_component_and_aggregate_tampering_is_rejected(component):
    evidence = _build()
    payload = evidence.model_dump(mode="python")
    if component == "provider":
        payload["provider"]["provider_id"] = "provider-a"
    elif component == "evaluation":
        payload["evaluation"]["prompt_version"] = "v2"
    elif component == "contract":
        payload["contract"]["structured_output_required"] = False
    elif component == "execution_policy":
        payload["execution_policy"]["max_attempts"] = 4
    else:
        payload["provenance_id"] = "provenance-" + ("0" * 64)
    with pytest.raises(ValidationError):
        ProviderConformanceProvenance.model_validate(payload)


def test_nested_contracts_reject_unsafe_or_incomplete_values():
    with pytest.raises(ValidationError):
        ProviderIdentityProvenance(provider_id="Bearer private-token", model_id="test-model")
    with pytest.raises(ValidationError):
        ExecutionPolicyProvenance(timeout_seconds=10.0, max_attempts=None,
            max_retry_delay_seconds=5.0,
            provider_required=False, retry_policy_id="bounded-http-retry",
            execution_policy_id="execution-policy-" + ("0" * 64))
