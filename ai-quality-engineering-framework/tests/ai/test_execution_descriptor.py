from __future__ import annotations

import json

import pytest

import ai_quality.dataset as dataset_module
import ai_quality.evaluator_registry as evaluator_module
import ai_quality.execution_descriptor as descriptor_module
import ai_quality.prompt_registry as prompt_module
from ai_eval.models import Phase10Report
from ai_quality.dataset import (
    REGISTERED_DATASET_ID,
    REGISTERED_DATASET_VERSION,
    DatasetRegistry,
    RegisteredDatasetBundle,
)
from ai_quality.evaluator_registry import (
    REGISTERED_EVALUATOR_ID,
    REGISTERED_EVALUATOR_VERSION,
    EvaluatorRegistry,
    RegisteredEvaluator,
)
from ai_quality.evaluators import QualityGateEvaluator
from ai_quality.execution_descriptor import TrustedExecutionDescriptor
from ai_quality.models import AirlineAssistantResponse, GoldenCase
from ai_quality.prompt_registry import Prompt, PromptRegistry, RegisteredPrompt
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.provider_conformance import (
    ProviderConformanceRunner,
    adapt_trusted_provider_result_to_phase10,
)
from ai_quality.providers import LLMProvider


def _handles():
    return (
        PromptRegistry().resolve("airline_assistant", "v1"),
        DatasetRegistry().resolve(
            REGISTERED_DATASET_ID,
            REGISTERED_DATASET_VERSION,
        ),
        EvaluatorRegistry().resolve(
            REGISTERED_EVALUATOR_ID,
            REGISTERED_EVALUATOR_VERSION,
        ),
    )


def _descriptor(**overrides) -> TrustedExecutionDescriptor:
    prompt, dataset, evaluator = _handles()
    values = {
        "registered_prompt": prompt,
        "registered_dataset": dataset,
        "registered_evaluator": evaluator,
        "provider_settings": None,
        "provider_required": False,
        "provider_id": "test-provider",
        "model_id": "test-model",
    }
    values.update(overrides)
    return TrustedExecutionDescriptor.create(**values)


class _RecordingProvider(LLMProvider):
    def __init__(self):
        self.prompts = []
        self.case_ids = []
        self.user_inputs = []

    def health_check(self) -> bool:
        return True

    def generate(self, user_input, *, prompt, context=None):
        return "unused"

    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        self.prompts.append(prompt)
        self.user_inputs.append(user_input)
        if case is not None:
            self.case_ids.append(case.id)
        return AirlineAssistantResponse(
            intent="flight_search",
            response="I can help with flights.",
            prompt_version="v1",
        )


def test_registered_handles_create_immutable_descriptor():
    descriptor = _descriptor()
    prompt, dataset, evaluator = (
        descriptor.prompt,
        descriptor.dataset,
        descriptor.evaluator,
    )

    assert isinstance(prompt, RegisteredPrompt)
    assert isinstance(dataset, RegisteredDatasetBundle)
    assert isinstance(evaluator, RegisteredEvaluator)
    with pytest.raises(AttributeError, match="immutable"):
        descriptor.prompt = Prompt("x", "v1", "x", "x")
    with pytest.raises(AttributeError):
        descriptor.provider_settings.max_attempts = 5
    with pytest.raises(AttributeError):
        descriptor.contract.contract_version = "2.0"


@pytest.mark.parametrize(
    "override",
    [
        {"registered_prompt": Prompt("airline_assistant", "v1", "x", "x")},
        {"registered_dataset": [GoldenCase(id="x", category="x", user_input="x", expected_intent="x")]},
        {"registered_evaluator": QualityGateEvaluator()},
    ],
)
def test_plain_matching_objects_cannot_create_trusted_descriptor(override):
    with pytest.raises(ValueError, match="requires a"):
        _descriptor(**override)


def test_matching_public_objects_cannot_manufacture_authority():
    prompt, dataset, evaluator = _handles()
    public_prompt = Prompt(
        prompt.prompt.name,
        prompt.prompt.version,
        prompt.prompt.purpose,
        prompt.prompt.text,
    )
    public_cases = [case.model_copy(deep=True) for case in dataset.materialize_cases()]
    public_evaluator = QualityGateEvaluator(evaluator._policy)

    for override in (
        {"registered_prompt": public_prompt},
        {"registered_dataset": public_cases},
        {"registered_evaluator": public_evaluator},
    ):
        with pytest.raises(ValueError, match="requires a"):
            _descriptor(**override)


def test_forged_registered_handles_are_only_identity_requests():
    prompt, dataset, evaluator = _handles()
    forged_prompt = object.__new__(RegisteredPrompt)
    object.__setattr__(forged_prompt, "_prompt", prompt.prompt)
    object.__setattr__(forged_prompt, "_enrollment_proof", object())
    forged_dataset = object.__new__(RegisteredDatasetBundle)
    object.__setattr__(forged_dataset, "_identity", dataset._identity)
    object.__setattr__(forged_dataset, "_canonical_source", dataset._canonical_source)
    object.__setattr__(forged_dataset, "_enrollment_proof", object())
    forged_evaluator = object.__new__(RegisteredEvaluator)
    object.__setattr__(forged_evaluator, "_declaration", evaluator._declaration)
    object.__setattr__(forged_evaluator, "_evaluator", evaluator._evaluator)
    object.__setattr__(forged_evaluator, "_policy", evaluator._policy)
    object.__setattr__(forged_evaluator, "_enrollment_proof", object())

    prompt_descriptor = _descriptor(registered_prompt=forged_prompt)
    dataset_descriptor = _descriptor(registered_dataset=forged_dataset)
    evaluator_descriptor = _descriptor(registered_evaluator=forged_evaluator)

    assert prompt_descriptor.prompt is not forged_prompt
    assert dataset_descriptor.dataset is not forged_dataset
    assert evaluator_descriptor.evaluator is not forged_evaluator


def test_same_class_forged_proofs_cannot_supply_trusted_execution_state():
    prompt, dataset, evaluator = _handles()

    attacker_prompt = Prompt(
        prompt.prompt.name,
        prompt.prompt.version,
        "attacker",
        "ATTACKER CONTROLLED PROMPT",
    )
    forged_prompt = object.__new__(RegisteredPrompt)
    object.__setattr__(forged_prompt, "_prompt", attacker_prompt)
    prompt_proof = object.__new__(type(prompt._enrollment_proof))
    object.__setattr__(prompt_proof, "_validator", lambda candidate: True)
    object.__setattr__(forged_prompt, "_enrollment_proof", prompt_proof)

    attacker_cases = json.loads(dataset._canonical_source)
    attacker_cases[0]["user_input"] = "ATTACKER CONTROLLED DATASET"
    forged_dataset = object.__new__(RegisteredDatasetBundle)
    object.__setattr__(forged_dataset, "_identity", dataset._identity)
    object.__setattr__(
        forged_dataset,
        "_canonical_source",
        json.dumps(attacker_cases, sort_keys=True),
    )
    dataset_proof = object.__new__(type(dataset._enrollment_proof))
    object.__setattr__(dataset_proof, "_validator", lambda candidate: True)
    object.__setattr__(forged_dataset, "_enrollment_proof", dataset_proof)

    class AttackerEvaluator:
        def __init__(self):
            self.thresholds = evaluator._policy
            self.calls = 0

        def evaluate(self, cases, responses):
            self.calls += 1
            raise AssertionError("attacker evaluator executed")

    attacker_evaluator = AttackerEvaluator()
    forged_evaluator = object.__new__(RegisteredEvaluator)
    object.__setattr__(forged_evaluator, "_declaration", evaluator._declaration)
    object.__setattr__(forged_evaluator, "_policy", evaluator._policy)
    object.__setattr__(forged_evaluator, "_evaluator", attacker_evaluator)
    evaluator_proof = object.__new__(type(evaluator._enrollment_proof))
    object.__setattr__(evaluator_proof, "_validator", lambda candidate: True)
    object.__setattr__(forged_evaluator, "_enrollment_proof", evaluator_proof)

    assert type(prompt_proof) is type(prompt._enrollment_proof)
    assert type(dataset_proof) is type(dataset._enrollment_proof)
    assert type(evaluator_proof) is type(evaluator._enrollment_proof)

    descriptor = _descriptor(
        registered_prompt=forged_prompt,
        registered_dataset=forged_dataset,
        registered_evaluator=forged_evaluator,
    )
    provider = _RecordingProvider()
    result = ProviderConformanceRunner(
        provider=provider,
        provider_label="test-provider",
        model_label="test-model",
    ).run_trusted(descriptor)
    phase10 = adapt_trusted_provider_result_to_phase10(
        Phase10Report(
            baseline={},
            frameworks=[],
            comparisons=[],
            overall_passed=True,
        ),
        result,
    )

    assert descriptor.prompt.prompt.text != attacker_prompt.text
    assert descriptor.dataset._canonical_source != forged_dataset._canonical_source
    assert descriptor.evaluator._evaluator is not attacker_evaluator
    assert all(prompt.text != attacker_prompt.text for prompt in provider.prompts)
    assert "ATTACKER CONTROLLED DATASET" not in provider.user_inputs
    assert attacker_evaluator.calls == 0
    assert result.provenance.evaluation.prompt_id == "airline_assistant"
    assert result.provenance.evaluation.golden_dataset_id == REGISTERED_DATASET_ID
    assert result.provenance.evaluation.evaluator_id == REGISTERED_EVALUATOR_ID
    assert "provider_conformance" in phase10.baseline


def test_enrollment_has_no_public_issuer_or_mutable_authority_registry():
    for module, forbidden in (
        (prompt_module, ("_issue_registered_prompt", "_ISSUED_PROMPTS")),
        (dataset_module, ("_issue_registered_dataset", "_ISSUED_DATASETS")),
        (evaluator_module, ("_issue_registered_evaluator", "_ISSUED_EVALUATORS")),
    ):
        for name in forbidden:
            assert not hasattr(module, name)


def test_module_registration_replacement_cannot_add_enrollment(monkeypatch):
    monkeypatch.setattr(prompt_module, "_REGISTERED_PROMPT_PATHS", {})
    monkeypatch.setattr(dataset_module, "_REGISTERED_DATASETS", {})
    monkeypatch.setattr(evaluator_module, "_REGISTERED_EVALUATORS", {})

    prompt, dataset, evaluator = _handles()

    assert prompt.prompt.name == "airline_assistant"
    assert dataset.dataset_id == REGISTERED_DATASET_ID
    assert evaluator.evaluator_id == REGISTERED_EVALUATOR_ID


def test_evaluator_constructor_and_factory_replacement_cannot_redirect_canonicalization(
    monkeypatch,
):
    calls = 0

    class AttackerEvaluator:
        def __init__(self, policy):
            nonlocal calls
            calls += 1
            self.thresholds = policy

    monkeypatch.setattr(evaluator_module, "QualityGateEvaluator", AttackerEvaluator)
    monkeypatch.setattr(
        evaluator_module,
        "_quality_gate_factory",
        lambda policy: AttackerEvaluator(policy),
    )

    descriptor = _descriptor()

    assert calls == 0
    assert isinstance(descriptor.evaluator._evaluator, QualityGateEvaluator)


def test_exposed_evaluator_policy_mutation_cannot_redirect_canonical_policy():
    _, _, evaluator = _handles()
    original = evaluator._policy.intent_accuracy
    try:
        object.__setattr__(evaluator._policy, "intent_accuracy", 0.0)
        descriptor = _descriptor(registered_evaluator=evaluator)
    finally:
        object.__setattr__(evaluator._policy, "intent_accuracy", original)

    assert descriptor.evaluator._policy.intent_accuracy == original
    assert descriptor.evaluator._policy is not evaluator._policy


def test_canonicalizer_module_replacement_cannot_redirect_descriptor(monkeypatch):
    prompt, dataset, evaluator = _handles()
    forged_prompt = object.__new__(RegisteredPrompt)
    object.__setattr__(forged_prompt, "_prompt", prompt.prompt)
    object.__setattr__(forged_prompt, "_enrollment_proof", object())
    forged_dataset = object.__new__(RegisteredDatasetBundle)
    object.__setattr__(forged_dataset, "_identity", dataset._identity)
    object.__setattr__(forged_dataset, "_canonical_source", dataset._canonical_source)
    object.__setattr__(forged_dataset, "_enrollment_proof", object())
    forged_evaluator = object.__new__(RegisteredEvaluator)
    object.__setattr__(forged_evaluator, "_declaration", evaluator._declaration)
    object.__setattr__(forged_evaluator, "_evaluator", evaluator._evaluator)
    object.__setattr__(forged_evaluator, "_policy", evaluator._policy)
    object.__setattr__(forged_evaluator, "_enrollment_proof", object())

    monkeypatch.setattr(
        descriptor_module,
        "_canonicalize_registered_prompt",
        lambda value: forged_prompt,
    )
    monkeypatch.setattr(
        descriptor_module,
        "_canonicalize_registered_dataset",
        lambda value: forged_dataset,
    )
    monkeypatch.setattr(
        descriptor_module,
        "_canonicalize_registered_evaluator",
        lambda value: forged_evaluator,
    )

    descriptor = _descriptor()
    assert descriptor.prompt is not forged_prompt
    assert descriptor.dataset is not forged_dataset
    assert descriptor.evaluator is not forged_evaluator


def test_registry_instance_class_and_subclass_substitution_cannot_supply_execution(
    monkeypatch,
):
    prompt, _, _ = _handles()
    forged = object.__new__(RegisteredPrompt)
    object.__setattr__(forged, "_prompt", prompt.prompt)
    object.__setattr__(forged, "_enrollment_proof", object())

    instance = PromptRegistry()
    instance.resolve = lambda *args, **kwargs: forged
    assert _descriptor(
        registered_prompt=instance.resolve("airline_assistant", "v1")
    ).prompt is not forged

    class FakeRegistry(PromptRegistry):
        def resolve(self, *args, **kwargs):
            return forged

    assert _descriptor(
        registered_prompt=FakeRegistry().resolve("airline_assistant", "v1")
    ).prompt is not forged

    monkeypatch.setattr(PromptRegistry, "resolve", lambda *args, **kwargs: forged)
    assert _descriptor(
        registered_prompt=PromptRegistry().resolve("airline_assistant", "v1")
    ).prompt is not forged


@pytest.mark.parametrize(
    ("registry_type", "resolve_args", "override_name", "handle_index"),
    [
        (PromptRegistry, ("airline_assistant", "v1"), "registered_prompt", 0),
        (
            DatasetRegistry,
            (REGISTERED_DATASET_ID, REGISTERED_DATASET_VERSION),
            "registered_dataset",
            1,
        ),
        (
            EvaluatorRegistry,
            (REGISTERED_EVALUATOR_ID, REGISTERED_EVALUATOR_VERSION),
            "registered_evaluator",
            2,
        ),
    ],
)
def test_each_registry_substitution_cannot_supply_execution_state(
    monkeypatch,
    registry_type,
    resolve_args,
    override_name,
    handle_index,
):
    prompt, dataset, evaluator = _handles()
    if handle_index == 0:
        forged = object.__new__(RegisteredPrompt)
        object.__setattr__(forged, "_prompt", prompt.prompt)
    elif handle_index == 1:
        forged = object.__new__(RegisteredDatasetBundle)
        object.__setattr__(forged, "_identity", dataset._identity)
        object.__setattr__(forged, "_canonical_source", dataset._canonical_source)
    else:
        forged = object.__new__(RegisteredEvaluator)
        object.__setattr__(forged, "_declaration", evaluator._declaration)
        object.__setattr__(forged, "_evaluator", evaluator._evaluator)
        object.__setattr__(forged, "_policy", evaluator._policy)
    object.__setattr__(forged, "_enrollment_proof", object())

    instance = registry_type()
    instance.resolve = lambda *args, **kwargs: forged

    class FakeRegistry(registry_type):
        def resolve(self, *args, **kwargs):
            return forged

    for resolved in (
        instance.resolve(*resolve_args),
        FakeRegistry().resolve(*resolve_args),
    ):
        descriptor = _descriptor(**{override_name: resolved})
        assert (descriptor.prompt, descriptor.dataset, descriptor.evaluator)[
            handle_index
        ] is not forged

    monkeypatch.setattr(registry_type, "resolve", lambda *args, **kwargs: forged)
    descriptor = _descriptor(
        **{override_name: registry_type().resolve(*resolve_args)}
    )
    assert (descriptor.prompt, descriptor.dataset, descriptor.evaluator)[
        handle_index
    ] is not forged


def test_invalid_or_conflicting_provider_settings_are_rejected():
    invalid = RealLLMProviderSettings(max_attempts=0)
    with pytest.raises(ValueError):
        _descriptor(provider_settings=invalid)

    conflicting = RealLLMProviderSettings(required=True)
    with pytest.raises(ValueError, match="conflict"):
        _descriptor(provider_settings=conflicting, provider_required=False)


def test_contract_constants_cannot_be_supplied_or_replaced():
    with pytest.raises(TypeError):
        TrustedExecutionDescriptor.create(contract_version="2.0")
    descriptor = _descriptor()
    with pytest.raises(AttributeError, match="immutable"):
        descriptor._contract = object()


def test_runner_executes_only_descriptor_components():
    descriptor = _descriptor()
    caller_cases = descriptor.dataset.materialize_cases()
    caller_cases[0].id = "caller-substitution"
    caller_prompt = Prompt("caller", "v1", "caller", "caller text")
    caller_evaluator = QualityGateEvaluator()
    provider = _RecordingProvider()
    runner = ProviderConformanceRunner(
        provider=provider,
        provider_label="test-provider",
        model_label="test-model",
        evaluator=caller_evaluator,
    )

    report = runner.run_trusted(descriptor).report

    assert report.provenance.evaluation.prompt_id == "airline_assistant"
    assert report.provenance.evaluation.golden_dataset_id == REGISTERED_DATASET_ID
    assert report.provenance.evaluation.evaluator_id == REGISTERED_EVALUATOR_ID
    assert "caller-substitution" not in provider.case_ids
    assert all(prompt.text == descriptor.prompt.prompt.text for prompt in provider.prompts[1:])
    assert caller_prompt.text not in report.model_dump_json()


def test_materialization_is_fresh_through_descriptor():
    descriptor = _descriptor()
    first = descriptor.materialize_cases()
    second = descriptor.materialize_cases()
    first[0].user_input = "changed"

    assert first is not second
    assert second[0].user_input != "changed"


def test_descriptor_has_no_public_serialization_or_sensitive_projection():
    descriptor = _descriptor()
    representation = repr(descriptor)

    with pytest.raises(TypeError):
        json.dumps(descriptor)
    for forbidden in (
        descriptor.prompt.prompt.text,
        descriptor.prompt.prompt.purpose,
        descriptor.dataset._canonical_source,
        "88d258e382112f5d024e21712bd3ece805925383797fc19183fa3282507691d8",
        "_capability",
        "factory",
        "object at 0x",
        "https://",
    ):
        assert forbidden not in representation


@pytest.mark.parametrize("public_value", [{}, "{}", object()])
def test_public_evidence_cannot_be_promoted_to_descriptor(public_value):
    with pytest.raises((TypeError, ValueError)):
        TrustedExecutionDescriptor.create(
            registered_prompt=public_value,
            registered_dataset=public_value,
            registered_evaluator=public_value,
            provider_settings=None,
            provider_required=False,
            provider_id="test-provider",
            model_id="test-model",
        )
