from __future__ import annotations

import json

import pytest

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
from ai_quality.provider_conformance import ProviderConformanceRunner
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

    def health_check(self) -> bool:
        return True

    def generate(self, user_input, *, prompt, context=None):
        return "unused"

    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        self.prompts.append(prompt)
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
    with pytest.raises(ValueError, match="requires an enrolled"):
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
        with pytest.raises(ValueError, match="requires an enrolled"):
            _descriptor(**override)


def test_forged_registered_handles_are_rejected():
    prompt, dataset, evaluator = _handles()
    forged_prompt = object.__new__(RegisteredPrompt)
    object.__setattr__(forged_prompt, "_prompt", prompt.prompt)
    object.__setattr__(forged_prompt, "_capability", object())
    forged_dataset = object.__new__(RegisteredDatasetBundle)
    object.__setattr__(forged_dataset, "_identity", dataset._identity)
    object.__setattr__(forged_dataset, "_canonical_source", dataset._canonical_source)
    object.__setattr__(forged_dataset, "_capability", object())
    forged_evaluator = object.__new__(RegisteredEvaluator)
    object.__setattr__(forged_evaluator, "_declaration", evaluator._declaration)
    object.__setattr__(forged_evaluator, "_evaluator", evaluator._evaluator)
    object.__setattr__(forged_evaluator, "_policy", evaluator._policy)
    object.__setattr__(forged_evaluator, "_capability", object())

    for override in (
        {"registered_prompt": forged_prompt},
        {"registered_dataset": forged_dataset},
        {"registered_evaluator": forged_evaluator},
    ):
        with pytest.raises(ValueError, match="requires an enrolled"):
            _descriptor(**override)


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
    assert all(prompt is descriptor.prompt.prompt for prompt in provider.prompts[1:])
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
