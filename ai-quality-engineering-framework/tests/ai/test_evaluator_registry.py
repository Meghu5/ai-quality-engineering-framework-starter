from __future__ import annotations

import json

import pytest

from ai_quality.evaluator_registry import (
    REGISTERED_EVALUATOR_ID,
    REGISTERED_EVALUATOR_VERSION,
    REGISTERED_POLICY_ID,
    REGISTERED_POLICY_VERSION,
    EvaluatorRegistry,
    RegisteredEvaluator,
    evaluator_provenance_for,
)
from ai_quality.evaluators import QualityGateEvaluator
from ai_quality.models import AirlineAssistantResponse, GoldenCase
from ai_quality.thresholds import AIQualityThresholds


def _resolve(**kwargs) -> RegisteredEvaluator:
    return EvaluatorRegistry().resolve(
        REGISTERED_EVALUATOR_ID,
        REGISTERED_EVALUATOR_VERSION,
        **kwargs,
    )


def _case() -> GoldenCase:
    return GoldenCase(
        id="evaluator-case",
        category="flight_search",
        user_input="Find a flight",
        expected_intent="flight_search",
    )


def _response() -> AirlineAssistantResponse:
    return AirlineAssistantResponse(
        intent="flight_search",
        response="I can help with flights.",
        prompt_version="v1",
    )


def test_resolve_returns_immutable_registered_evaluator():
    registered = _resolve()

    assert isinstance(registered, RegisteredEvaluator)
    assert registered.evaluator_id == REGISTERED_EVALUATOR_ID
    assert registered.evaluator_version == REGISTERED_EVALUATOR_VERSION
    assert registered.policy_id == REGISTERED_POLICY_ID
    assert registered.policy_version == REGISTERED_POLICY_VERSION
    with pytest.raises(AttributeError, match="immutable"):
        registered.policy_version = "2.0"
    with pytest.raises(TypeError, match="resolved by EvaluatorRegistry"):
        RegisteredEvaluator()


@pytest.mark.parametrize(
    ("evaluator_id", "version"),
    [
        (REGISTERED_EVALUATOR_ID, "2.0"),
        ("other-evaluator", REGISTERED_EVALUATOR_VERSION),
        (REGISTERED_EVALUATOR_ID.upper(), REGISTERED_EVALUATOR_VERSION),
    ],
)
def test_invalid_resolution_fails_deterministically(evaluator_id, version):
    with pytest.raises(KeyError, match="registration not found"):
        EvaluatorRegistry().resolve(evaluator_id, version)


def test_registry_factory_creates_builtin_with_governed_policy():
    registered = _resolve()
    expected = QualityGateEvaluator().evaluate([_case()], [_response()])

    assert isinstance(registered._evaluator, QualityGateEvaluator)
    assert registered.evaluate([_case()], [_response()]) == expected


def test_policy_change_requires_governed_version():
    changed = AIQualityThresholds(intent_accuracy=0.90)
    with pytest.raises(ValueError, match="requires a governed version"):
        _resolve(policy=changed)


@pytest.mark.parametrize(
    "policy",
    [
        object(),
        AIQualityThresholds(intent_accuracy=2.0),
        AIQualityThresholds(intent_accuracy=float("nan")),
    ],
)
def test_registered_policy_is_typed_and_bounded(policy):
    with pytest.raises((TypeError, ValueError)):
        _resolve(policy=policy)


def test_runtime_policy_mutation_fails_closed():
    registered = _resolve()
    registered._evaluator.thresholds = AIQualityThresholds(intent_accuracy=0.90)

    with pytest.raises(ValueError, match="policy was modified"):
        registered.evaluate([_case()], [_response()])


def test_matching_plain_evaluator_metadata_cannot_manufacture_authority():
    registered = _resolve()
    spoofed = QualityGateEvaluator()
    spoofed.evaluator_id = registered.evaluator_id
    spoofed.evaluator_version = registered.evaluator_version
    spoofed.policy_id = registered.policy_id
    spoofed.policy_version = registered.policy_version

    declaration = evaluator_provenance_for(spoofed)
    assert declaration.evaluator_id == "unregistered"
    assert declaration.evaluator_version == "0"
    assert declaration.policy_id == "unregistered"
    assert declaration.policy_version == "0"


def test_self_declared_and_arbitrary_metadata_are_ignored():
    class SpoofedEvaluator:
        evaluator_id = REGISTERED_EVALUATOR_ID
        evaluator_version = REGISTERED_EVALUATOR_VERSION
        policy_id = REGISTERED_POLICY_ID
        policy_version = REGISTERED_POLICY_VERSION
        metadata = {"secret": "private evaluator metadata"}

        def provenance_declaration(self):
            return evaluator_provenance_for(_resolve())

    declaration = evaluator_provenance_for(SpoofedEvaluator())
    serialized = declaration.model_dump_json()
    assert declaration.evaluator_id == "unregistered"
    assert "private evaluator metadata" not in serialized
    assert REGISTERED_EVALUATOR_ID not in serialized


def test_forged_capability_is_rejected():
    legitimate = _resolve()
    forged = object.__new__(RegisteredEvaluator)
    object.__setattr__(forged, "_declaration", legitimate._declaration)
    object.__setattr__(forged, "_evaluator", legitimate._evaluator)
    object.__setattr__(forged, "_policy", legitimate._policy)
    object.__setattr__(forged, "_enrollment_proof", object())

    with pytest.raises(ValueError, match="capability is invalid"):
        evaluator_provenance_for(forged)


def test_repr_and_provenance_exclude_runtime_authority_and_factory():
    registered = _resolve()
    declaration = evaluator_provenance_for(registered)
    serialized = json.dumps(declaration.model_dump(mode="json"), sort_keys=True)

    for forbidden in (
        "_capability",
        "_evaluator",
        "_policy",
        "factory",
        "object at 0x",
    ):
        assert forbidden not in serialized
        assert forbidden not in repr(registered)
