from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Callable
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field

from ai_quality.evaluators import QualityGateEvaluator
from ai_quality.models import AirlineAssistantResponse, GoldenCase, QualityReport
from ai_quality.thresholds import AIQualityThresholds, DEFAULT_AI_QUALITY_THRESHOLDS


REGISTERED_EVALUATOR_ID = "quality-gate-evaluator"
REGISTERED_EVALUATOR_VERSION = "1.0"
REGISTERED_POLICY_ID = "quality-gate-thresholds"
REGISTERED_POLICY_VERSION = "1.0"
UNREGISTERED_EVALUATOR_ID = "unregistered"
UNREGISTERED_EVALUATOR_VERSION = "0"
UNREGISTERED_POLICY_ID = "unregistered"
UNREGISTERED_POLICY_VERSION = "0"


class EvaluatorProvenanceDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evaluator_id: str = Field(pattern=r"[a-z][a-z0-9_.-]{0,63}")
    evaluator_version: str = Field(pattern=r"(?:v)?[0-9]+(?:\.[0-9]+){0,3}")
    policy_id: str = Field(pattern=r"[a-z][a-z0-9_.-]{0,63}")
    policy_version: str = Field(pattern=r"(?:v)?[0-9]+(?:\.[0-9]+){0,3}")
    policy_fingerprint: str = Field(pattern=r"evaluator-policy-[0-9a-f]{64}")


@dataclass(frozen=True)
class _EvaluatorRegistration:
    evaluator_id: str
    evaluator_version: str
    policy_id: str
    policy_version: str
    policy: AIQualityThresholds
    policy_projector: Callable[[AIQualityThresholds], AIQualityThresholds]
    factory: Callable[[AIQualityThresholds], QualityGateEvaluator]


class RegisteredEvaluator:
    __slots__ = (
        "_declaration",
        "_evaluator",
        "_policy",
        "_capability",
        "__weakref__",
    )

    def __new__(cls, *args, **kwargs):
        raise TypeError("registered evaluators must be resolved by EvaluatorRegistry")

    @property
    def evaluator_id(self) -> str:
        return self._declaration.evaluator_id

    @property
    def evaluator_version(self) -> str:
        return self._declaration.evaluator_version

    @property
    def policy_id(self) -> str:
        return self._declaration.policy_id

    @property
    def policy_version(self) -> str:
        return self._declaration.policy_version

    @property
    def policy_fingerprint(self) -> str:
        return self._declaration.policy_fingerprint

    def evaluate(
        self,
        cases: list[GoldenCase],
        responses: list[AirlineAssistantResponse],
    ) -> QualityReport:
        if not _is_issued_evaluator(self):
            raise ValueError("registered evaluator capability is invalid")
        if self._evaluator.thresholds != self._policy:
            raise ValueError("registered evaluator policy was modified")
        return self._evaluator.evaluate(cases, responses)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("registered evaluators are immutable")

    def __repr__(self) -> str:
        return (
            "RegisteredEvaluator("
            f"evaluator_id={self.evaluator_id!r}, "
            f"evaluator_version={self.evaluator_version!r}, "
            f"policy_id={self.policy_id!r}, "
            f"policy_version={self.policy_version!r})"
        )


def _project_quality_thresholds(
    thresholds: AIQualityThresholds,
) -> AIQualityThresholds:
    if not isinstance(thresholds, AIQualityThresholds):
        raise TypeError("registered evaluator policy must use AIQualityThresholds")
    values = asdict(thresholds)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        or value > 1.0
        for value in values.values()
    ):
        raise ValueError("registered evaluator thresholds must be finite values from 0 to 1")
    return AIQualityThresholds(**values)


def _quality_gate_factory(thresholds: AIQualityThresholds) -> QualityGateEvaluator:
    return QualityGateEvaluator(thresholds)


_QUALITY_GATE_REGISTRATION = _EvaluatorRegistration(
    evaluator_id=REGISTERED_EVALUATOR_ID,
    evaluator_version=REGISTERED_EVALUATOR_VERSION,
    policy_id=REGISTERED_POLICY_ID,
    policy_version=REGISTERED_POLICY_VERSION,
    policy=DEFAULT_AI_QUALITY_THRESHOLDS,
    policy_projector=_project_quality_thresholds,
    factory=_quality_gate_factory,
)
_REGISTERED_EVALUATORS = MappingProxyType(
    {
        (REGISTERED_EVALUATOR_ID, REGISTERED_EVALUATOR_VERSION): (
            _QUALITY_GATE_REGISTRATION
        )
    }
)
_ISSUED_EVALUATORS: WeakKeyDictionary[RegisteredEvaluator, object] = (
    WeakKeyDictionary()
)


class EvaluatorRegistry:
    def resolve(
        self,
        evaluator_id: str,
        evaluator_version: str,
        *,
        policy: AIQualityThresholds | None = None,
    ) -> RegisteredEvaluator:
        registration = _REGISTERED_EVALUATORS.get(
            (evaluator_id, evaluator_version)
        )
        if registration is None:
            raise KeyError("evaluator registration not found")
        projected = registration.policy_projector(policy or registration.policy)
        governed = registration.policy_projector(registration.policy)
        if projected != governed:
            raise ValueError(
                "registered evaluator policy change requires a governed version"
            )
        evaluator = registration.factory(projected)
        declaration = EvaluatorProvenanceDeclaration(
            evaluator_id=registration.evaluator_id,
            evaluator_version=registration.evaluator_version,
            policy_id=registration.policy_id,
            policy_version=registration.policy_version,
            policy_fingerprint=_policy_fingerprint(projected),
        )
        return _issue_registered_evaluator(evaluator, declaration, projected)


def evaluator_provenance_for(evaluator: object) -> EvaluatorProvenanceDeclaration:
    if isinstance(evaluator, RegisteredEvaluator):
        if not _is_issued_evaluator(evaluator):
            raise ValueError("registered evaluator capability is invalid")
        return evaluator._declaration
    return EvaluatorProvenanceDeclaration(
        evaluator_id=UNREGISTERED_EVALUATOR_ID,
        evaluator_version=UNREGISTERED_EVALUATOR_VERSION,
        policy_id=UNREGISTERED_POLICY_ID,
        policy_version=UNREGISTERED_POLICY_VERSION,
        policy_fingerprint=_policy_fingerprint({"status": "unregistered"}),
    )


def _issue_registered_evaluator(
    evaluator: QualityGateEvaluator,
    declaration: EvaluatorProvenanceDeclaration,
    policy: AIQualityThresholds,
) -> RegisteredEvaluator:
    handle = object.__new__(RegisteredEvaluator)
    object.__setattr__(handle, "_evaluator", evaluator)
    object.__setattr__(handle, "_declaration", declaration)
    object.__setattr__(handle, "_policy", policy)
    capability = object()
    object.__setattr__(handle, "_capability", capability)
    _ISSUED_EVALUATORS[handle] = capability
    return handle


def _is_issued_evaluator(evaluator: RegisteredEvaluator) -> bool:
    return _ISSUED_EVALUATORS.get(evaluator) is evaluator._capability


def _policy_fingerprint(value: object) -> str:
    if isinstance(value, AIQualityThresholds):
        value = asdict(value)
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return f"evaluator-policy-{digest}"
