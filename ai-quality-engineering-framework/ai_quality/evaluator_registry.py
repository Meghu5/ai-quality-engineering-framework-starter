from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Callable

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
        "_enrollment_proof",
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
class EvaluatorRegistry:
    def resolve(
        self,
        evaluator_id: str,
        evaluator_version: str,
        *,
        policy: AIQualityThresholds | None = None,
    ) -> RegisteredEvaluator:
        raise RuntimeError("evaluator enrollment entry point is not bound")


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


def _bind_evaluator_enrollment():
    registrations = MappingProxyType(
        {
            key: (
                registration.evaluator_id,
                registration.evaluator_version,
                registration.policy_id,
                registration.policy_version,
                tuple(asdict(registration.policy).items()),
            )
            for key, registration in _REGISTERED_EVALUATORS.items()
        }
    )
    policy_fingerprint = _policy_fingerprint
    evaluator_constructor = QualityGateEvaluator
    policy_constructor = AIQualityThresholds

    class EnrollmentProof:
        __slots__ = ("_validator",)

        def __new__(cls, *args, **kwargs):
            raise TypeError("evaluator enrollment proofs are framework-owned")

        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError("evaluator enrollment proofs are immutable")

        def validates(self, candidate: object) -> bool:
            return self._validator(candidate)

    def build_canonical(
        evaluator_id: str,
        evaluator_version: str,
        *,
        policy: AIQualityThresholds | None = None,
    ) -> RegisteredEvaluator:
        registration = registrations.get((evaluator_id, evaluator_version))
        if registration is None:
            raise KeyError("evaluator registration not found")
        (
            governed_evaluator_id,
            governed_evaluator_version,
            policy_id,
            policy_version,
            governed_policy_items,
        ) = registration
        governed_policy = policy_constructor(**dict(governed_policy_items))
        if policy is not None and policy != governed_policy:
            raise ValueError(
                "registered evaluator policy change requires a governed version"
            )
        evaluator = evaluator_constructor(governed_policy)
        declaration = EvaluatorProvenanceDeclaration(
            evaluator_id=governed_evaluator_id,
            evaluator_version=governed_evaluator_version,
            policy_id=policy_id,
            policy_version=policy_version,
            policy_fingerprint=policy_fingerprint(governed_policy),
        )
        handle = object.__new__(RegisteredEvaluator)
        object.__setattr__(handle, "_evaluator", evaluator)
        object.__setattr__(handle, "_declaration", declaration)
        object.__setattr__(handle, "_policy", governed_policy)
        proof = object.__new__(EnrollmentProof)

        def validates(candidate: object) -> bool:
            return (
                candidate is handle
                and candidate._evaluator is evaluator
                and candidate._declaration is declaration
                and candidate._policy is governed_policy
            )

        object.__setattr__(proof, "_validator", validates)
        object.__setattr__(handle, "_enrollment_proof", proof)
        return handle

    def resolve(
        self: EvaluatorRegistry,
        evaluator_id: str,
        evaluator_version: str,
        *,
        policy: AIQualityThresholds | None = None,
    ) -> RegisteredEvaluator:
        return build_canonical(
            evaluator_id,
            evaluator_version,
            policy=policy,
        )

    def canonicalize(evaluator: RegisteredEvaluator) -> RegisteredEvaluator:
        if not isinstance(evaluator, RegisteredEvaluator):
            raise ValueError("trusted descriptor requires an evaluator identity request")
        try:
            declaration = evaluator._declaration
            if not isinstance(declaration, EvaluatorProvenanceDeclaration):
                raise TypeError
            evaluator_id = declaration.evaluator_id
            evaluator_version = declaration.evaluator_version
        except (AttributeError, TypeError):
            raise ValueError(
                "trusted descriptor requires a valid evaluator identity request"
            ) from None
        return build_canonical(evaluator_id, evaluator_version)

    def is_issued(evaluator: RegisteredEvaluator) -> bool:
        try:
            proof = evaluator._enrollment_proof
            return isinstance(proof, EnrollmentProof) and proof.validates(evaluator)
        except (AttributeError, TypeError):
            return False

    return resolve, is_issued, canonicalize


EvaluatorRegistry.resolve, _is_issued_evaluator, _canonicalize_registered_evaluator = (
    _bind_evaluator_enrollment()
)
del _bind_evaluator_enrollment
