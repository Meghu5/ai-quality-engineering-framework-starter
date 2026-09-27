from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_quality.dataset import GoldenDatasetIdentity
from ai_quality.evaluator_registry import (
    EvaluatorProvenanceDeclaration,
    evaluator_provenance_for,
)
from ai_quality.provider_capabilities import (
    AIRLINE_RESPONSE_SCHEMA_ID, AIRLINE_RESPONSE_SCHEMA_VERSION,
    AIRLINE_RESPONSE_VALIDATOR_ID, AIRLINE_RESPONSE_VALIDATOR_VERSION,
    ProviderCapabilityRequirement,
)
from ai_quality.provider_config import RealLLMProviderSettings
from ai_quality.execution_descriptor import (
    SafeProviderExecutionSettings,
    TrustedExecutionDescriptor,
)
from ai_quality.privacy import safe_case_id

PROVENANCE_CONTRACT_VERSION = "2.0"
PROVENANCE_PRODUCER_ID = "provider-conformance"
PROVENANCE_IMPLEMENTATION_VERSION = "phase12-step9"
RETRY_POLICY_ID = "bounded-http-retry"
RETRY_POLICY_VERSION = "1.0"
CAPABILITY_CONTRACT_ID = "provider-capability-contract"
CAPABILITY_CONTRACT_VERSION = "1.0"

_SAFE_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_SAFE_VERSION = re.compile(r"(?:v)?[0-9]+(?:\.[0-9]+){0,3}")
_OPAQUE_IDENTIFIER = re.compile(r"(?:case|opaque)-[0-9a-f]{16}|(?:provider|model)-[a-z0-9_.-]{1,64}")
_SAFE_CASE_ID = re.compile(r"case-[0-9a-f]{16}")
_FINGERPRINT = re.compile(r"[a-z][a-z0-9-]*-[0-9a-f]{64}")


class GoldenCaseReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    case_id: str = Field(pattern=_SAFE_CASE_ID.pattern)
    revision: str = Field(pattern=_SAFE_VERSION.pattern)


class ProducerProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    component_id: Literal["provider-conformance"] = PROVENANCE_PRODUCER_ID
    implementation_version: Literal["phase12-step9"] = PROVENANCE_IMPLEMENTATION_VERSION


class EvaluationInputProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    prompt_id: Literal["airline_assistant", "unregistered"]
    prompt_version: Literal["v1", "v2", "0"]
    golden_dataset_id: Literal["airline-quality-golden-cases", "unregistered"]
    golden_dataset_version: str = Field(pattern=_SAFE_VERSION.pattern)
    golden_case_set_id: str = Field(pattern=_FINGERPRINT.pattern)
    case_references: tuple[GoldenCaseReference, ...]
    case_count: int = Field(ge=0, strict=True)
    evaluator_id: str = Field(pattern=_SAFE_NAME.pattern)
    evaluator_version: str = Field(pattern=_SAFE_VERSION.pattern)
    evaluator_policy_id: str = Field(pattern=_SAFE_NAME.pattern)
    evaluator_policy_version: str = Field(pattern=_SAFE_VERSION.pattern)
    evaluator_policy_fingerprint: str = Field(pattern=_FINGERPRINT.pattern)
    evaluation_input_id: str = Field(pattern=_FINGERPRINT.pattern)

    @model_validator(mode="after")
    def validate_identity(self) -> "EvaluationInputProvenance":
        if self.case_count != len(self.case_references):
            raise ValueError("case references must match case count")
        if (self.prompt_id == "unregistered") != (self.prompt_version == "0"):
            raise ValueError("prompt identity and version are inconsistent")
        golden = {"dataset_id": self.golden_dataset_id, "dataset_version": self.golden_dataset_version,
                  "cases": [item.model_dump(mode="json") for item in self.case_references]}
        if self.golden_case_set_id != _fingerprint("golden-set", golden):
            raise ValueError("golden_case_set_id does not match golden evidence")
        if self.evaluation_input_id != _fingerprint("evaluation-input", self.model_dump(mode="json", exclude={"evaluation_input_id"})):
            raise ValueError("evaluation_input_id does not match evaluation evidence")
        return self


class ProviderIdentityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    provider_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=80)
    provider_identity_id: str = Field(pattern=_FINGERPRINT.pattern)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "ProviderIdentityProvenance":
        for value in (self.provider_id, self.model_id):
            if not _SAFE_NAME.fullmatch(value) and not _OPAQUE_IDENTIFIER.fullmatch(value):
                raise ValueError("provider identity must already be sanitized")
        if self.provider_identity_id != _fingerprint("provider-identity", self.model_dump(mode="json", exclude={"provider_identity_id"})):
            raise ValueError("provider_identity_id does not match provider evidence")
        return self


class ContractIdentityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    response_schema_id: Literal["airline-assistant-response"]
    response_schema_version: Literal["1.0"]
    validator_id: Literal["airline-assistant-response-validator"]
    validator_version: Literal["1.0"]
    capability_contract_id: Literal["provider-capability-contract"]
    capability_contract_version: Literal["1.0"]
    structured_output_required: bool = Field(strict=True)
    contract_identity_id: str = Field(pattern=_FINGERPRINT.pattern)

    @model_validator(mode="after")
    def validate_identity(self) -> "ContractIdentityProvenance":
        if self.contract_identity_id != _fingerprint("contract-identity", self.model_dump(mode="json", exclude={"contract_identity_id"})):
            raise ValueError("contract_identity_id does not match contract evidence")
        return self


class ExecutionPolicyProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    timeout_seconds: float | None = Field(default=None, gt=0, strict=True)
    max_attempts: int | None = Field(default=None, ge=1, strict=True)
    max_retry_delay_seconds: float | None = Field(default=None, ge=0, strict=True)
    provider_required: bool = Field(strict=True)
    retry_policy_id: Literal["bounded-http-retry", "provider-managed"]
    retry_policy_version: Literal["1.0"] = RETRY_POLICY_VERSION
    execution_policy_id: str = Field(pattern=_FINGERPRINT.pattern)

    @model_validator(mode="after")
    def validate_policy(self) -> "ExecutionPolicyProvenance":
        if any(value is not None and not math.isfinite(value) for value in (self.timeout_seconds, self.max_retry_delay_seconds)):
            raise ValueError("execution policy values must be finite")
        real = (self.timeout_seconds, self.max_attempts, self.max_retry_delay_seconds)
        if self.retry_policy_id == RETRY_POLICY_ID and any(value is None for value in real):
            raise ValueError("bounded retry policy requires complete safe settings")
        if self.retry_policy_id == "provider-managed" and any(value is not None for value in real):
            raise ValueError("provider-managed policy cannot contain HTTP retry settings")
        if self.execution_policy_id != _fingerprint("execution-policy", self.model_dump(mode="json", exclude={"execution_policy_id"})):
            raise ValueError("execution_policy_id does not match execution policy")
        return self


class ProviderConformanceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    contract_version: Literal["2.0"] = PROVENANCE_CONTRACT_VERSION
    producer: ProducerProvenance
    evaluation: EvaluationInputProvenance
    provider: ProviderIdentityProvenance
    contract: ContractIdentityProvenance
    execution_policy: ExecutionPolicyProvenance
    execution_evidence_id: str = Field(pattern=_FINGERPRINT.pattern)
    provenance_id: str = Field(pattern=_FINGERPRINT.pattern)

    @model_validator(mode="after")
    def validate_provenance_id(self) -> "ProviderConformanceProvenance":
        if self.execution_evidence_id != _evidence_id(self.evaluation, self.provider, self.contract, self.execution_policy):
            raise ValueError("execution_evidence_id does not match evidence")
        if self.provenance_id != _fingerprint("provenance", self.model_dump(mode="json", exclude={"provenance_id"})):
            raise ValueError("provenance_id does not match provenance evidence")
        return self


@dataclass(frozen=True)
class _ConformanceExecutionDescriptor:
    prompt_id: str
    prompt_version: str
    dataset: GoldenDatasetIdentity
    safe_case_ids: tuple[str, ...]
    evaluator: EvaluatorProvenanceDeclaration
    provider_id: str
    model_id: str
    settings: RealLLMProviderSettings | SafeProviderExecutionSettings | None
    provider_required: bool
    capability_requirement: ProviderCapabilityRequirement
    trusted_source: TrustedExecutionDescriptor | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        registered = (
            self.prompt_id != "unregistered"
            or self.dataset.dataset_id != "unregistered"
            or self.evaluator.evaluator_id != "unregistered"
        )
        if not registered:
            if self.trusted_source is not None:
                raise ValueError("unregistered provenance cannot claim a trusted source")
            return
        source = self.trusted_source
        if source is None:
            raise ValueError(
                "registered provenance requires a trusted execution descriptor"
            )
        expected = (
            source.prompt.prompt.name,
            source.prompt.prompt.version,
            source.dataset._identity,
            tuple(safe_case_id(item).value for item in source.dataset.case_ids),
            evaluator_provenance_for(source.evaluator),
            source.provider_id,
            source.model_id,
            source.provider_settings,
            source.provider_settings.provider_required,
            source.capability_requirement,
        )
        actual = (
            self.prompt_id,
            self.prompt_version,
            self.dataset,
            self.safe_case_ids,
            self.evaluator,
            self.provider_id,
            self.model_id,
            self.settings,
            self.provider_required,
            self.capability_requirement,
        )
        if actual != expected:
            raise ValueError("registered provenance must match its trusted source")


def _evaluator_declaration(evaluator: object) -> EvaluatorProvenanceDeclaration:
    return evaluator_provenance_for(evaluator)


def _build_provider_conformance_provenance(descriptor: _ConformanceExecutionDescriptor) -> ProviderConformanceProvenance:
    if len(descriptor.safe_case_ids) != len(descriptor.dataset.cases):
        raise ValueError("dataset identity must match runtime cases")
    references = tuple(GoldenCaseReference(case_id=safe_id, revision=source.revision)
                       for safe_id, source in zip(descriptor.safe_case_ids, descriptor.dataset.cases))
    golden = {"dataset_id": descriptor.dataset.dataset_id, "dataset_version": descriptor.dataset.dataset_version,
              "cases": [item.model_dump(mode="json") for item in references]}
    evaluation_values = {
        "prompt_id": descriptor.prompt_id, "prompt_version": descriptor.prompt_version,
        "golden_dataset_id": descriptor.dataset.dataset_id,
        "golden_dataset_version": descriptor.dataset.dataset_version,
        "golden_case_set_id": _fingerprint("golden-set", golden), "case_references": references,
        "case_count": len(references), "evaluator_id": descriptor.evaluator.evaluator_id,
        "evaluator_version": descriptor.evaluator.evaluator_version,
        "evaluator_policy_id": descriptor.evaluator.policy_id,
        "evaluator_policy_version": descriptor.evaluator.policy_version,
        "evaluator_policy_fingerprint": descriptor.evaluator.policy_fingerprint,
    }
    evaluation = EvaluationInputProvenance(**evaluation_values,
        evaluation_input_id=_fingerprint("evaluation-input", evaluation_values))
    provider_values = {"provider_id": descriptor.provider_id, "model_id": descriptor.model_id}
    provider = ProviderIdentityProvenance(**provider_values,
        provider_identity_id=_fingerprint("provider-identity", provider_values))
    requirement = descriptor.capability_requirement
    contract_values = {
        "response_schema_id": AIRLINE_RESPONSE_SCHEMA_ID, "response_schema_version": AIRLINE_RESPONSE_SCHEMA_VERSION,
        "validator_id": AIRLINE_RESPONSE_VALIDATOR_ID, "validator_version": AIRLINE_RESPONSE_VALIDATOR_VERSION,
        "capability_contract_id": CAPABILITY_CONTRACT_ID, "capability_contract_version": CAPABILITY_CONTRACT_VERSION,
        "structured_output_required": requirement.structured_output_required,
    }
    contract = ContractIdentityProvenance(**contract_values,
        contract_identity_id=_fingerprint("contract-identity", contract_values))
    execution = _execution_policy(descriptor.settings, descriptor.provider_required)
    producer = ProducerProvenance()
    evidence_id = _evidence_id(evaluation, provider, contract, execution)
    values = {"contract_version": PROVENANCE_CONTRACT_VERSION, "producer": producer, "evaluation": evaluation,
              "provider": provider, "contract": contract, "execution_policy": execution,
              "execution_evidence_id": evidence_id}
    return ProviderConformanceProvenance(**values,
        provenance_id=_fingerprint("provenance", values))


def _build_trusted_provider_conformance_provenance(
    descriptor: TrustedExecutionDescriptor,
) -> ProviderConformanceProvenance:
    projection = _ConformanceExecutionDescriptor(
        prompt_id=descriptor.prompt.prompt.name,
        prompt_version=descriptor.prompt.prompt.version,
        dataset=descriptor.dataset._identity,
        safe_case_ids=tuple(
            safe_case_id(case_id).value for case_id in descriptor.dataset.case_ids
        ),
        evaluator=evaluator_provenance_for(descriptor.evaluator),
        provider_id=descriptor.provider_id,
        model_id=descriptor.model_id,
        settings=descriptor.provider_settings,
        provider_required=descriptor.provider_settings.provider_required,
        capability_requirement=descriptor.capability_requirement,
        trusted_source=descriptor,
    )
    return _build_provider_conformance_provenance(projection)


def _execution_policy(settings, provider_required) -> ExecutionPolicyProvenance:
    values = {
        "timeout_seconds": (
            float(settings.timeout_seconds)
            if settings and settings.timeout_seconds is not None
            else None
        ),
        "max_attempts": settings.max_attempts if settings else None,
        "max_retry_delay_seconds": (
            float(settings.max_retry_delay_seconds)
            if settings and settings.max_retry_delay_seconds is not None
            else None
        ),
        "provider_required": provider_required,
        "retry_policy_id": (
            settings.retry_policy_id
            if isinstance(settings, SafeProviderExecutionSettings)
            else RETRY_POLICY_ID
            if settings
            else "provider-managed"
        ),
        "retry_policy_version": RETRY_POLICY_VERSION,
    }
    return ExecutionPolicyProvenance(**values, execution_policy_id=_fingerprint("execution-policy", values))


def _evidence_id(evaluation, provider, contract, execution) -> str:
    return _fingerprint("execution-evidence", {
        "evaluation_input_id": evaluation.evaluation_input_id,
        "provider_identity_id": provider.provider_identity_id,
        "contract_identity_id": contract.contract_identity_id,
        "execution_policy_id": execution.execution_policy_id,
    })


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint(prefix: str, value: object) -> str:
    canonical = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return f"{prefix}-{hashlib.sha256(canonical.encode('ascii')).hexdigest()}"
