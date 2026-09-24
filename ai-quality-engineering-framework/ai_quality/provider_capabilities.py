from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from observability.models import FailureCategory


AIRLINE_RESPONSE_SCHEMA_ID = "airline-assistant-response"
AIRLINE_RESPONSE_SCHEMA_VERSION = "1.0"
AIRLINE_RESPONSE_VALIDATOR_ID = "airline-assistant-response-validator"
AIRLINE_RESPONSE_VALIDATOR_VERSION = "1.0"

CapabilityStatus = Literal["supported", "unsupported", "unknown", "schema_mismatch"]
DeclarationState = Literal["compatible", "unknown", "incompatible", "invalid", "absent"]
EmpiricalState = Literal["verified", "failed", "not_run"]
VerificationBasis = Literal["declaration", "empirical", "both"]
CapabilityReasonCode = Literal[
    "provider_capability_unsupported",
    "provider_capability_invalid",
    "provider_schema_mismatch",
]


class ProviderCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    structured_output: bool = Field(strict=True)
    response_schema_id: Literal["airline-assistant-response"]
    response_schema_versions: tuple[Literal["1.0"], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_safe_schema_values(self) -> "ProviderCapabilities":
        if len(set(self.response_schema_versions)) != len(
            self.response_schema_versions
        ):
            raise ValueError("response_schema_versions must be unique")
        return self


class ProviderCapabilityRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    structured_output_required: bool = Field(default=True, strict=True)
    response_schema_id: Literal["airline-assistant-response"] = (
        AIRLINE_RESPONSE_SCHEMA_ID
    )
    response_schema_version: Literal["1.0"] = AIRLINE_RESPONSE_SCHEMA_VERSION


class ProviderCapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: CapabilityStatus
    structured_output_supported: bool | None = Field(default=None, strict=True)
    response_schema_id: Literal["airline-assistant-response"] = (
        AIRLINE_RESPONSE_SCHEMA_ID
    )
    response_schema_version: Literal["1.0"] = AIRLINE_RESPONSE_SCHEMA_VERSION
    failure_category: FailureCategory | None = None
    reason_code: CapabilityReasonCode | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "ProviderCapabilityEvidence":
        expected = {
            "supported": (True, None, None),
            "unsupported": (
                False,
                FailureCategory.CONTRACT,
                "provider_capability_unsupported",
            ),
            "schema_mismatch": (
                True,
                FailureCategory.CONTRACT,
                "provider_schema_mismatch",
            ),
        }
        if self.status in expected:
            if (
                self.structured_output_supported,
                self.failure_category,
                self.reason_code,
            ) != expected[self.status]:
                raise ValueError("capability evidence is inconsistent with status")
        elif self.status == "unknown":
            valid_unknown = (
                self.structured_output_supported is None
                and (
                    (self.failure_category is None and self.reason_code is None)
                    or (
                        self.failure_category == FailureCategory.CONTRACT
                        and self.reason_code == "provider_capability_invalid"
                    )
                )
            )
            if not valid_unknown:
                raise ValueError("unknown capability evidence is inconsistent")
        return self


class SchemaCompatibilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    declaration: DeclarationState
    empirical: EmpiricalState
    verification_basis: VerificationBasis | None = None
    expected_schema_id: Literal["airline-assistant-response"] = (
        AIRLINE_RESPONSE_SCHEMA_ID
    )
    expected_schema_version: Literal["1.0"] = AIRLINE_RESPONSE_SCHEMA_VERSION
    validator_id: Literal["airline-assistant-response-validator"] = (
        AIRLINE_RESPONSE_VALIDATOR_ID
    )
    validator_version: Literal["1.0"] = AIRLINE_RESPONSE_VALIDATOR_VERSION
    golden_contract_valid: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_compatibility(self) -> "SchemaCompatibilityEvidence":
        if self.declaration in {"incompatible", "invalid"}:
            if self.empirical != "not_run" or self.verification_basis is not None:
                raise ValueError(
                    "incompatible or invalid declarations cannot be empirically verified"
                )
        elif self.empirical == "verified":
            expected_basis = (
                "both" if self.declaration == "compatible" else "empirical"
            )
            if self.verification_basis != expected_basis:
                raise ValueError("verification basis is inconsistent with evidence")
        elif self.empirical == "failed":
            if self.verification_basis is not None:
                raise ValueError("failed empirical verification has no compatibility basis")
        elif self.empirical == "not_run":
            expected_basis = (
                "declaration" if self.declaration == "compatible" else None
            )
            if self.verification_basis != expected_basis:
                raise ValueError("verification basis is inconsistent with evidence")
        if self.golden_contract_valid and self.empirical != "verified":
            raise ValueError(
                "golden contract validity requires empirical verification"
            )
        return self


@runtime_checkable
class CapabilityProvider(Protocol):
    def get_capabilities(self) -> ProviderCapabilities | None: ...


def assess_provider_capabilities(
    provider: object,
    requirement: ProviderCapabilityRequirement,
) -> ProviderCapabilityEvidence:
    if not isinstance(provider, CapabilityProvider):
        return ProviderCapabilityEvidence(status="unknown")
    try:
        raw_declaration = provider.get_capabilities()
        if raw_declaration is None:
            return ProviderCapabilityEvidence(status="unknown")
        declaration = ProviderCapabilities.model_validate(raw_declaration, strict=True)
    except Exception:
        if _is_well_formed_schema_mismatch(locals().get("raw_declaration")):
            return ProviderCapabilityEvidence(
                status="schema_mismatch",
                structured_output_supported=True,
                failure_category=FailureCategory.CONTRACT,
                reason_code="provider_schema_mismatch",
            )
        return ProviderCapabilityEvidence(
            status="unknown",
            failure_category=FailureCategory.CONTRACT,
            reason_code="provider_capability_invalid",
        )
    if requirement.structured_output_required and not declaration.structured_output:
        return ProviderCapabilityEvidence(
            status="unsupported",
            structured_output_supported=False,
            failure_category=FailureCategory.CONTRACT,
            reason_code="provider_capability_unsupported",
        )
    if (
        declaration.response_schema_id != requirement.response_schema_id
        or requirement.response_schema_version
        not in declaration.response_schema_versions
    ):
        return ProviderCapabilityEvidence(
            status="schema_mismatch",
            structured_output_supported=True,
            failure_category=FailureCategory.CONTRACT,
            reason_code="provider_schema_mismatch",
        )
    return ProviderCapabilityEvidence(
        status="supported",
        structured_output_supported=True,
    )


def _is_well_formed_schema_mismatch(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "structured_output",
        "response_schema_id",
        "response_schema_versions",
    }:
        return False
    versions = value["response_schema_versions"]
    return (
        value["structured_output"] is True
        and isinstance(value["response_schema_id"], str)
        and isinstance(versions, tuple)
        and bool(versions)
        and all(isinstance(version, str) for version in versions)
        and (
            value["response_schema_id"] != AIRLINE_RESPONSE_SCHEMA_ID
            or AIRLINE_RESPONSE_SCHEMA_VERSION not in versions
        )
    )


def supported_capability_evidence() -> ProviderCapabilityEvidence:
    return ProviderCapabilityEvidence(
        status="supported",
        structured_output_supported=True,
    )


def schema_compatibility_for_declaration(
    provider: object | None,
    capability: ProviderCapabilityEvidence | None,
) -> SchemaCompatibilityEvidence:
    if capability is None:
        declaration: DeclarationState = (
            "unknown" if isinstance(provider, CapabilityProvider) else "absent"
        )
    elif capability.status == "supported":
        declaration = "compatible"
    elif capability.status in {"unsupported", "schema_mismatch"}:
        declaration = "incompatible"
    elif capability.reason_code == "provider_capability_invalid":
        declaration = "invalid"
    else:
        declaration = (
            "unknown" if isinstance(provider, CapabilityProvider) else "absent"
        )
    return SchemaCompatibilityEvidence(
        declaration=declaration,
        empirical="not_run",
        verification_basis=(
            "declaration" if declaration == "compatible" else None
        ),
    )


def with_empirical_compatibility(
    evidence: SchemaCompatibilityEvidence,
    empirical: Literal["verified", "failed"],
    *,
    golden_contract_valid: bool = False,
) -> SchemaCompatibilityEvidence:
    basis: VerificationBasis | None = None
    if empirical == "verified":
        basis = "both" if evidence.declaration == "compatible" else "empirical"
    return SchemaCompatibilityEvidence(
        declaration=evidence.declaration,
        empirical=empirical,
        verification_basis=basis,
        golden_contract_valid=golden_contract_valid,
    )
