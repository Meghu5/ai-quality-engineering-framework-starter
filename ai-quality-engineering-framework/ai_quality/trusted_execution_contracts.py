from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, TypeVar

import rfc8785
from pydantic import (
    BaseModel,
    AfterValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)


REQUEST_MAX_BYTES = 4096
MANIFEST_MAX_BYTES = 1_000_000
RESULT_MAX_BYTES = 2_000_000
MANIFEST_SEQUENCE_MAX = 9_007_199_254_740_991
POLICY_STRING_MAX_LENGTH = 1024
REPORT_DETAIL_MAX_DEPTH = 8
REPORT_DETAIL_MAX_ITEMS = 512
REPORT_DETAIL_MAX_STRING_LENGTH = 4096
REPORT_DETAIL_MAX_BYTES = 65_536

Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9-]{0,63}$",
    ),
]
Version = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=32,
        pattern=r"^(?:v)?[0-9]+(?:\.[0-9]+){0,3}$",
    ),
]
OpaqueId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=5,
        max_length=96,
        pattern=r"^[a-z][a-z0-9-]*-[A-Za-z0-9_-]+$",
    ),
]
Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$"),
]
def _validate_rfc3339(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be a valid RFC 3339 instant") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value


def _timestamp_instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=20,
        max_length=38,
        pattern=(
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:0[0-9]|1[0-4]):[0-5][0-9])$"
        ),
    ),
    AfterValidator(_validate_rfc3339),
]
Base64Url = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=16,
        max_length=1024,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
PolicyString = Annotated[
    str,
    StringConstraints(strict=True, max_length=POLICY_STRING_MAX_LENGTH),
]

_Contract = TypeVar("_Contract", bound="ContractModel")


def canonical_bytes(value: Any) -> bytes:
    """Return RFC 8785 bytes; unsupported and non-finite values are rejected."""
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise ValueError("value is not RFC 8785 canonicalizable") from exc


def canonical_digest(domain: bytes, value: Any) -> str:
    if not domain or not domain.endswith(b"\0"):
        raise ValueError("digest domain must be non-empty and null terminated")
    digest = hashlib.sha256(domain + canonical_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def json_digest(value: Any) -> str:
    return canonical_digest(b"AQEF-JSON-v1\0", value)


def _load_unique_json(raw: str | bytes | bytearray, *, max_bytes: int) -> Any:
    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if len(encoded) > max_bytes:
        raise ValueError("serialized contract exceeds size limit")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    try:
        return json.loads(
            encoded,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("malformed JSON contract") from exc


def _validate_report_details(value: Any) -> None:
    items = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if isinstance(current, str):
            if len(current) > REPORT_DETAIL_MAX_STRING_LENGTH:
                raise ValueError("report detail string exceeds length limit")
            continue
        if current is None or isinstance(current, (bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError("non-finite numbers are not allowed")
            continue
        if callable(current) or isinstance(current, (type, BaseModel)):
            raise ValueError("executable Python objects are not contract data")
        if not isinstance(current, (dict, list, tuple)):
            raise ValueError("report details must contain JSON values only")
        if depth >= REPORT_DETAIL_MAX_DEPTH:
            raise ValueError("report detail nesting exceeds depth limit")
        if isinstance(current, dict):
            entries = current.items()
            items += len(current)
            for key, item in entries:
                if not isinstance(key, str) or len(key) > 64:
                    raise ValueError("report detail keys must be bounded strings")
                stack.append((item, depth + 1))
        else:
            items += len(current)
            stack.extend((item, depth + 1) for item in current)
        if items > REPORT_DETAIL_MAX_ITEMS:
            raise ValueError("report detail item count exceeds limit")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    json_max_bytes: ClassVar[int] = MANIFEST_MAX_BYTES

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.model_dump(mode="json"))

    @classmethod
    def parse_raw(cls, *args: Any, **kwargs: Any) -> "ContractModel":
        raise TypeError(
            "legacy raw parsing is unavailable; use model_validate_json()"
        )

    @classmethod
    def parse_file(cls, *args: Any, **kwargs: Any) -> "ContractModel":
        raise TypeError(
            "legacy file parsing is unavailable; use model_validate_json()"
        )

    @classmethod
    def model_validate_json(
        cls: type[_Contract],
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> _Contract:
        _load_unique_json(json_data, max_bytes=cls.json_max_bytes)
        return super().model_validate_json(
            json_data,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def from_json_strict(
        cls: type[_Contract],
        raw: str | bytes,
        *,
        max_bytes: int,
    ) -> _Contract:
        _load_unique_json(raw, max_bytes=max_bytes)
        encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
        return cls.model_validate_json(encoded)


class TrustedExecutionRequest(ContractModel):
    json_max_bytes: ClassVar[int] = REQUEST_MAX_BYTES
    schema_version: Literal["1.0"]
    request_id: OpaqueId
    verifier_nonce: Base64Url
    prompt_id: Identifier
    prompt_version: Version
    dataset_id: Identifier
    dataset_version: Version
    evaluator_id: Identifier
    evaluator_version: Version
    policy_id: Identifier
    policy_version: Version
    provider_reference: Identifier
    execution_policy_reference: Identifier
    correlation_id: OpaqueId

    @model_validator(mode="after")
    def validate_size(self) -> "TrustedExecutionRequest":
        if len(self.canonical_bytes()) > REQUEST_MAX_BYTES:
            raise ValueError("canonical request exceeds size limit")
        return self

    @classmethod
    def from_json(cls, raw: str | bytes) -> "TrustedExecutionRequest":
        return cls.from_json_strict(raw, max_bytes=REQUEST_MAX_BYTES)


class PromptArtifact(ContractModel):
    prompt_id: Identifier
    prompt_version: Version
    media_type: Literal["text/plain; charset=utf-8"]
    byte_length: int = Field(strict=True, ge=0, le=1_000_000)
    content_digest: Digest
    manifest_binding: OpaqueId


class OrderedCase(ContractModel):
    case_id: Identifier
    revision: Version


class DatasetArtifact(ContractModel):
    dataset_id: Identifier
    dataset_version: Version
    media_type: Literal["application/aqef-golden-dataset+json"]
    byte_length: int = Field(strict=True, ge=2, le=10_000_000)
    content_digest: Digest
    case_set_digest: Digest
    ordered_cases: tuple[OrderedCase, ...] = Field(min_length=1, max_length=10_000)
    manifest_binding: OpaqueId

    @model_validator(mode="after")
    def unique_cases(self) -> "DatasetArtifact":
        case_ids = tuple(case.case_id for case in self.ordered_cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("ordered case IDs must be unique")
        return self


class EvaluatorArtifact(ContractModel):
    evaluator_id: Identifier
    evaluator_version: Version
    implementation_id: Identifier
    implementation_version: Version
    implementation_digest: Digest
    policy_id: Identifier
    policy_version: Version
    policy_digest: Digest


class PolicyValue(ContractModel):
    name: Identifier
    value: PolicyString | int | float | bool

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: str | int | float | bool) -> str | int | float | bool:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("policy values must be finite")
        return value


class PolicyArtifact(ContractModel):
    policy_id: Identifier
    policy_version: Version
    policy_digest: Digest
    values: tuple[PolicyValue, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def unique_values(self) -> "PolicyArtifact":
        names = tuple(item.name for item in self.values)
        if len(names) != len(set(names)):
            raise ValueError("policy value names must be unique")
        return self


class WorkerManifest(ContractModel):
    worker_identity: Identifier
    minimum_version: Version
    allowed_artifact_digests: tuple[Digest, ...] = Field(
        min_length=1,
        max_length=64,
    )


class ProviderManifest(ContractModel):
    provider_reference: Identifier
    adapter_identity: Identifier
    adapter_version: Version
    adapter_digest: Digest


class ExecutionPolicyManifest(ContractModel):
    execution_policy_reference: Identifier
    policy_version: Version
    policy_digest: Digest
    max_attempts: int = Field(strict=True, ge=1, le=5)
    timeout_milliseconds: int = Field(strict=True, ge=1, le=300_000)
    max_retry_delay_milliseconds: int = Field(strict=True, ge=0, le=300_000)


class KeyMetadata(ContractModel):
    manifest_key_id: Identifier
    algorithm: Literal["Ed25519"]


class ManifestPayload(ContractModel):
    manifest_version: Literal["1.0"]
    manifest_id: Identifier
    sequence: int = Field(strict=True, ge=1, le=MANIFEST_SEQUENCE_MAX)
    issued_at: Rfc3339Timestamp
    not_before: Rfc3339Timestamp
    expires_at: Rfc3339Timestamp
    worker: WorkerManifest
    prompts: tuple[PromptArtifact, ...] = Field(min_length=1, max_length=1024)
    datasets: tuple[DatasetArtifact, ...] = Field(min_length=1, max_length=1024)
    evaluators: tuple[EvaluatorArtifact, ...] = Field(min_length=1, max_length=1024)
    policies: tuple[PolicyArtifact, ...] = Field(min_length=1, max_length=1024)
    providers: tuple[ProviderManifest, ...] = Field(min_length=1, max_length=1024)
    execution_policies: tuple[ExecutionPolicyManifest, ...] = Field(
        min_length=1,
        max_length=1024,
    )
    key_metadata: KeyMetadata

    def digest(self) -> str:
        return canonical_digest(
            b"AQEF-MANIFEST-v1\0",
            self.model_dump(mode="json"),
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> "ManifestPayload":
        bindings = {
            *(artifact.manifest_binding for artifact in self.prompts),
            *(artifact.manifest_binding for artifact in self.datasets),
        }
        expected_binding = f"manifest-{self.manifest_id}-{self.sequence}"
        if bindings != {expected_binding}:
            raise ValueError("artifact manifest binding is inconsistent")
        for values, key in (
            (self.prompts, lambda item: (item.prompt_id, item.prompt_version)),
            (self.datasets, lambda item: (item.dataset_id, item.dataset_version)),
            (self.evaluators, lambda item: (item.evaluator_id, item.evaluator_version)),
            (self.policies, lambda item: (item.policy_id, item.policy_version)),
            (self.providers, lambda item: item.provider_reference),
            (
                self.execution_policies,
                lambda item: item.execution_policy_reference,
            ),
        ):
            identities = tuple(key(item) for item in values)
            if len(identities) != len(set(identities)):
                raise ValueError("manifest identities must be unique")
        not_before = _timestamp_instant(self.not_before)
        issued_at = _timestamp_instant(self.issued_at)
        expires_at = _timestamp_instant(self.expires_at)
        if not (not_before <= issued_at <= expires_at):
            raise ValueError("manifest timestamps are inconsistent")
        if len(self.canonical_bytes()) > MANIFEST_MAX_BYTES:
            raise ValueError("canonical manifest exceeds size limit")
        return self


class ManifestSignatureEnvelope(ContractModel):
    algorithm: Literal["Ed25519"]
    key_id: Identifier
    manifest_digest: Digest
    signature: Base64Url


class SignedManifest(ContractModel):
    json_max_bytes: ClassVar[int] = MANIFEST_MAX_BYTES
    payload: ManifestPayload
    signature: ManifestSignatureEnvelope

    @model_validator(mode="after")
    def verify_digest_binding(self) -> "SignedManifest":
        if self.signature.manifest_digest != self.payload.digest():
            raise ValueError("manifest signature envelope digest is inconsistent")
        return self

    @classmethod
    def from_json(cls, raw: str | bytes) -> "SignedManifest":
        return cls.from_json_strict(raw, max_bytes=MANIFEST_MAX_BYTES)


class TrustedReportPayload(ContractModel):
    schema_version: Version
    execution_status: Literal["not_executed", "unavailable", "failed", "executed"]
    outcome: Identifier
    policy_passed: bool = Field(strict=True)
    case_count: int = Field(strict=True, ge=0, le=10_000)
    completed_case_count: int = Field(strict=True, ge=0, le=10_000)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("details", mode="before")
    @classmethod
    def bounded_details(cls, value: Any) -> Any:
        _validate_report_details(value)
        return value

    @model_validator(mode="after")
    def freeze_details(self) -> "TrustedReportPayload":
        frozen = _freeze_json(self.details)
        if len(canonical_bytes(_thaw_json(frozen))) > REPORT_DETAIL_MAX_BYTES:
            raise ValueError("canonical report details exceed size limit")
        object.__setattr__(self, "details", frozen)
        return self

    @field_serializer("details")
    def serialize_details(self, value: Any) -> Any:
        return _thaw_json(value)

    @model_validator(mode="after")
    def consistent_counts(self) -> "TrustedReportPayload":
        if self.completed_case_count > self.case_count:
            raise ValueError("completed case count exceeds case count")
        return self


class TrustedProvenance(ContractModel):
    schema_version: Literal["1.0"]
    execution_id: OpaqueId
    request_id: OpaqueId
    verifier_nonce: Base64Url
    worker_identity: Identifier
    worker_version: Version
    worker_artifact_digest: Digest
    manifest_digest: Digest
    prompt_digest: Digest
    dataset_digest: Digest
    case_set_digest: Digest
    evaluator_identity: Identifier
    evaluator_digest: Digest
    policy_digest: Digest
    provider_identity: Identifier
    execution_policy_digest: Digest
    report_digest: Digest

    def digest(self) -> str:
        return canonical_digest(
            b"AQEF-PROVENANCE-v1\0",
            self.model_dump(mode="json"),
        )


class AttestedExecutionResult(ContractModel):
    json_max_bytes: ClassVar[int] = RESULT_MAX_BYTES
    schema_version: Literal["1.0"]
    execution_id: OpaqueId
    request_id: OpaqueId
    verifier_nonce: Base64Url
    worker_identity: Identifier
    worker_version: Version
    worker_artifact_digest: Digest
    manifest_digest: Digest
    prompt_digest: Digest
    dataset_digest: Digest
    case_set_digest: Digest
    evaluator_identity: Identifier
    evaluator_digest: Digest
    policy_digest: Digest
    provider_identity: Identifier
    execution_policy_digest: Digest
    report: TrustedReportPayload
    report_digest: Digest
    provenance: TrustedProvenance
    provenance_digest: Digest
    started_at: Rfc3339Timestamp
    completed_at: Rfc3339Timestamp
    attestation_key_id: Identifier
    attestation: Base64Url

    @model_validator(mode="after")
    def verify_internal_bindings(self) -> "AttestedExecutionResult":
        report_digest = canonical_digest(
            b"AQEF-REPORT-v1\0",
            self.report.model_dump(mode="json"),
        )
        if self.report_digest != report_digest:
            raise ValueError("report digest is inconsistent")
        if self.provenance_digest != self.provenance.digest():
            raise ValueError("provenance digest is inconsistent")
        expected = {
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "verifier_nonce": self.verifier_nonce,
            "worker_identity": self.worker_identity,
            "worker_version": self.worker_version,
            "worker_artifact_digest": self.worker_artifact_digest,
            "manifest_digest": self.manifest_digest,
            "prompt_digest": self.prompt_digest,
            "dataset_digest": self.dataset_digest,
            "case_set_digest": self.case_set_digest,
            "evaluator_identity": self.evaluator_identity,
            "evaluator_digest": self.evaluator_digest,
            "policy_digest": self.policy_digest,
            "provider_identity": self.provider_identity,
            "execution_policy_digest": self.execution_policy_digest,
            "report_digest": self.report_digest,
        }
        actual = self.provenance.model_dump(
            include=set(expected),
            mode="json",
        )
        if actual != expected:
            raise ValueError("provenance does not match attested execution")
        if _timestamp_instant(self.completed_at) < _timestamp_instant(self.started_at):
            raise ValueError("completion time precedes start time")
        if len(self.canonical_bytes()) > RESULT_MAX_BYTES:
            raise ValueError("canonical attested result exceeds size limit")
        return self

    def attestation_payload(self) -> bytes:
        values = self.model_dump(mode="json", exclude={"attestation"})
        return canonical_bytes(values)

    def attestation_digest(self) -> str:
        values = self.model_dump(mode="json", exclude={"attestation"})
        return canonical_digest(b"AQEF-EXECUTION-ATTESTATION-v1\0", values)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "AttestedExecutionResult":
        return cls.from_json_strict(raw, max_bytes=RESULT_MAX_BYTES)


class Phase10VerificationDecision(ContractModel):
    status: Literal["accepted", "rejected"]
    reason_code: Identifier
    execution_id: OpaqueId
    request_id: OpaqueId
    verifier_nonce: Base64Url

    @model_validator(mode="after")
    def validate_reason(self) -> "Phase10VerificationDecision":
        if self.status == "accepted" and self.reason_code != "verified":
            raise ValueError("accepted decisions require the verified reason code")
        if self.status == "rejected" and self.reason_code == "verified":
            raise ValueError("rejected decisions cannot use the verified reason code")
        return self


__all__ = [
    "AttestedExecutionResult",
    "DatasetArtifact",
    "EvaluatorArtifact",
    "ExecutionPolicyManifest",
    "KeyMetadata",
    "ManifestPayload",
    "ManifestSignatureEnvelope",
    "OrderedCase",
    "Phase10VerificationDecision",
    "PolicyArtifact",
    "PolicyValue",
    "PromptArtifact",
    "ProviderManifest",
    "SignedManifest",
    "TrustedExecutionRequest",
    "TrustedProvenance",
    "TrustedReportPayload",
    "WorkerManifest",
    "canonical_bytes",
    "canonical_digest",
    "json_digest",
]
