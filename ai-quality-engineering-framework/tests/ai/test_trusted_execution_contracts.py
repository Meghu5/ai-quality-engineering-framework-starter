from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_quality.trusted_execution_contracts import (
    AttestedExecutionResult,
    DatasetArtifact,
    EvaluatorArtifact,
    ExecutionPolicyManifest,
    KeyMetadata,
    ManifestPayload,
    ManifestSignatureEnvelope,
    OrderedCase,
    Phase10VerificationDecision,
    PolicyArtifact,
    PolicyValue,
    PromptArtifact,
    ProviderManifest,
    MANIFEST_SEQUENCE_MAX,
    POLICY_STRING_MAX_LENGTH,
    REPORT_DETAIL_MAX_BYTES,
    REPORT_DETAIL_MAX_DEPTH,
    REPORT_DETAIL_MAX_ITEMS,
    REPORT_DETAIL_MAX_STRING_LENGTH,
    SignedManifest,
    TrustedExecutionRequest,
    TrustedProvenance,
    TrustedReportPayload,
    WorkerManifest,
    canonical_bytes,
    canonical_digest,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
NONCE = "nonce_AAAAAAAAAAAAAAAAAAAAAA"


def _request_values() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "request_id": "req-1234567890abcdef",
        "verifier_nonce": NONCE,
        "prompt_id": "airline-assistant",
        "prompt_version": "v1",
        "dataset_id": "airline-quality-golden-cases",
        "dataset_version": "1.0",
        "evaluator_id": "quality-gate-evaluator",
        "evaluator_version": "1.0",
        "policy_id": "quality-gate-thresholds",
        "policy_version": "1.0",
        "provider_reference": "provider-test",
        "execution_policy_reference": "bounded-http-retry",
        "correlation_id": "corr-1234567890abcdef",
    }


def _prompt() -> PromptArtifact:
    return PromptArtifact(
        prompt_id="airline-assistant",
        prompt_version="v1",
        media_type="text/plain; charset=utf-8",
        byte_length=128,
        content_digest=DIGEST_A,
        manifest_binding="manifest-airline-quality-production-1",
    )


def _dataset() -> DatasetArtifact:
    return DatasetArtifact(
        dataset_id="airline-quality-golden-cases",
        dataset_version="1.0",
        media_type="application/aqef-golden-dataset+json",
        byte_length=1024,
        content_digest=DIGEST_B,
        case_set_digest=DIGEST_C,
        ordered_cases=(OrderedCase(case_id="case-flight-search", revision="1"),),
        manifest_binding="manifest-airline-quality-production-1",
    )


def _evaluator() -> EvaluatorArtifact:
    return EvaluatorArtifact(
        evaluator_id="quality-gate-evaluator",
        evaluator_version="1.0",
        implementation_id="aqef-quality-gate",
        implementation_version="1.0.0",
        implementation_digest=DIGEST_A,
        policy_id="quality-gate-thresholds",
        policy_version="1.0",
        policy_digest=DIGEST_B,
    )


def _manifest_payload() -> ManifestPayload:
    return ManifestPayload(
        manifest_version="1.0",
        manifest_id="airline-quality-production",
        sequence=1,
        issued_at="2026-09-28T10:00:00Z",
        not_before="2026-09-28T09:00:00Z",
        expires_at="2027-09-28T10:00:00Z",
        worker=WorkerManifest(
            worker_identity="aqef-trusted-worker",
            minimum_version="1.0.0",
            allowed_artifact_digests=(DIGEST_A,),
        ),
        prompts=(_prompt(),),
        datasets=(_dataset(),),
        evaluators=(_evaluator(),),
        policies=(
            PolicyArtifact(
                policy_id="quality-gate-thresholds",
                policy_version="1.0",
                policy_digest=DIGEST_B,
                values=(PolicyValue(name="intent-accuracy", value=0.95),),
            ),
        ),
        providers=(
            ProviderManifest(
                provider_reference="provider-test",
                adapter_identity="deterministic-provider",
                adapter_version="1.0",
                adapter_digest=DIGEST_C,
            ),
        ),
        execution_policies=(
            ExecutionPolicyManifest(
                execution_policy_reference="bounded-http-retry",
                policy_version="1.0",
                policy_digest=DIGEST_A,
                max_attempts=3,
                timeout_milliseconds=30_000,
                max_retry_delay_milliseconds=5_000,
            ),
        ),
        key_metadata=KeyMetadata(
            manifest_key_id="release-key",
            algorithm="Ed25519",
        ),
    )


def _signed_manifest() -> SignedManifest:
    payload = _manifest_payload()
    return SignedManifest(
        payload=payload,
        signature=ManifestSignatureEnvelope(
            algorithm="Ed25519",
            key_id="release-key",
            manifest_digest=payload.digest(),
            signature="signature_AAAAAAAAAAAAAAAAAAAAAA",
        ),
    )


def _report() -> TrustedReportPayload:
    return TrustedReportPayload(
        schema_version="1.0",
        execution_status="executed",
        outcome="passed",
        policy_passed=True,
        case_count=1,
        completed_case_count=1,
        details={"quality_score": 0.99},
    )


def _provenance(report_digest: str) -> TrustedProvenance:
    return TrustedProvenance(
        schema_version="1.0",
        execution_id="exec-1234567890abcdef",
        request_id="req-1234567890abcdef",
        verifier_nonce=NONCE,
        worker_identity="aqef-trusted-worker",
        worker_version="1.0.0",
        worker_artifact_digest=DIGEST_A,
        manifest_digest=DIGEST_B,
        prompt_digest=DIGEST_A,
        dataset_digest=DIGEST_B,
        case_set_digest=DIGEST_C,
        evaluator_identity="quality-gate-evaluator",
        evaluator_digest=DIGEST_A,
        policy_digest=DIGEST_B,
        provider_identity="provider-test",
        execution_policy_digest=DIGEST_C,
        report_digest=report_digest,
    )


def _result() -> AttestedExecutionResult:
    report = _report()
    report_digest = canonical_digest(
        b"AQEF-REPORT-v1\0",
        report.model_dump(mode="json"),
    )
    provenance = _provenance(report_digest)
    return AttestedExecutionResult(
        schema_version="1.0",
        execution_id=provenance.execution_id,
        request_id=provenance.request_id,
        verifier_nonce=provenance.verifier_nonce,
        worker_identity=provenance.worker_identity,
        worker_version=provenance.worker_version,
        worker_artifact_digest=provenance.worker_artifact_digest,
        manifest_digest=provenance.manifest_digest,
        prompt_digest=provenance.prompt_digest,
        dataset_digest=provenance.dataset_digest,
        case_set_digest=provenance.case_set_digest,
        evaluator_identity=provenance.evaluator_identity,
        evaluator_digest=provenance.evaluator_digest,
        policy_digest=provenance.policy_digest,
        provider_identity=provenance.provider_identity,
        execution_policy_digest=provenance.execution_policy_digest,
        report=report,
        report_digest=report_digest,
        provenance=provenance,
        provenance_digest=provenance.digest(),
        started_at="2026-09-28T10:00:00Z",
        completed_at="2026-09-28T10:00:01Z",
        attestation_key_id="worker-key",
        attestation="attestation_AAAAAAAAAAAAAAAAAAAA",
    )


def test_positive_contracts_and_deterministic_canonicalization():
    request = TrustedExecutionRequest(**_request_values())
    manifest = _signed_manifest()
    result = _result()
    decision = Phase10VerificationDecision(
        status="accepted",
        reason_code="verified",
        execution_id=result.execution_id,
        request_id=result.request_id,
        verifier_nonce=result.verifier_nonce,
    )

    assert request == TrustedExecutionRequest.from_json(request.canonical_bytes())
    assert manifest == SignedManifest.from_json(manifest.canonical_bytes())
    assert result == AttestedExecutionResult.from_json(result.canonical_bytes())
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert manifest.payload.digest().startswith("sha256:")
    assert result.attestation_digest().startswith("sha256:")
    assert decision.status == "accepted"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1.0),
        ("request_id", "x"),
        ("prompt_id", "../prompt"),
        ("dataset_id", "C:\\dataset.json"),
        ("evaluator_id", "https://example.test/evaluator"),
        ("policy_id", "package.module"),
        ("provider_reference", {"callable": "value"}),
        ("execution_policy_reference", ["policy"]),
    ],
)
def test_request_rejects_wrong_types_and_unsafe_identifiers(field, value):
    values = _request_values()
    values[field] = value
    with pytest.raises(ValidationError):
        TrustedExecutionRequest(**values)


def test_request_rejects_unknown_missing_duplicate_and_oversized_input():
    values = _request_values()
    with pytest.raises(ValidationError):
        TrustedExecutionRequest(**values, metadata="forbidden")
    missing = dict(values)
    missing.pop("prompt_id")
    with pytest.raises(ValidationError):
        TrustedExecutionRequest(**missing)
    duplicate = json.dumps(values)[:-1] + ',"prompt_id":"substituted"}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        TrustedExecutionRequest.from_json(duplicate)
    with pytest.raises(ValueError, match="size limit"):
        TrustedExecutionRequest.from_json(" " * 5000 + json.dumps(values))


def test_duplicate_json_keys_cannot_bypass_any_public_parser():
    request = TrustedExecutionRequest(**_request_values())
    duplicate = request.model_dump_json().replace(
        '"prompt_id":"airline-assistant"',
        '"prompt_id":"airline-assistant","prompt_id":"substituted-prompt"',
    )

    for parser in (
        TrustedExecutionRequest.from_json,
        TrustedExecutionRequest.model_validate_json,
    ):
        with pytest.raises(ValueError, match="duplicate JSON key"):
            parser(duplicate)


def test_nested_duplicate_json_keys_cannot_bypass_any_public_parser():
    manifest = _signed_manifest()
    duplicate = manifest.model_dump_json().replace(
        '"worker_identity":"aqef-trusted-worker"',
        (
            '"worker_identity":"aqef-trusted-worker",'
            '"worker_identity":"substituted-worker"'
        ),
    )

    for parser in (SignedManifest.from_json, SignedManifest.model_validate_json):
        with pytest.raises(ValueError, match="duplicate JSON key"):
            parser(duplicate)


def test_public_json_parser_rejects_non_finite_and_oversized_input():
    report = _report().model_dump_json().replace("0.99", "NaN")
    with pytest.raises(ValueError, match="non-finite"):
        TrustedReportPayload.model_validate_json(report)
    request = TrustedExecutionRequest(**_request_values()).model_dump_json()
    with pytest.raises(ValueError, match="size limit"):
        TrustedExecutionRequest.model_validate_json(" " * 5000 + request)


def test_legacy_raw_parsers_cannot_bypass_controlled_json_ingestion():
    request = TrustedExecutionRequest(**_request_values()).model_dump_json()
    duplicate = request.replace(
        '"prompt_id":"airline-assistant"',
        '"prompt_id":"airline-assistant","prompt_id":"substituted-prompt"',
    )
    nested_duplicate = _signed_manifest().model_dump_json().replace(
        '"worker_identity":"aqef-trusted-worker"',
        (
            '"worker_identity":"aqef-trusted-worker",'
            '"worker_identity":"substituted-worker"'
        ),
    )

    for model, raw in (
        (TrustedExecutionRequest, duplicate),
        (TrustedExecutionRequest, " " * 5000 + request),
        (SignedManifest, nested_duplicate),
        (TrustedReportPayload, _report().model_dump_json().replace("0.99", "NaN")),
    ):
        with pytest.raises(TypeError, match="legacy raw parsing is unavailable"):
            model.parse_raw(raw)

    with pytest.raises(TypeError, match="legacy file parsing is unavailable"):
        TrustedExecutionRequest.parse_file("untrusted-contract.json")

    assert TrustedExecutionRequest.from_json(request) == (
        TrustedExecutionRequest.model_validate_json(request)
    )


def test_artifact_and_manifest_consistency_validation():
    with pytest.raises(ValidationError, match="unique"):
        DatasetArtifact(
            **_dataset().model_dump(exclude={"ordered_cases"}),
            ordered_cases=(
                OrderedCase(case_id="case-one", revision="1"),
                OrderedCase(case_id="case-one", revision="2"),
            ),
        )
    payload = _manifest_payload()
    with pytest.raises(ValidationError, match="digest is inconsistent"):
        SignedManifest(
            payload=payload,
            signature=ManifestSignatureEnvelope(
                algorithm="Ed25519",
                key_id="release-key",
                manifest_digest=DIGEST_A,
                signature="signature_AAAAAAAAAAAAAAAAAAAAAA",
            ),
        )


def test_signature_envelope_is_not_part_of_manifest_payload_digest():
    first = _signed_manifest()
    second = first.model_copy(
        update={
            "signature": first.signature.model_copy(
                update={"signature": "signature_BBBBBBBBBBBBBBBBBBBBBB"}
            )
        }
    )

    assert first.payload.digest() == second.payload.digest()


def test_attested_result_rejects_inconsistent_digests_and_bindings():
    result = _result()
    for update in (
        {"report_digest": DIGEST_A},
        {"provenance_digest": DIGEST_A},
        {"prompt_digest": DIGEST_C},
        {"completed_at": "2026-09-28T09:59:59Z"},
    ):
        with pytest.raises(ValidationError):
            AttestedExecutionResult.model_validate(
                result.model_dump(mode="python") | update
            )


def test_digest_covered_report_data_is_deeply_immutable_and_alias_safe():
    original = {
        "quality": {"score": 0.99},
        "checks": ["contract", {"safety": True}],
    }
    report = TrustedReportPayload(
        schema_version="1.0",
        execution_status="executed",
        outcome="passed",
        policy_passed=True,
        case_count=1,
        completed_case_count=1,
        details=original,
    )
    report_digest = canonical_digest(
        b"AQEF-REPORT-v1\0",
        report.model_dump(mode="json"),
    )

    original["quality"]["score"] = 0.01
    original["checks"].append("substituted")
    with pytest.raises(TypeError):
        report.details["quality"] = {"score": 0.01}
    with pytest.raises(TypeError):
        report.details["quality"]["score"] = 0.01
    with pytest.raises(AttributeError):
        report.details["checks"].append("substituted")

    assert report.details["quality"]["score"] == 0.99
    assert report.details["checks"] == ("contract", {"safety": True})
    assert report_digest == canonical_digest(
        b"AQEF-REPORT-v1\0",
        report.model_dump(mode="json"),
    )
    round_trip = TrustedReportPayload.model_validate_json(report.model_dump_json())
    assert round_trip == report

    result = _result()
    attested_report_digest = result.report_digest
    provenance_digest = result.provenance_digest
    with pytest.raises(TypeError):
        result.report.details["quality_score"] = 0.0
    assert result.report_digest == attested_report_digest
    assert result.provenance_digest == provenance_digest
    assert result.provenance.digest() == provenance_digest


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-01T00:00:00Z",
        "2026-01-01T23:59:59Z",
        "2026-09-28T12:00:00Z",
        "2026-02-28T12:00:00Z",
        "2024-02-29T12:00:00Z",
        "2026-09-28T12:00:00+04:00",
    ],
)
def test_valid_rfc3339_timestamp_is_accepted(timestamp):
    values = _manifest_payload().model_dump(mode="python")
    values.update(
        {
            "not_before": timestamp,
            "issued_at": timestamp,
            "expires_at": timestamp,
        }
    )
    assert ManifestPayload.model_validate(values).issued_at == timestamp


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-01-01T24:00:00Z",
        "2026-01-01T24:00:00+00:00",
        "2026-01-01T24:00:00+04:00",
        "2026-99-99T99:99:99Z",
        "2026-02-30T12:00:00Z",
        "2026-13-01T12:00:00Z",
        "2026-09-28T25:00:00Z",
        "2026-09-28T12:60:00Z",
    ],
)
def test_invalid_rfc3339_timestamp_is_rejected(timestamp):
    values = _manifest_payload().model_dump(mode="python")
    values["issued_at"] = timestamp
    with pytest.raises(ValidationError):
        ManifestPayload.model_validate(values)


def test_manifest_validity_period_compares_actual_instants():
    values = _manifest_payload().model_dump(mode="python")
    values.update(
        {
            "not_before": "2026-09-28T12:00:00+04:00",
            "issued_at": "2026-09-28T08:30:00Z",
            "expires_at": "2026-09-28T09:00:00Z",
        }
    )
    assert ManifestPayload.model_validate(values).issued_at.endswith("Z")
    values["expires_at"] = "2026-09-28T08:29:59Z"
    with pytest.raises(ValidationError, match="timestamps are inconsistent"):
        ManifestPayload.model_validate(values)


def test_policy_string_and_manifest_sequence_limits_are_enforced():
    assert PolicyValue(
        name="policy-value",
        value="x" * POLICY_STRING_MAX_LENGTH,
    ).value
    with pytest.raises(ValidationError):
        PolicyValue(
            name="policy-value",
            value="x" * (POLICY_STRING_MAX_LENGTH + 1),
        )

    values = _manifest_payload().model_dump(mode="python")
    values["sequence"] = MANIFEST_SEQUENCE_MAX
    bindings = f"manifest-airline-quality-production-{MANIFEST_SEQUENCE_MAX}"
    values["prompts"][0]["manifest_binding"] = bindings
    values["datasets"][0]["manifest_binding"] = bindings
    assert ManifestPayload.model_validate(values).sequence == MANIFEST_SEQUENCE_MAX
    values["sequence"] = MANIFEST_SEQUENCE_MAX + 1
    with pytest.raises(ValidationError):
        ManifestPayload.model_validate(values)


def _nested_details(depth: int):
    value = "leaf"
    for _ in range(depth):
        value = {"node": value}
    return value


def _report_with_details(details) -> TrustedReportPayload:
    return TrustedReportPayload(
        schema_version="1.0",
        execution_status="executed",
        outcome="passed",
        policy_passed=True,
        case_count=1,
        completed_case_count=1,
        details=details,
    )


def test_report_detail_depth_item_and_string_limits_are_enforced():
    assert _report_with_details(_nested_details(REPORT_DETAIL_MAX_DEPTH))
    with pytest.raises(ValidationError, match="depth limit"):
        _report_with_details(_nested_details(REPORT_DETAIL_MAX_DEPTH + 1))

    at_item_limit = {f"k{index}": index for index in range(REPORT_DETAIL_MAX_ITEMS)}
    assert _report_with_details(at_item_limit)
    above_item_limit = dict(at_item_limit)
    above_item_limit["overflow"] = True
    with pytest.raises(ValidationError, match="item count"):
        _report_with_details(above_item_limit)

    assert _report_with_details({"value": "x" * REPORT_DETAIL_MAX_STRING_LENGTH})
    with pytest.raises(ValidationError, match="string exceeds"):
        _report_with_details(
            {"value": "x" * (REPORT_DETAIL_MAX_STRING_LENGTH + 1)}
        )


def test_report_detail_canonical_size_limit_is_enforced_at_boundary():
    details = {f"k{index:02d}": "" for index in range(16)}
    remaining = REPORT_DETAIL_MAX_BYTES - len(canonical_bytes(details))
    for key in details:
        allocated = min(remaining, REPORT_DETAIL_MAX_STRING_LENGTH)
        details[key] = "x" * allocated
        remaining -= allocated
    assert remaining == 0
    report = _report_with_details(details)
    assert len(canonical_bytes(report.model_dump(mode="json")["details"])) == (
        REPORT_DETAIL_MAX_BYTES
    )

    oversized = dict(details)
    first_key = next(
        key
        for key, value in oversized.items()
        if len(value) < REPORT_DETAIL_MAX_STRING_LENGTH
    )
    oversized[first_key] += "x"
    with pytest.raises(ValidationError, match="size limit"):
        _report_with_details(oversized)


def test_non_finite_and_noncanonicalizable_values_are_rejected():
    with pytest.raises(ValidationError):
        TrustedReportPayload(
            schema_version="1.0",
            execution_status="executed",
            outcome="passed",
            policy_passed=True,
            case_count=1,
            completed_case_count=1,
            details={"score": float("nan")},
        )
    with pytest.raises(ValueError):
        canonical_bytes({"score": float("inf")})
    with pytest.raises(ValueError):
        canonical_bytes({"callback": lambda: None})


def test_contracts_never_execute_smuggled_python_objects():
    calls = 0

    class Executable:
        def __call__(self):
            nonlocal calls
            calls += 1

    executable = Executable()
    request = _request_values()
    request["prompt_id"] = executable
    with pytest.raises(ValidationError):
        TrustedExecutionRequest(**request)
    with pytest.raises(ValidationError):
        PolicyValue(name="intent-accuracy", value=executable)
    with pytest.raises(ValidationError):
        TrustedReportPayload(
            schema_version="1.0",
            execution_status="executed",
            outcome="passed",
            policy_passed=True,
            case_count=0,
            completed_case_count=0,
            details={"callback": executable},
        )

    assert calls == 0


def test_verification_decision_is_data_not_self_authenticating_authority():
    accepted = Phase10VerificationDecision(
        status="accepted",
        reason_code="verified",
        execution_id="exec-1234567890abcdef",
        request_id="req-1234567890abcdef",
        verifier_nonce=NONCE,
    )
    assert accepted.status == "accepted"
    assert not hasattr(accepted, "verify")
    assert not hasattr(accepted, "authorize")
    with pytest.raises(ValidationError):
        Phase10VerificationDecision(
            status="accepted",
            reason_code="signature-invalid",
            execution_id=accepted.execution_id,
            request_id=accepted.request_id,
            verifier_nonce=accepted.verifier_nonce,
        )
