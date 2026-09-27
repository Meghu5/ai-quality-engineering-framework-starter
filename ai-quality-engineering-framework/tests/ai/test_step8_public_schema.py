from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_eval.models import Phase10Report
from ai_quality.dataset import (
    REGISTERED_DATASET_ID,
    REGISTERED_DATASET_VERSION,
    DatasetRegistry,
)
from ai_quality.evaluator_registry import (
    REGISTERED_EVALUATOR_ID,
    REGISTERED_EVALUATOR_VERSION,
    EvaluatorRegistry,
)
from ai_quality.execution_descriptor import TrustedExecutionDescriptor
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.provider_conformance import (
    ProviderConformanceReport,
    ProviderConformanceRunner,
    adapt_trusted_provider_result_to_phase10,
)
from ai_quality.provider_provenance import ProviderConformanceProvenance
from ai_quality.trusted_provider_result import (
    TrustedProviderConformanceResult,
    _public_provenance_from_trusted_result,
    _public_report_from_trusted_result,
)


REMOVED_REPORT_FIELDS = (
    "execution_descriptor_id",
    "prompt_id",
    "prompt_version",
    "golden_dataset_id",
    "golden_dataset_version",
    "evaluator_id",
    "evaluator_version",
    "evaluator_policy_id",
    "evaluator_policy_version",
    "evaluator_policy_fingerprint",
)


def _trusted_result():
    descriptor = TrustedExecutionDescriptor.create(
        registered_prompt=PromptRegistry().resolve("airline_assistant", "v1"),
        registered_dataset=DatasetRegistry().resolve(
            REGISTERED_DATASET_ID,
            REGISTERED_DATASET_VERSION,
        ),
        registered_evaluator=EvaluatorRegistry().resolve(
            REGISTERED_EVALUATOR_ID,
            REGISTERED_EVALUATOR_VERSION,
        ),
        provider_settings=None,
        provider_required=False,
        provider_id="test-provider",
        model_id="test-model",
    )
    runner = ProviderConformanceRunner(
        provider=None,
        provider_label="test-provider",
        model_label="test-model",
    )
    return descriptor, runner.run_trusted(descriptor)


def test_current_public_schema_has_one_nested_execution_identity():
    _, result = _trusted_result()
    report = _public_report_from_trusted_result(result)
    provenance = _public_provenance_from_trusted_result(result)
    payload = report.model_dump(mode="json")

    assert report.schema_version == "2.0"
    assert provenance.contract_version == "2.0"
    assert provenance.execution_evidence_id.startswith("execution-evidence-")
    assert "execution_descriptor_id" not in provenance.model_dump(mode="json")
    assert all(field not in payload for field in REMOVED_REPORT_FIELDS)
    assert payload["provenance"]["execution_evidence_id"] == result.execution_evidence_id


@pytest.mark.parametrize("field", REMOVED_REPORT_FIELDS)
def test_removed_flattened_identity_fields_are_forbidden(field):
    _, result = _trusted_result()
    payload = result.report.model_dump(mode="python")
    payload[field] = "caller-controlled"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProviderConformanceReport.model_validate(payload)


def test_execution_policy_no_longer_duplicates_contract_identity():
    _, result = _trusted_result()
    provenance = _public_provenance_from_trusted_result(result)
    policy = provenance.execution_policy.model_dump(mode="json")
    contract = provenance.contract.model_dump(mode="json")

    assert "structured_output_required" not in policy
    assert "capability_requirement_id" not in policy
    assert "capability_requirement_version" not in policy
    assert contract["structured_output_required"] is True
    assert contract["capability_contract_id"] == "provider-capability-contract"
    assert contract["capability_contract_version"] == "1.0"


def test_legacy_report_and_provenance_versions_are_explicitly_rejected():
    _, result = _trusted_result()
    report_payload = result.report.model_dump(mode="json")
    provenance_payload = result.provenance.model_dump(mode="json")
    report_payload["schema_version"] = "1.0"
    provenance_payload["contract_version"] = "1.0"

    with pytest.raises(ValidationError, match="2.0"):
        ProviderConformanceReport.model_validate(report_payload)
    with pytest.raises(ValidationError, match="2.0"):
        ProviderConformanceProvenance.model_validate(provenance_payload)


def test_matching_public_evidence_id_cannot_establish_authority():
    _, result = _trusted_result()
    public_provenance = ProviderConformanceProvenance.model_validate_json(
        result.provenance.model_dump_json()
    )
    public_report = ProviderConformanceReport.model_validate_json(
        result.report.model_dump_json()
    )

    assert public_provenance.execution_evidence_id == result.execution_evidence_id
    with pytest.raises(TypeError):
        TrustedExecutionDescriptor(public_provenance.execution_evidence_id)
    with pytest.raises(TypeError):
        TrustedProviderConformanceResult(public_report)
    with pytest.raises(TypeError, match="runner-issued trusted result"):
        _public_report_from_trusted_result(public_report)


def test_phase10_contains_only_canonical_nested_identity():
    _, result = _trusted_result()
    phase10 = adapt_trusted_provider_result_to_phase10(
        Phase10Report(
            baseline={},
            frameworks=[],
            comparisons=[],
            overall_passed=True,
        ),
        result,
    )
    evidence = phase10.baseline["provider_conformance"]
    serialized = json.dumps(evidence, sort_keys=True)

    assert all(field not in evidence for field in REMOVED_REPORT_FIELDS)
    assert evidence["provenance"]["execution_evidence_id"] == result.execution_evidence_id
    assert "execution_descriptor_id" not in serialized
