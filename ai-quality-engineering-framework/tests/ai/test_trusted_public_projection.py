from __future__ import annotations

import json

import pytest

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
from ai_quality.trusted_provider_result import (
    _public_provenance_from_trusted_result,
    _public_report_from_trusted_result,
)


def _descriptor(*, prompt_version: str = "v1") -> TrustedExecutionDescriptor:
    return TrustedExecutionDescriptor.create(
        registered_prompt=PromptRegistry().resolve(
            "airline_assistant",
            prompt_version,
        ),
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


def _trusted_result():
    descriptor = _descriptor()
    runner = ProviderConformanceRunner(
        provider=None,
        provider_label="test-provider",
        model_label="test-model",
    )
    return descriptor, runner.run_trusted(descriptor)


def _phase10() -> Phase10Report:
    return Phase10Report(
        baseline={},
        frameworks=[],
        comparisons=[],
        overall_passed=True,
    )


def test_trusted_result_projects_consistent_public_provenance_and_report():
    descriptor, result = _trusted_result()
    provenance = _public_provenance_from_trusted_result(result)
    report = _public_report_from_trusted_result(result)

    assert report is not result.report
    assert provenance is not result.provenance
    assert report.provenance == provenance
    assert provenance.execution_evidence_id == result.execution_evidence_id
    assert "execution_descriptor_id" not in type(report).model_fields
    assert "execution_descriptor_id" not in type(provenance).model_fields
    assert provenance.evaluation.prompt_id == descriptor.prompt.prompt.name
    assert provenance.evaluation.prompt_version == descriptor.prompt.prompt.version
    assert provenance.evaluation.golden_dataset_id == descriptor.dataset.dataset_id
    assert provenance.evaluation.evaluator_id == descriptor.evaluator.evaluator_id
    assert provenance.provider.provider_id == descriptor.provider_id


@pytest.mark.parametrize("source", ["report", "provenance", "dict", "json"])
def test_public_evidence_cannot_generate_trusted_projection(source):
    _, result = _trusted_result()
    values = {
        "report": result.report,
        "provenance": result.provenance,
        "dict": result.report.model_dump(mode="json"),
        "json": result.report.model_dump_json(),
    }

    with pytest.raises(TypeError, match="runner-issued trusted result"):
        _public_provenance_from_trusted_result(values[source])
    with pytest.raises(TypeError, match="runner-issued trusted result"):
        _public_report_from_trusted_result(values[source])


@pytest.mark.parametrize(
    "override",
    [
        {"prompt_id": "substituted"},
        {"dataset_id": "substituted"},
        {"evaluator_id": "substituted"},
        {"policy_version": "9.9"},
        {"provider_id": "provider-a"},
        {"contract_version": "9.9"},
    ],
)
def test_projection_accepts_no_caller_metadata_overrides(override):
    _, result = _trusted_result()

    with pytest.raises(TypeError):
        _public_report_from_trusted_result(result, **override)


def test_descriptor_substitution_invalidates_public_projection():
    _, result = _trusted_result()
    object.__setattr__(
        result,
        "_TrustedProviderConformanceResult__descriptor",
        _descriptor(prompt_version="v2"),
    )

    with pytest.raises(TypeError, match="runner-issued trusted result"):
        _public_report_from_trusted_result(result)


def test_outcome_substitution_invalidates_public_projection():
    _, result = _trusted_result()
    object.__setattr__(result.report, "outcome", "passed")

    with pytest.raises(TypeError, match="runner-issued trusted result"):
        _public_report_from_trusted_result(result)


def test_public_projection_is_detached_and_json_parseable():
    _, result = _trusted_result()
    first_report = _public_report_from_trusted_result(result)
    first_provenance = _public_provenance_from_trusted_result(result)
    object.__setattr__(first_report, "outcome", "passed")
    object.__setattr__(first_provenance, "provenance_id", "provenance-" + "0" * 64)

    second_report = _public_report_from_trusted_result(result)
    second_provenance = _public_provenance_from_trusted_result(result)
    parsed = ProviderConformanceReport.model_validate_json(
        second_report.model_dump_json()
    )

    assert second_report.outcome == "not_configured"
    assert second_provenance.provenance_id != first_provenance.provenance_id
    assert parsed == second_report


def test_projection_is_privacy_safe_through_phase10():
    descriptor, result = _trusted_result()
    provenance = _public_provenance_from_trusted_result(result)
    report = _public_report_from_trusted_result(result)
    phase10 = adapt_trusted_provider_result_to_phase10(_phase10(), result)
    serialized = json.dumps(
        {
            "provenance": provenance.model_dump(mode="json"),
            "report": report.model_dump(mode="json"),
            "phase10": phase10.model_dump(mode="json"),
        },
        sort_keys=True,
    )

    for forbidden in (
        descriptor.prompt.prompt.text,
        descriptor.prompt.prompt.purpose,
        descriptor.dataset._canonical_source,
        "_capability",
        "trusted_result_permit",
        "completed_execution",
        "trusted_execution",
        "factory",
        "Authorization",
        "api_key",
        "cookie",
        "environment",
        "request",
        "raw response",
    ):
        assert forbidden not in serialized
