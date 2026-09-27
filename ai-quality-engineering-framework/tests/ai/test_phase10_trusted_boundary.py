from __future__ import annotations

import copy
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
from ai_quality.models import AirlineAssistantResponse
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.provider_conformance import (
    ProviderConformanceReport,
    ProviderConformanceRunner,
    ProviderExecutionPolicy,
    adapt_provider_report_to_phase10,
    adapt_trusted_provider_result_to_phase10,
)
from ai_quality.providers import LLMProvider
from ai_quality.trusted_provider_result import TrustedProviderConformanceResult


class _ConformingProvider(LLMProvider):
    def health_check(self) -> bool:
        return True

    def generate(self, user_input, *, prompt, context=None):
        return "unused"

    def generate_structured(self, user_input, *, prompt, case=None, context=None):
        if case is None:
            return AirlineAssistantResponse(
                intent="flight_search",
                response="Structured airline response.",
                prompt_version=prompt.version,
            )
        response_terms = case.reference_answer_terms + case.required_context_terms
        return AirlineAssistantResponse(
            intent=case.expected_intent,
            response=" ".join(response_terms) or "Structured airline response.",
            entities=dict(case.expected_entities),
            actions=[case.required_action.model_copy(deep=True)]
            if case.required_action is not None
            else [],
            safety={
                "safe": True,
                "refusal": case.expect_refusal,
                "pii_detected": case.expect_pii,
                "pii_redacted": True,
                "prompt_injection_detected": case.category == "prompt_injection",
            },
            prompt_version=prompt.version,
        )


def _phase10() -> Phase10Report:
    return Phase10Report(
        baseline={"phase8": {"overall_passed": True}},
        frameworks=[],
        comparisons=[],
        overall_passed=True,
    )


def _descriptor(*, prompt_version: str = "v1", required: bool = False):
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
        provider_required=required,
        provider_id="test-provider",
        model_id="test-model",
    )


def _trusted(*, provider=True, required=False):
    descriptor = _descriptor(required=required)
    runner = ProviderConformanceRunner(
        provider=_ConformingProvider() if provider else None,
        provider_label="test-provider",
        model_label="test-model",
        policy=ProviderExecutionPolicy(required=required),
    )
    return descriptor, runner.run_trusted(descriptor)


def test_trusted_result_enters_phase10_without_changing_scoring_evidence():
    descriptor, result = _trusted()
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)
    evidence = adapted.baseline["provider_conformance"]

    assert result.outcome == "passed"
    assert result.report.quality_report.overall_passed
    assert evidence == result.report.model_dump(mode="json")
    assert evidence["provenance"] == result.provenance.model_dump(mode="json")
    assert (
        evidence["provenance"]["execution_evidence_id"]
        == result.execution_evidence_id
    )
    assert (
        descriptor.dataset.dataset_id
        == evidence["provenance"]["evaluation"]["golden_dataset_id"]
    )
    assert adapted.overall_passed


@pytest.mark.parametrize("kind", ["report", "clone", "json", "dict", "provenance"])
def test_public_evidence_cannot_enter_trusted_phase10_adapter(kind):
    _, result = _trusted()
    values = {
        "report": result.report,
        "clone": result.report.model_copy(deep=True),
        "json": result.report.model_dump_json(),
        "dict": result.report.model_dump(mode="json"),
        "provenance": result.provenance,
    }

    with pytest.raises(TypeError, match="runner-issued trusted result"):
        adapt_trusted_provider_result_to_phase10(_phase10(), values[kind])


def test_legacy_public_report_adapter_fails_closed():
    _, result = _trusted()
    parsed = ProviderConformanceReport.model_validate_json(
        result.report.model_dump_json()
    )

    with pytest.raises(TypeError, match="cannot enter"):
        adapt_provider_report_to_phase10(_phase10(), parsed)


def test_forged_and_copied_trusted_results_are_rejected():
    _, result = _trusted()
    forged = object.__new__(TrustedProviderConformanceResult)
    object.__setattr__(
        forged,
        "_TrustedProviderConformanceResult__descriptor",
        result.descriptor,
    )
    object.__setattr__(
        forged,
        "_TrustedProviderConformanceResult__report",
        result.report,
    )
    object.__setattr__(
        forged,
        "_TrustedProviderConformanceResult__execution_validator",
        lambda candidate: False,
    )

    with pytest.raises(TypeError, match="runner-issued trusted result"):
        adapt_trusted_provider_result_to_phase10(_phase10(), forged)
    with pytest.raises(TypeError, match="only be minted"):
        copy.copy(result)


def test_descriptor_or_report_substitution_invalidates_enrollment():
    _, result = _trusted()
    other = _descriptor(prompt_version="v2")
    object.__setattr__(
        result,
        "_TrustedProviderConformanceResult__descriptor",
        other,
    )

    with pytest.raises(TypeError, match="runner-issued trusted result"):
        adapt_trusted_provider_result_to_phase10(_phase10(), result)


def test_failure_outcome_is_not_promoted_to_phase10_success():
    _, result = _trusted(provider=False, required=True)
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)

    assert result.outcome == "not_configured"
    assert not result.report.policy_passed
    assert not adapted.overall_passed


def test_phase10_projection_contains_no_internal_authority():
    descriptor, result = _trusted()
    adapted = adapt_trusted_provider_result_to_phase10(_phase10(), result)
    serialized = json.dumps(adapted.model_dump(mode="json"), sort_keys=True)

    for forbidden in (
        "_capability",
        "trusted_result_permit",
        "completed_execution",
        "trusted_execution",
        descriptor.prompt.prompt.text,
        descriptor.dataset._canonical_source,
        "Authorization",
        "api_key",
        "cookie",
        "environment",
    ):
        assert forbidden not in serialized
