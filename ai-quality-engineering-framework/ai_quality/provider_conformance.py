from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_eval.models import ExecutionStatus, Phase10Report
from ai_quality.dataset import DATASET_PATH, load_golden_cases
from ai_quality.evaluators import QualityGateEvaluator
from ai_quality.models import AirlineAssistantResponse, GoldenCase, QualityReport
from ai_quality.provider_capabilities import (
    ProviderCapabilityEvidence,
    ProviderCapabilityRequirement,
    SchemaCompatibilityEvidence,
    assess_provider_capabilities,
    schema_compatibility_for_declaration,
    with_empirical_compatibility,
)
from ai_quality.privacy import (
    _SafeIdentifier,
    safe_case_id,
    safe_model_label,
    safe_provider_label,
)
from ai_quality.prompt_registry import Prompt
from ai_quality.provider_resilience import ProviderFailureMetadata
from ai_quality.providers import (
    LLMProvider,
    OptionalRealLLMProvider,
    RealProviderDisabledError,
    RealProviderError,
    RealProviderResponseError,
    RealProviderTransportError,
)
from observability.models import FailureCategory
from observability.tracing import TracingFacade, create_tracing_facade

if TYPE_CHECKING:
    from ai_quality.provider_config import RealLLMProviderSettings


ProviderConformanceOutcome = Literal[
    "not_configured",
    "unavailable",
    "transport_failed",
    "contract_failed",
    "capability_failed",
    "schema_mismatch",
    "quality_failed",
    "passed",
]

ReasonCode = Literal[
    "provider_not_configured",
    "provider_unavailable",
    "provider_authentication_failed",
    "provider_authorization_failed",
    "provider_timeout",
    "provider_network_failed",
    "provider_rate_limited",
    "provider_failed",
    "provider_response_malformed",
    "provider_capability_unsupported",
    "provider_capability_invalid",
    "provider_schema_mismatch",
    "quality_gate_failed",
]

_PREFLIGHT_USER_INPUT = "Verify structured airline response compatibility."
_PREFLIGHT_PROMPT = Prompt(
    name="provider_capability_preflight",
    version="v1",
    purpose="provider schema conformance",
    text="Return one valid structured airline assistant response.",
)

class ProviderFailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    category: FailureCategory
    retryable: bool = Field(strict=True)
    attempt: int = Field(ge=1, strict=True)
    max_attempts: int = Field(ge=1, strict=True)
    status_code: int | None = Field(default=None, ge=100, le=599, strict=True)

    @model_validator(mode="after")
    def validate_attempt(self) -> "ProviderFailureEvidence":
        if self.attempt > self.max_attempts:
            raise ValueError("attempt must not exceed max_attempts")
        return self


class _ImmutableQualityReport(QualityReport):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProviderConformanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"] = "1.0"
    provider: str
    model: str
    required: bool = Field(strict=True)
    execution_status: ExecutionStatus
    outcome: ProviderConformanceOutcome
    failure_category: FailureCategory | None = None
    reason_code: ReasonCode | None
    case_count: int = Field(ge=0, strict=True)
    completed_case_count: int = Field(ge=0, strict=True)
    case_ids: tuple[str, ...] = Field(default_factory=tuple)
    failure: ProviderFailureEvidence | None = None
    capability: ProviderCapabilityEvidence | None = None
    schema_compatibility: SchemaCompatibilityEvidence
    quality_report: _ImmutableQualityReport | None = None
    policy_passed: bool = Field(strict=True)

    @field_validator("provider", "model", mode="before")
    @classmethod
    def sanitize_labels(cls, value: object, info) -> str:
        if isinstance(value, _SafeIdentifier):
            return value.value
        if not isinstance(value, str):
            raise ValueError("provider and model labels must be strings")
        sanitizer = safe_provider_label if info.field_name == "provider" else safe_model_label
        return sanitizer(value).value

    @field_validator("case_ids", mode="before")
    @classmethod
    def sanitize_case_ids(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, (str, _SafeIdentifier)) for item in value
        ):
            raise ValueError("case_ids must be strings")
        return tuple(
            item.value if isinstance(item, _SafeIdentifier) else safe_case_id(item).value
            for item in value
        )

    @field_validator("quality_report", mode="before")
    @classmethod
    def freeze_quality_report(cls, value: object) -> object:
        if isinstance(value, QualityReport):
            return _ImmutableQualityReport.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "ProviderConformanceReport":
        if self.completed_case_count > self.case_count:
            raise ValueError("completed_case_count must not exceed case_count")
        if len(self.case_ids) != self.case_count:
            raise ValueError("case_ids must match case_count")
        expected_status: dict[ProviderConformanceOutcome, ExecutionStatus] = {
            "not_configured": (
                "provider_required" if self.required else "not_executed"
            ),
            "unavailable": "unavailable",
            "transport_failed": "failed",
            "contract_failed": "failed",
            "capability_failed": "failed",
            "schema_mismatch": "failed",
            "quality_failed": "executed",
            "passed": "executed",
        }
        expected_reason: dict[ProviderConformanceOutcome, set[ReasonCode | None]] = {
            "not_configured": {"provider_not_configured"},
            "unavailable": {"provider_unavailable"},
            "transport_failed": {
                "provider_authentication_failed",
                "provider_authorization_failed",
                "provider_timeout",
                "provider_network_failed",
                "provider_rate_limited",
                "provider_failed",
            },
            "contract_failed": {"provider_response_malformed"},
            "capability_failed": {
                "provider_capability_unsupported",
                "provider_capability_invalid",
            },
            "schema_mismatch": {"provider_schema_mismatch"},
            "quality_failed": {"quality_gate_failed"},
            "passed": {None},
        }
        if self.execution_status != expected_status[self.outcome]:
            raise ValueError("execution_status is inconsistent with outcome")
        if self.reason_code not in expected_reason[self.outcome]:
            raise ValueError("reason_code is inconsistent with outcome")
        if self.outcome in {"transport_failed", "contract_failed", "schema_mismatch"}:
            expected_failure_reason = _failure_reason_code(
                self.outcome, self.failure_category
            )
            if (
                expected_failure_reason is None
                or self.reason_code != expected_failure_reason
            ):
                raise ValueError("reason_code is inconsistent with failure category")
        expected_policy = ProviderExecutionPolicy(required=self.required).permits(
            self.outcome
        )
        if self.policy_passed != expected_policy:
            raise ValueError("policy_passed is inconsistent with outcome")

        if self.capability is not None:
            capability_outcome = {
                "unsupported": "capability_failed",
                "schema_mismatch": "schema_mismatch",
            }.get(self.capability.status)
            if capability_outcome is not None and self.outcome != capability_outcome:
                raise ValueError("capability status is inconsistent with outcome")
            if (
                self.capability.status == "unknown"
                and self.capability.reason_code == "provider_capability_invalid"
                and self.outcome != "capability_failed"
            ):
                raise ValueError("invalid capability declaration must fail capability")
            declaration_by_status = {
                "supported": {"compatible"},
                "unsupported": {"incompatible"},
                "schema_mismatch": {"incompatible"},
                "unknown": {"unknown", "invalid", "absent"},
            }
            if self.schema_compatibility.declaration not in declaration_by_status[
                self.capability.status
            ]:
                raise ValueError(
                    "schema compatibility declaration conflicts with capability"
                )

        if self.outcome == "not_configured":
            if (
                self.completed_case_count != 0
                or self.failure is not None
                or self.failure_category is not None
            ):
                raise ValueError("not_configured cannot contain completed cases or failure")
        elif self.outcome == "unavailable":
            if self.completed_case_count != 0 or self.failure is None:
                raise ValueError("unavailable requires safe failure metadata")
            if self.failure.category not in {
                FailureCategory.NETWORK,
                FailureCategory.PROVIDER,
                FailureCategory.TIMEOUT,
                FailureCategory.UNKNOWN,
            }:
                raise ValueError("unavailable failure category is inconsistent")
            if self.failure_category != self.failure.category:
                raise ValueError("failure category is inconsistent with failure metadata")
        elif self.outcome in {
            "transport_failed",
            "contract_failed",
            "capability_failed",
            "schema_mismatch",
        }:
            if self.failure is None:
                raise ValueError("provider failures require safe failure metadata")
            if self.failure_category != self.failure.category:
                raise ValueError("failure category is inconsistent with failure metadata")
            if self.completed_case_count >= self.case_count:
                raise ValueError("provider failures require an incomplete case")
            if (
                self.outcome == "contract_failed"
                and self.failure.category != FailureCategory.PROVIDER
            ):
                raise ValueError("contract failure category is inconsistent")
            if (
                self.outcome == "transport_failed"
                and self.failure.category not in _TRANSPORT_REASON_CODES
            ):
                raise ValueError("transport failure category is inconsistent")
            if self.outcome in {"capability_failed", "schema_mismatch"}:
                if (
                    self.failure.category != FailureCategory.CONTRACT
                    or self.capability is None
                ):
                    raise ValueError(
                        "capability failures require contract metadata"
                    )
            if (
                self.outcome == "capability_failed"
                or (
                    self.outcome == "schema_mismatch"
                    and self.schema_compatibility.empirical == "failed"
                )
            ) and self.completed_case_count != 0:
                raise ValueError("capability failures cannot complete golden cases")
            if self.outcome == "capability_failed" and self.capability.status not in {
                "unsupported",
                "unknown",
            }:
                raise ValueError("capability failure evidence is inconsistent")
            if (
                self.outcome == "schema_mismatch"
                and self.capability.status != "schema_mismatch"
                and self.schema_compatibility.empirical
                not in {"failed", "verified"}
            ):
                raise ValueError("schema mismatch evidence is inconsistent")

        if self.outcome in {"passed", "quality_failed"}:
            if self.execution_status != "executed":
                raise ValueError("quality outcomes require executed status")
            if self.completed_case_count != self.case_count:
                raise ValueError("quality outcomes require every case to complete")
            if self.quality_report is None:
                raise ValueError("quality outcomes require a quality report")
            if self.failure is not None:
                raise ValueError("quality outcomes cannot contain provider failure metadata")
            if self.failure_category is not None:
                raise ValueError("quality outcomes cannot contain a failure category")
            if self.quality_report.total_cases != self.case_count:
                raise ValueError("quality report total_cases must match case_count")
            expected_passed = self.outcome == "passed"
            if self.quality_report.overall_passed != expected_passed:
                raise ValueError("quality outcome must match the quality report")
            if (
                self.schema_compatibility.empirical != "verified"
                or not self.schema_compatibility.golden_contract_valid
            ):
                raise ValueError(
                    "quality outcomes require verified schema compatibility"
                )
        elif self.quality_report is not None:
            raise ValueError("non-quality outcomes cannot contain quality scores")
        if (
            self.outcome not in {"passed", "quality_failed"}
            and self.schema_compatibility.golden_contract_valid
        ):
            raise ValueError(
                "incomplete execution cannot claim golden contract validity"
            )
        return self


@dataclass(frozen=True)
class ProviderExecutionPolicy:
    required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")

    @classmethod
    def from_settings(
        cls, settings: "RealLLMProviderSettings"
    ) -> "ProviderExecutionPolicy":
        settings.validate()
        return cls(required=settings.required)

    def permits(self, outcome: ProviderConformanceOutcome) -> bool:
        if outcome == "passed":
            return True
        if outcome in {"not_configured", "unavailable"}:
            return not self.required
        return False


class QualityEvaluator(Protocol):
    def evaluate(
        self,
        cases: list[GoldenCase],
        responses: list[AirlineAssistantResponse],
    ) -> QualityReport: ...


class ProviderConformanceRunner:
    def __init__(
        self,
        *,
        provider: LLMProvider | None,
        provider_label: str,
        model_label: str,
        policy: ProviderExecutionPolicy | None = None,
        evaluator: QualityEvaluator | None = None,
        tracer: TracingFacade | None = None,
    ) -> None:
        self.provider = provider
        self.provider_label = safe_provider_label(provider_label)
        self.model_label = safe_model_label(model_label)
        configured_policy = (
            ProviderExecutionPolicy.from_settings(provider.settings)
            if isinstance(provider, OptionalRealLLMProvider)
            else ProviderExecutionPolicy()
        )
        if (
            policy is not None
            and isinstance(provider, OptionalRealLLMProvider)
            and policy != configured_policy
        ):
            raise ValueError("injected policy conflicts with provider configuration")
        self.policy = policy or configured_policy
        self.evaluator = evaluator or QualityGateEvaluator()
        self.tracer = tracer or create_tracing_facade()

    def run(
        self, cases: list[GoldenCase], *, prompt: Prompt
    ) -> ProviderConformanceReport:
        safe_case_ids = [safe_case_id(case.id) for case in cases]
        attributes = {
            "provider_name": self.provider_label.value,
            "model_name": self.model_label.value,
            "operation": "provider_conformance",
            "evaluation_status": "not_executed",
        }
        with self.tracer.start_trace(
            "provider.conformance",
            operation_type="evaluation",
            attributes=attributes,
        ) as span:
            report = self._run(cases, safe_case_ids=safe_case_ids, prompt=prompt)
            span.set_attribute("evaluation_status", report.execution_status)
            if report.capability is not None:
                span.set_attribute("capability_status", report.capability.status)
                span.set_attribute(
                    "response_schema_id", report.capability.response_schema_id
                )
                span.set_attribute(
                    "response_schema_version",
                    report.capability.response_schema_version,
                )
            return report

    def _run(
        self,
        cases: list[GoldenCase],
        *,
        safe_case_ids: list[_SafeIdentifier],
        prompt: Prompt,
    ) -> ProviderConformanceReport:
        if self.provider is None:
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status=(
                    "provider_required" if self.policy.required else "not_executed"
                ),
                outcome="not_configured",
                reason_code="provider_not_configured",
            )

        try:
            available = self.provider.health_check()
        except Exception:
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status="unavailable",
                outcome="unavailable",
                reason_code="provider_unavailable",
                failure=_provider_failure(),
            )
        if not available:
            if isinstance(self.provider, OptionalRealLLMProvider):
                return self._report(
                    safe_case_ids=safe_case_ids,
                    completed=0,
                    execution_status=(
                        "provider_required"
                        if self.policy.required
                        else "not_executed"
                    ),
                    outcome="not_configured",
                    reason_code="provider_not_configured",
                )
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status="unavailable",
                outcome="unavailable",
                reason_code="provider_unavailable",
                failure=_unavailable_failure(),
            )

        requirement = ProviderCapabilityRequirement()
        declared_capability = assess_provider_capabilities(
            self.provider, requirement
        )
        schema_compatibility = schema_compatibility_for_declaration(
            self.provider, declared_capability
        )
        if declared_capability.status == "unsupported":
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status="failed",
                outcome="capability_failed",
                reason_code="provider_capability_unsupported",
                failure=_capability_failure(),
                capability=declared_capability,
                schema_compatibility=schema_compatibility,
            )
        if (
            declared_capability.status == "unknown"
            and declared_capability.reason_code == "provider_capability_invalid"
        ):
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status="failed",
                outcome="capability_failed",
                reason_code="provider_capability_invalid",
                failure=_capability_failure(),
                capability=declared_capability,
                schema_compatibility=schema_compatibility,
            )
        if declared_capability.status == "schema_mismatch":
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status="failed",
                outcome="schema_mismatch",
                reason_code="provider_schema_mismatch",
                failure=_capability_failure(),
                capability=declared_capability,
                schema_compatibility=schema_compatibility,
            )

        try:
            preflight_response = self.provider.generate_structured(
                _PREFLIGHT_USER_INPUT,
                prompt=_PREFLIGHT_PROMPT,
                case=None,
                context=None,
            )
        except RealProviderDisabledError:
            return self._report(
                safe_case_ids=safe_case_ids,
                completed=0,
                execution_status=(
                    "provider_required" if self.policy.required else "not_executed"
                ),
                outcome="not_configured",
                reason_code="provider_not_configured",
                capability=declared_capability,
                schema_compatibility=schema_compatibility,
            )
        except RealProviderTransportError as exc:
            return self._failure_report(
                safe_case_ids,
                0,
                "transport_failed",
                exc.failure,
                capability=declared_capability,
                schema_compatibility=with_empirical_compatibility(
                    schema_compatibility, "failed"
                ),
            )
        except RealProviderResponseError as exc:
            if exc.failure and exc.failure.category == FailureCategory.CONTRACT:
                return self._failure_report(
                    safe_case_ids,
                    0,
                    "schema_mismatch",
                    exc.failure,
                    capability=declared_capability,
                    schema_compatibility=with_empirical_compatibility(
                        schema_compatibility, "failed"
                    ),
                )
            return self._failure_report(
                safe_case_ids,
                0,
                "contract_failed",
                exc.failure or _provider_failure(),
                capability=declared_capability,
                schema_compatibility=with_empirical_compatibility(
                    schema_compatibility, "failed"
                ),
            )
        except RealProviderError as exc:
            failure = exc.failure or _provider_failure()
            outcome: Literal[
                "transport_failed", "contract_failed", "schema_mismatch"
            ] = (
                "schema_mismatch"
                if failure.category == FailureCategory.CONTRACT
                else "transport_failed"
            )
            return self._failure_report(
                safe_case_ids,
                0,
                outcome,
                failure,
                capability=declared_capability,
                schema_compatibility=with_empirical_compatibility(
                    schema_compatibility, "failed"
                ),
            )
        except Exception:
            return self._failure_report(
                safe_case_ids,
                0,
                "transport_failed",
                _provider_failure(),
                capability=declared_capability,
                schema_compatibility=with_empirical_compatibility(
                    schema_compatibility, "failed"
                ),
            )

        if _validate_structured_response(preflight_response) is None:
            return self._failure_report(
                safe_case_ids,
                0,
                "schema_mismatch",
                _capability_failure(),
                capability=declared_capability,
                schema_compatibility=with_empirical_compatibility(
                    schema_compatibility, "failed"
                ),
            )

        schema_compatibility = with_empirical_compatibility(
            schema_compatibility, "verified"
        )

        responses: list[AirlineAssistantResponse] = []
        for case in cases:
            try:
                runtime_response = self.provider.generate_structured(
                    case.user_input,
                    prompt=prompt,
                    case=case,
                    context=case.context,
                )
            except RealProviderDisabledError:
                return self._report(
                    safe_case_ids=safe_case_ids,
                    completed=len(responses),
                    execution_status=(
                        "provider_required" if self.policy.required else "not_executed"
                    ),
                    outcome="not_configured",
                    reason_code="provider_not_configured",
                    capability=declared_capability,
                    schema_compatibility=schema_compatibility,
                )
            except RealProviderTransportError as exc:
                return self._failure_report(
                    safe_case_ids,
                    len(responses),
                    "transport_failed",
                    exc.failure,
                    capability=declared_capability,
                    schema_compatibility=schema_compatibility,
                )
            except RealProviderResponseError as exc:
                if exc.failure and exc.failure.category == FailureCategory.CONTRACT:
                    return self._failure_report(
                        safe_case_ids,
                        len(responses),
                        "schema_mismatch",
                        exc.failure,
                        capability=declared_capability,
                        schema_compatibility=schema_compatibility,
                    )
                return self._failure_report(
                    safe_case_ids,
                    len(responses),
                    "contract_failed",
                    exc.failure,
                    capability=declared_capability,
                    schema_compatibility=schema_compatibility,
                )
            except RealProviderError as exc:
                failure = exc.failure or _provider_failure()
                outcome: ProviderConformanceOutcome = (
                    "schema_mismatch"
                    if failure.category == FailureCategory.CONTRACT
                    else "transport_failed"
                )
                return self._failure_report(
                    safe_case_ids,
                    len(responses),
                    outcome,
                    failure,
                    capability=declared_capability,
                    schema_compatibility=schema_compatibility,
                )
            except Exception:
                return self._failure_report(
                    safe_case_ids,
                    len(responses),
                    "transport_failed",
                    _provider_failure(),
                    capability=declared_capability,
                    schema_compatibility=schema_compatibility,
                )
            response = _validate_structured_response(runtime_response)
            if response is None:
                return self._failure_report(
                    safe_case_ids,
                    len(responses),
                    "schema_mismatch",
                    _capability_failure(),
                    capability=declared_capability,
                    schema_compatibility=schema_compatibility,
                )
            responses.append(response)

        schema_compatibility = with_empirical_compatibility(
            schema_compatibility,
            "verified",
            golden_contract_valid=True,
        )
        quality_report = self.evaluator.evaluate(cases, responses)
        outcome: ProviderConformanceOutcome = (
            "passed" if quality_report.overall_passed else "quality_failed"
        )
        return self._report(
            safe_case_ids=safe_case_ids,
            completed=len(responses),
            execution_status="executed",
            outcome=outcome,
            reason_code=None if outcome == "passed" else "quality_gate_failed",
            quality_report=quality_report,
            capability=declared_capability,
            schema_compatibility=schema_compatibility,
        )

    def _failure_report(
        self,
        safe_case_ids: list[_SafeIdentifier],
        completed: int,
        outcome: Literal[
            "transport_failed", "contract_failed", "schema_mismatch"
        ],
        failure: ProviderFailureMetadata,
        *,
        capability: ProviderCapabilityEvidence | None = None,
        schema_compatibility: SchemaCompatibilityEvidence | None = None,
    ) -> ProviderConformanceReport:
        return self._report(
            safe_case_ids=safe_case_ids,
            completed=completed,
            execution_status="failed",
            outcome=outcome,
            reason_code=_failure_reason_code(outcome, failure.category),
            failure=failure,
            capability=capability,
            schema_compatibility=schema_compatibility,
        )

    def _report(
        self,
        *,
        safe_case_ids: list[_SafeIdentifier],
        completed: int,
        execution_status: ExecutionStatus,
        outcome: ProviderConformanceOutcome,
        reason_code: ReasonCode | None,
        failure: ProviderFailureMetadata | None = None,
        capability: ProviderCapabilityEvidence | None = None,
        schema_compatibility: SchemaCompatibilityEvidence | None = None,
        quality_report: QualityReport | None = None,
    ) -> ProviderConformanceReport:
        return ProviderConformanceReport(
            provider=self.provider_label,
            model=self.model_label,
            required=self.policy.required,
            execution_status=execution_status,
            outcome=outcome,
            failure_category=failure.category if failure is not None else None,
            reason_code=reason_code,
            case_count=len(safe_case_ids),
            completed_case_count=completed,
            case_ids=safe_case_ids,
            failure=(
                ProviderFailureEvidence(
                    category=failure.category,
                    retryable=failure.retryable,
                    attempt=failure.attempt,
                    max_attempts=failure.max_attempts,
                    status_code=failure.status_code,
                )
                if failure is not None
                else None
            ),
            capability=capability,
            schema_compatibility=(
                schema_compatibility
                or schema_compatibility_for_declaration(self.provider, capability)
            ),
            quality_report=quality_report,
            policy_passed=self.policy.permits(outcome),
        )


def adapt_provider_report_to_phase10(
    phase10_report: Phase10Report,
    provider_report: ProviderConformanceReport,
) -> Phase10Report:
    baseline = dict(phase10_report.baseline)
    baseline["provider_conformance"] = provider_report.model_dump(mode="json")
    return phase10_report.model_copy(
        update={
            "baseline": baseline,
            "overall_passed": bool(
                phase10_report.overall_passed and provider_report.policy_passed
            ),
        }
    )


def serialize_provider_report(report: ProviderConformanceReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_provider_phase10_report(path: Path, report: Phase10Report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def conformance_exit_code(report: ProviderConformanceReport) -> int:
    return 0 if report.policy_passed else 1


def run_provider_conformance(
    *,
    runner: ProviderConformanceRunner,
    prompt: Prompt,
    phase10_report: Phase10Report,
    output_path: Path,
    cases: list[GoldenCase] | None = None,
    golden_cases_path: Path | None = None,
    case_loader: Callable[[Path], list[GoldenCase]] = load_golden_cases,
) -> int:
    if cases is not None and golden_cases_path is not None:
        raise ValueError("cases and golden_cases_path are mutually exclusive")
    resolved_cases = (
        cases
        if cases is not None
        else case_loader(golden_cases_path or DATASET_PATH)
    )
    provider_report = runner.run(resolved_cases, prompt=prompt)
    combined_report = adapt_provider_report_to_phase10(
        phase10_report, provider_report
    )
    write_provider_phase10_report(output_path, combined_report)
    return conformance_exit_code(provider_report)


def _provider_failure() -> ProviderFailureMetadata:
    return ProviderFailureMetadata(
        category=FailureCategory.PROVIDER,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )


def _validate_structured_response(value: object) -> AirlineAssistantResponse | None:
    try:
        return AirlineAssistantResponse.model_validate(value)
    except (TypeError, ValueError):
        return None


def _unavailable_failure() -> ProviderFailureMetadata:
    return ProviderFailureMetadata(
        category=FailureCategory.PROVIDER,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )


def _capability_failure() -> ProviderFailureMetadata:
    return ProviderFailureMetadata(
        category=FailureCategory.CONTRACT,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )


def _failure_reason_code(
    outcome: ProviderConformanceOutcome,
    category: FailureCategory | None,
) -> ReasonCode | None:
    if outcome == "schema_mismatch":
        return "provider_schema_mismatch"
    if outcome == "capability_failed":
        return None
    if outcome == "contract_failed":
        return (
            "provider_response_malformed"
            if category == FailureCategory.PROVIDER
            else None
        )
    if outcome != "transport_failed":
        return None
    return _TRANSPORT_REASON_CODES.get(category)


_TRANSPORT_REASON_CODES: dict[FailureCategory, ReasonCode] = {
    FailureCategory.AUTHENTICATION: "provider_authentication_failed",
    FailureCategory.AUTHORIZATION: "provider_authorization_failed",
    FailureCategory.TIMEOUT: "provider_timeout",
    FailureCategory.NETWORK: "provider_network_failed",
    FailureCategory.RATE_LIMIT: "provider_rate_limited",
    FailureCategory.PROVIDER: "provider_failed",
}
