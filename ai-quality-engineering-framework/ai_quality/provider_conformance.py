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
    "quality_failed",
    "passed",
]

ReasonCode = Literal[
    "provider_not_configured",
    "provider_unavailable",
    "provider_transport_failure",
    "provider_contract_failure",
    "provider_quality_failure",
    "provider_conformance_passed",
]

class ProviderFailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    provider: str
    model: str
    required: bool = Field(strict=True)
    execution_status: ExecutionStatus
    outcome: ProviderConformanceOutcome
    reason_code: ReasonCode
    case_count: int = Field(ge=0, strict=True)
    completed_case_count: int = Field(ge=0, strict=True)
    case_ids: tuple[str, ...] = Field(default_factory=tuple)
    failure: ProviderFailureEvidence | None = None
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
            "quality_failed": "executed",
            "passed": "executed",
        }
        expected_reason: dict[ProviderConformanceOutcome, ReasonCode] = {
            "not_configured": "provider_not_configured",
            "unavailable": "provider_unavailable",
            "transport_failed": "provider_transport_failure",
            "contract_failed": "provider_contract_failure",
            "quality_failed": "provider_quality_failure",
            "passed": "provider_conformance_passed",
        }
        if self.execution_status != expected_status[self.outcome]:
            raise ValueError("execution_status is inconsistent with outcome")
        if self.reason_code != expected_reason[self.outcome]:
            raise ValueError("reason_code is inconsistent with outcome")
        expected_policy = ProviderExecutionPolicy(required=self.required).permits(
            self.outcome
        )
        if self.policy_passed != expected_policy:
            raise ValueError("policy_passed is inconsistent with outcome")

        if self.outcome == "not_configured":
            if self.completed_case_count != 0 or self.failure is not None:
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
        elif self.outcome in {"transport_failed", "contract_failed"}:
            if self.failure is None:
                raise ValueError("provider failures require safe failure metadata")
            if self.completed_case_count >= self.case_count:
                raise ValueError("provider failures require an incomplete case")
            if (
                self.outcome == "contract_failed"
                and self.failure.category
                not in {FailureCategory.CONTRACT, FailureCategory.PROVIDER}
            ):
                raise ValueError("contract failure category is inconsistent")
            if (
                self.outcome == "transport_failed"
                and self.failure.category == FailureCategory.CONTRACT
            ):
                raise ValueError("transport failure category is inconsistent")

        if self.outcome in {"passed", "quality_failed"}:
            if self.execution_status != "executed":
                raise ValueError("quality outcomes require executed status")
            if self.completed_case_count != self.case_count:
                raise ValueError("quality outcomes require every case to complete")
            if self.quality_report is None:
                raise ValueError("quality outcomes require a quality report")
            if self.failure is not None:
                raise ValueError("quality outcomes cannot contain provider failure metadata")
            if self.quality_report.total_cases != self.case_count:
                raise ValueError("quality report total_cases must match case_count")
            expected_passed = self.outcome == "passed"
            if self.quality_report.overall_passed != expected_passed:
                raise ValueError("quality outcome must match the quality report")
        elif self.quality_report is not None:
            raise ValueError("non-quality outcomes cannot contain quality scores")
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
                failure=_unknown_failure(),
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

        responses: list[AirlineAssistantResponse] = []
        for case in cases:
            try:
                response = self.provider.generate_structured(
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
                )
            except RealProviderTransportError as exc:
                return self._failure_report(
                    safe_case_ids, len(responses), "transport_failed", exc.failure
                )
            except RealProviderResponseError as exc:
                return self._failure_report(
                    safe_case_ids, len(responses), "contract_failed", exc.failure
                )
            except RealProviderError as exc:
                failure = exc.failure or _unknown_failure()
                outcome: ProviderConformanceOutcome = (
                    "contract_failed"
                    if failure.category == FailureCategory.CONTRACT
                    else "transport_failed"
                )
                return self._failure_report(
                    safe_case_ids, len(responses), outcome, failure
                )
            except Exception:
                return self._failure_report(
                    safe_case_ids,
                    len(responses),
                    "transport_failed",
                    _unknown_failure(),
                )
            responses.append(response)

        quality_report = self.evaluator.evaluate(cases, responses)
        outcome: ProviderConformanceOutcome = (
            "passed" if quality_report.overall_passed else "quality_failed"
        )
        return self._report(
            safe_case_ids=safe_case_ids,
            completed=len(responses),
            execution_status="executed",
            outcome=outcome,
            reason_code=(
                "provider_conformance_passed"
                if outcome == "passed"
                else "provider_quality_failure"
            ),
            quality_report=quality_report,
        )

    def _failure_report(
        self,
        safe_case_ids: list[_SafeIdentifier],
        completed: int,
        outcome: Literal["transport_failed", "contract_failed"],
        failure: ProviderFailureMetadata,
    ) -> ProviderConformanceReport:
        return self._report(
            safe_case_ids=safe_case_ids,
            completed=completed,
            execution_status="failed",
            outcome=outcome,
            reason_code=(
                "provider_contract_failure"
                if outcome == "contract_failed"
                else "provider_transport_failure"
            ),
            failure=failure,
        )

    def _report(
        self,
        *,
        safe_case_ids: list[_SafeIdentifier],
        completed: int,
        execution_status: ExecutionStatus,
        outcome: ProviderConformanceOutcome,
        reason_code: ReasonCode,
        failure: ProviderFailureMetadata | None = None,
        quality_report: QualityReport | None = None,
    ) -> ProviderConformanceReport:
        return ProviderConformanceReport(
            provider=self.provider_label,
            model=self.model_label,
            required=self.policy.required,
            execution_status=execution_status,
            outcome=outcome,
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


def _unknown_failure() -> ProviderFailureMetadata:
    return ProviderFailureMetadata(
        category=FailureCategory.UNKNOWN,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )


def _unavailable_failure() -> ProviderFailureMetadata:
    return ProviderFailureMetadata(
        category=FailureCategory.PROVIDER,
        retryable=False,
        attempt=1,
        max_attempts=1,
    )
