from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_eval.models import ExecutionStatus
    from ai_quality.execution_descriptor import TrustedExecutionDescriptor
    from ai_quality.provider_conformance import (
        ProviderConformanceOutcome,
        ProviderConformanceReport,
    )
    from ai_quality.provider_provenance import ProviderConformanceProvenance


class TrustedProviderConformanceResult:
    __slots__ = (
        "__descriptor",
        "__execution_validator",
        "__phase10_consumed",
        "__report",
        "__weakref__",
    )

    def __new__(cls, *args, **kwargs):
        raise TypeError("trusted provider results can only be minted by the runner")

    @property
    def descriptor(self) -> TrustedExecutionDescriptor:
        return self.__descriptor

    @property
    def report(self) -> ProviderConformanceReport:
        return self.__report

    @property
    def provenance(self) -> ProviderConformanceProvenance:
        provenance = self.__report.provenance
        if provenance is None:  # Construction prevents this state.
            raise RuntimeError("trusted result is missing provenance")
        return provenance

    @property
    def execution_status(self) -> ExecutionStatus:
        return self.__report.execution_status

    @property
    def outcome(self) -> ProviderConformanceOutcome:
        return self.__report.outcome

    @property
    def execution_evidence_id(self) -> str:
        return self.provenance.execution_evidence_id

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("trusted provider results are immutable")

    def __repr__(self) -> str:
        return (
            "TrustedProviderConformanceResult("
            f"outcome={self.outcome!r}, "
            f"execution_status={self.execution_status!r}, "
            f"execution_evidence_id={self.execution_evidence_id!r})"
        )


def _is_issued_trusted_result(value: object) -> bool:
    try:
        if not isinstance(value, TrustedProviderConformanceResult):
            return False
        validator = value._TrustedProviderConformanceResult__execution_validator
        if not callable(validator) or not validator(value):
            return False
        from ai_quality.provider_provenance import (
            _build_trusted_provider_conformance_provenance,
        )

        expected = _build_trusted_provider_conformance_provenance(value.descriptor)
        return (
            value.report.provenance == expected
            and value.report.provenance.execution_evidence_id
            == expected.execution_evidence_id
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _public_provenance_from_trusted_result(
    value: TrustedProviderConformanceResult,
) -> ProviderConformanceProvenance:
    if not _is_issued_trusted_result(value):
        raise TypeError("public provenance requires a runner-issued trusted result")
    from ai_quality.provider_provenance import (
        _build_trusted_provider_conformance_provenance,
    )

    return _build_trusted_provider_conformance_provenance(
        value.descriptor
    ).model_copy(deep=True)


def _public_report_from_trusted_result(
    value: TrustedProviderConformanceResult,
) -> ProviderConformanceReport:
    if not _is_issued_trusted_result(value):
        raise TypeError("public report requires a runner-issued trusted result")
    provenance = _public_provenance_from_trusted_result(value)
    report = value._TrustedProviderConformanceResult__report
    return report.model_copy(
        deep=True,
        update={
            "provenance": provenance,
        },
    )


def _consume_public_report_for_phase10(
    value: TrustedProviderConformanceResult,
) -> ProviderConformanceReport:
    if not _is_issued_trusted_result(value):
        raise TypeError(
            "Phase 10 provider evidence requires a runner-issued trusted result"
        )
    if value._TrustedProviderConformanceResult__phase10_consumed:
        raise TypeError("trusted provider result has already been consumed by Phase 10")
    report = _public_report_from_trusted_result(value)
    object.__setattr__(
        value,
        "_TrustedProviderConformanceResult__phase10_consumed",
        True,
    )
    return report


def _report_fingerprint(report: ProviderConformanceReport) -> str:
    payload = json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
