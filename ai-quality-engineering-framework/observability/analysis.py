from __future__ import annotations

from collections import Counter
import re

from pydantic import BaseModel, ConfigDict, Field

from ai_eval.models import Phase10Report
from observability.ci import CiCorrelationReport
from observability.models import SpanEvidence, TraceEnvelope, TraceStatus
from observability.redaction import REDACTED, sanitize_attributes


SCHEMA_VERSION = "1.0"
_SAFE_DIMENSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/#-]{0,199}")


class LatencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    sample_count: int = Field(ge=0)
    minimum_ms: float | None = Field(default=None, ge=0)
    average_ms: float | None = Field(default=None, ge=0)
    maximum_ms: float | None = Field(default=None, ge=0)


class SlowOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    operation_name: str
    operation_type: str
    duration_ms: float = Field(ge=0)


class ParentIntegrityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    parent_span_id: str


class TraceIntegritySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duplicate_trace_ids: list[str] = Field(default_factory=list)
    duplicate_span_ids: list[str] = Field(default_factory=list)
    orphan_parents: list[ParentIntegrityIssue] = Field(default_factory=list)
    self_parented_span_ids: list[str] = Field(default_factory=list)
    cyclic_span_ids: list[str] = Field(default_factory=list)
    traces_without_roots: list[str] = Field(default_factory=list)
    traces_with_multiple_roots: list[str] = Field(default_factory=list)


class RagLineageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    rag_trace_count: int | None = Field(default=None, ge=0)
    complete_trace_count: int | None = Field(default=None, ge=0)
    incomplete_trace_ids: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class CorrelationCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    test_count: int | None = Field(default=None, ge=0)
    tests_with_traces: int | None = Field(default=None, ge=0)
    tests_without_traces: int | None = Field(default=None, ge=0)
    referenced_trace_count: int | None = Field(default=None, ge=0)
    trace_ids_missing_from_evidence: list[str] = Field(default_factory=list)
    trace_ids_without_tests: list[str] = Field(default_factory=list)
    linked_case_ids: list[str] = Field(default_factory=list)


class EvaluationCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    overall_passed: bool | None = None
    framework_statuses: dict[str, str] = Field(default_factory=dict)
    evaluated_case_ids: list[str] = Field(default_factory=list)
    failed_case_ids: list[str] = Field(default_factory=list)
    case_ids_without_trace_lineage: list[str] = Field(default_factory=list)


class OperationalAnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    trace_count: int = Field(ge=0)
    span_count: int = Field(ge=0)
    failed_span_count: int = Field(ge=0)
    failure_categories: dict[str, int] = Field(default_factory=dict)
    operation_counts: dict[str, int] = Field(default_factory=dict)
    operation_type_counts: dict[str, int] = Field(default_factory=dict)
    provider_counts: dict[str, int] = Field(default_factory=dict)
    model_counts: dict[str, int] = Field(default_factory=dict)
    evaluation_status_counts: dict[str, int] = Field(default_factory=dict)
    token_count_total: int | None = Field(default=None, ge=0)
    latency: LatencySummary
    slowest_operations: list[SlowOperation] = Field(default_factory=list)
    integrity: TraceIntegritySummary
    rag_lineage: RagLineageSummary
    correlation: CorrelationCoverage
    evaluation: EvaluationCoverage
    warnings: list[str] = Field(default_factory=list)


def analyze_operational_evidence(
    traces: list[TraceEnvelope],
    *,
    ci_report: CiCorrelationReport | None = None,
    phase10_report: Phase10Report | None = None,
    slow_operation_limit: int = 10,
) -> OperationalAnalysisReport:
    if slow_operation_limit < 0:
        raise ValueError("slow_operation_limit must not be negative")

    spans = [span for envelope in traces for span in envelope.spans]
    trace_ids = {envelope.trace_id for envelope in traces}
    attributes = {id(span): sanitize_attributes(span.attributes) for span in spans}

    failures = [span for span in spans if span.status == TraceStatus.ERROR]
    failure_categories = Counter(
        span.failure_category.value if span.failure_category else "unknown"
        for span in failures
    )
    operation_counts = Counter(_safe_dimension(span.operation_name) for span in spans)
    operation_type_counts = Counter(_safe_dimension(span.operation_type) for span in spans)
    provider_counts = _attribute_counter(spans, attributes, "provider_name")
    model_counts = _attribute_counter(spans, attributes, "model_name")
    evaluation_statuses = _attribute_counter(
        spans, attributes, "evaluation_status"
    )
    token_values = [
        value
        for span in spans
        if isinstance((value := attributes[id(span)].get("token_count")), int)
        and not isinstance(value, bool)
        and value >= 0
    ]

    durations = [span.duration_ms for span in spans]
    latency = LatencySummary(
        available=bool(durations),
        sample_count=len(durations),
        minimum_ms=min(durations) if durations else None,
        average_ms=(sum(durations) / len(durations)) if durations else None,
        maximum_ms=max(durations) if durations else None,
    )
    slowest = sorted(
        (
            SlowOperation(
                trace_id=span.trace_id,
                span_id=span.span_id,
                operation_name=_safe_dimension(span.operation_name),
                operation_type=_safe_dimension(span.operation_type),
                duration_ms=span.duration_ms,
            )
            for span in spans
        ),
        key=lambda item: (-item.duration_ms, item.trace_id, item.span_id),
    )[:slow_operation_limit]

    integrity = _analyze_integrity(traces)
    lineage = _analyze_rag_lineage(traces, attributes)
    correlation = _analyze_correlation(ci_report, trace_ids)
    evidence_case_ids = {
        value
        for span in spans
        for value in _attribute_strings(attributes[id(span)].get("case_id"))
    }
    evaluation = _analyze_evaluation(phase10_report, evidence_case_ids)

    warnings: list[str] = []
    if not traces:
        warnings.append("trace_evidence_unavailable")
    if ci_report is None:
        warnings.append("ci_correlation_unavailable")
    if phase10_report is None:
        warnings.append("phase10_evidence_unavailable")
    if integrity.duplicate_span_ids:
        warnings.append("duplicate_spans_detected")
    if integrity.orphan_parents:
        warnings.append("orphan_parents_detected")
    if integrity.cyclic_span_ids or integrity.self_parented_span_ids:
        warnings.append("cyclic_parentage_detected")
    if lineage.available and lineage.incomplete_trace_ids:
        warnings.append("incomplete_rag_lineage")
    if correlation.available and correlation.trace_ids_missing_from_evidence:
        warnings.append("correlation_references_missing_traces")

    return OperationalAnalysisReport(
        trace_count=len(trace_ids),
        span_count=len(spans),
        failed_span_count=len(failures),
        failure_categories=_sorted_counter(failure_categories),
        operation_counts=_sorted_counter(operation_counts),
        operation_type_counts=_sorted_counter(operation_type_counts),
        provider_counts=_sorted_counter(provider_counts),
        model_counts=_sorted_counter(model_counts),
        evaluation_status_counts=_sorted_counter(evaluation_statuses),
        token_count_total=sum(token_values) if token_values else None,
        latency=latency,
        slowest_operations=slowest,
        integrity=integrity,
        rag_lineage=lineage,
        correlation=correlation,
        evaluation=evaluation,
        warnings=sorted(warnings),
    )


def _analyze_integrity(traces: list[TraceEnvelope]) -> TraceIntegritySummary:
    trace_counts = Counter(item.trace_id for item in traces)
    span_counts = Counter(span.span_id for item in traces for span in item.spans)
    orphaned: list[ParentIntegrityIssue] = []
    self_parented: set[str] = set()
    cyclic: set[str] = set()
    without_roots: set[str] = set()
    multiple_roots: set[str] = set()

    for envelope in traces:
        span_by_id = {span.span_id: span for span in envelope.spans}
        roots = [span for span in envelope.spans if span.parent_span_id is None]
        if envelope.spans and not roots:
            without_roots.add(envelope.trace_id)
        if len(roots) > 1:
            multiple_roots.add(envelope.trace_id)
        for span in envelope.spans:
            parent_id = span.parent_span_id
            if parent_id is None:
                continue
            if parent_id == span.span_id:
                self_parented.add(span.span_id)
            elif parent_id not in span_by_id:
                orphaned.append(
                    ParentIntegrityIssue(
                        trace_id=envelope.trace_id,
                        span_id=span.span_id,
                        parent_span_id=parent_id,
                    )
                )
        cyclic.update(_cyclic_span_ids(span_by_id))

    return TraceIntegritySummary(
        duplicate_trace_ids=sorted(key for key, count in trace_counts.items() if count > 1),
        duplicate_span_ids=sorted(key for key, count in span_counts.items() if count > 1),
        orphan_parents=sorted(
            orphaned, key=lambda item: (item.trace_id, item.span_id, item.parent_span_id)
        ),
        self_parented_span_ids=sorted(self_parented),
        cyclic_span_ids=sorted(cyclic),
        traces_without_roots=sorted(without_roots),
        traces_with_multiple_roots=sorted(multiple_roots),
    )


def _cyclic_span_ids(span_by_id: dict[str, SpanEvidence]) -> set[str]:
    cyclic: set[str] = set()
    complete: set[str] = set()
    for start in span_by_id:
        path: list[str] = []
        positions: dict[str, int] = {}
        current: str | None = start
        while current is not None and current in span_by_id and current not in complete:
            if current in positions:
                cyclic.update(path[positions[current] :])
                break
            positions[current] = len(path)
            path.append(current)
            current = span_by_id[current].parent_span_id
        complete.update(path)
    return cyclic


def _analyze_rag_lineage(
    traces: list[TraceEnvelope], attributes: dict[int, dict]
) -> RagLineageSummary:
    rag_traces: list[TraceEnvelope] = []
    cases: set[str] = set()
    documents: set[str] = set()
    chunks: set[str] = set()
    incomplete: list[str] = []
    for envelope in traces:
        rag_spans = [
            span
            for span in envelope.spans
            if span.operation_type in {"rag", "retrieval"}
        ]
        if not rag_spans:
            continue
        rag_traces.append(envelope)
        trace_cases = _values_for_spans(rag_spans, attributes, "case_id")
        trace_documents = _values_for_spans(rag_spans, attributes, "document_id")
        trace_chunks = _values_for_spans(rag_spans, attributes, "chunk_id")
        cases.update(trace_cases)
        documents.update(trace_documents)
        chunks.update(trace_chunks)
        if not (trace_cases and trace_documents and trace_chunks):
            incomplete.append(envelope.trace_id)
    if not rag_traces:
        return RagLineageSummary(available=False)
    return RagLineageSummary(
        available=True,
        rag_trace_count=len(rag_traces),
        complete_trace_count=len(rag_traces) - len(incomplete),
        incomplete_trace_ids=sorted(incomplete),
        case_ids=sorted(cases),
        document_ids=sorted(documents),
        chunk_ids=sorted(chunks),
    )


def _analyze_correlation(
    report: CiCorrelationReport | None, evidence_trace_ids: set[str]
) -> CorrelationCoverage:
    if report is None:
        return CorrelationCoverage(available=False)
    referenced = {trace_id for item in report.tests for trace_id in item.trace_ids}
    linked_cases = {
        _safe_dimension(case_id) for item in report.tests for case_id in item.case_ids
    }
    return CorrelationCoverage(
        available=True,
        test_count=len(report.tests),
        tests_with_traces=sum(bool(item.trace_ids) for item in report.tests),
        tests_without_traces=sum(not item.trace_ids for item in report.tests),
        referenced_trace_count=len(referenced),
        trace_ids_missing_from_evidence=sorted(referenced - evidence_trace_ids),
        trace_ids_without_tests=sorted(evidence_trace_ids - referenced),
        linked_case_ids=sorted(linked_cases),
    )


def _analyze_evaluation(
    report: Phase10Report | None, evidence_case_ids: set[str]
) -> EvaluationCoverage:
    if report is None:
        return EvaluationCoverage(available=False)
    results = [result for framework in report.frameworks for result in framework.results]
    evaluated = {_safe_dimension(result.case_id) for result in results}
    failed = {
        _safe_dimension(result.case_id)
        for result in results
        if result.passed is False
    }
    return EvaluationCoverage(
        available=True,
        overall_passed=report.overall_passed,
        framework_statuses=dict(
            sorted(
                (_safe_dimension(item.framework), _safe_dimension(item.status))
                for item in report.frameworks
            )
        ),
        evaluated_case_ids=sorted(evaluated),
        failed_case_ids=sorted(failed),
        case_ids_without_trace_lineage=sorted(evaluated - evidence_case_ids),
    )


def _attribute_counter(
    spans: list[SpanEvidence], attributes: dict[int, dict], key: str
) -> Counter:
    counter: Counter = Counter()
    for span in spans:
        counter.update(_attribute_strings(attributes[id(span)].get(key)))
    return counter


def _values_for_spans(
    spans: list[SpanEvidence], attributes: dict[int, dict], key: str
) -> set[str]:
    return {
        value
        for span in spans
        for value in _attribute_strings(attributes[id(span)].get(key))
    }


def _attribute_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [_safe_dimension(value)]
    if isinstance(value, (list, tuple)):
        return [_safe_dimension(item) for item in value if isinstance(item, str)]
    return []


def _safe_dimension(value: str) -> str:
    return value if _SAFE_DIMENSION.fullmatch(value) else REDACTED


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}
