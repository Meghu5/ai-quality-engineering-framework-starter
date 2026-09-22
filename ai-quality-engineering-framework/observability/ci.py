from __future__ import annotations

import hashlib
import json
import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from observability.context import clear_context
from observability.models import FailureCategory, SpanEvidence, TraceEnvelope
from observability.redaction import REDACTED, redact_text


SCHEMA_VERSION = "1.0"
CI_REPORT_PATH = Path("reports") / "ci" / "test-trace-correlation.json"
FAILURE_REPORT_PATH = Path("reports") / "ci" / "failure-summary.json"
TRACE_REPORT_PATH = Path("reports") / "observability" / "traces.json"

TestOutcome = Literal["passed", "failed", "skipped", "xfailed", "xpassed"]
_SAFE_NODE_ID = re.compile(r"^[A-Za-z0-9_./\\:\[\]-]+$")
_SENSITIVE_NODE_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "password",
    "bearer",
    "cookie",
    "secret",
    "token",
)


class CiFailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    failure_category: FailureCategory


class CiTestEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    test_id: str
    node_id: str | None = None
    outcome: TestOutcome
    duration_ms: float = Field(ge=0)
    correlation_id: str
    trace_ids: list[str] = Field(default_factory=list)
    span_ids: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    failure_category: FailureCategory | None = None
    evaluation_statuses: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class CiCorrelationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    tests: list[CiTestEvidence] = Field(default_factory=list)
    failures: list[CiFailureEvidence] = Field(default_factory=list)
    summary: dict[str, int | dict[str, int]]


@dataclass
class TestCorrelationState:
    test_id: str
    node_id: str | None
    correlation_id: str
    spans: list[SpanEvidence] = field(default_factory=list)
    phase_outcomes: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0
    test_failure_category: FailureCategory | None = None


_current_test_state: ContextVar[TestCorrelationState | None] = ContextVar(
    "ci_test_correlation_state", default=None
)


def stable_test_id(node_id: str) -> str:
    return hashlib.sha256(node_id.encode("utf-8")).hexdigest()


def sanitize_node_id(node_id: str) -> str | None:
    candidate = node_id.strip()
    if not candidate:
        return None
    if "[" in candidate or "]" in candidate:
        prefix, separator, _ = candidate.partition("[")
        if not separator or not prefix or "]" not in candidate:
            return None
        candidate = f"{prefix}[redacted]"
    if len(candidate) > 240 or "?" in candidate or "=" in candidate:
        return None
    lowered = candidate.lower()
    if any(part in lowered for part in _SENSITIVE_NODE_PARTS):
        return None
    redacted = redact_text(candidate)
    if redacted != candidate or REDACTED in redacted:
        return None
    return candidate if _SAFE_NODE_ID.fullmatch(candidate) else None


def begin_test_correlation(node_id: str) -> tuple[TestCorrelationState, Token]:
    clear_context()
    test_id = stable_test_id(node_id)
    state = TestCorrelationState(
        test_id=test_id,
        node_id=sanitize_node_id(node_id),
        correlation_id=test_id,
    )
    token = _current_test_state.set(state)
    return state, token


def end_test_correlation(token: Token) -> None:
    clear_context()
    _current_test_state.reset(token)


def get_current_test_state() -> TestCorrelationState | None:
    return _current_test_state.get()


def get_current_test_correlation_id() -> str | None:
    state = get_current_test_state()
    return state.correlation_id if state is not None else None


def record_span_for_current_test(span: SpanEvidence) -> None:
    state = get_current_test_state()
    if state is not None and span.correlation_id == state.correlation_id:
        state.spans.append(span.model_copy(deep=True))


def classify_test_exception(exception_type: type[BaseException] | None) -> FailureCategory:
    if exception_type is None:
        return FailureCategory.UNKNOWN
    if issubclass(exception_type, AssertionError):
        return FailureCategory.CONTRACT
    if issubclass(exception_type, TimeoutError):
        return FailureCategory.TIMEOUT
    if exception_type.__name__ == "ValidationError" and exception_type.__module__.startswith(
        "pydantic"
    ):
        return FailureCategory.VALIDATION
    return FailureCategory.UNKNOWN


def build_test_evidence(state: TestCorrelationState) -> CiTestEvidence:
    outcome = resolve_test_outcome(state.phase_outcomes)
    categories = sorted(
        {
            span.failure_category
            for span in state.spans
            if span.failure_category is not None
        },
        key=lambda item: item.value,
    )
    category = categories[0] if categories else state.test_failure_category
    attributes = [span.attributes for span in state.spans]
    return CiTestEvidence(
        test_id=state.test_id,
        node_id=state.node_id,
        outcome=outcome,
        duration_ms=round(max(state.duration_ms, 0.0), 3),
        correlation_id=state.correlation_id,
        trace_ids=_sorted_values(span.trace_id for span in state.spans),
        span_ids=_sorted_values(span.span_id for span in state.spans),
        case_ids=_attribute_values(attributes, "case_id"),
        document_ids=_attribute_values(attributes, "document_id"),
        chunk_ids=_attribute_values(attributes, "chunk_id"),
        failure_category=category if outcome == "failed" else None,
        evaluation_statuses=_attribute_values(attributes, "evaluation_status"),
        artifact_refs=_artifact_refs(state.spans),
    )


def resolve_test_outcome(phase_outcomes: dict[str, str]) -> TestOutcome:
    values = set(phase_outcomes.values())
    if "failed" in values:
        return "failed"
    if "xpassed" in values:
        return "xpassed"
    if "xfailed" in values:
        return "xfailed"
    if "skipped" in values:
        return "skipped"
    return "passed"


def build_correlation_report(tests: list[CiTestEvidence]) -> CiCorrelationReport:
    ordered = sorted(tests, key=lambda item: (item.test_id, item.node_id or ""))
    failures = [
        CiFailureEvidence(test_id=item.test_id, failure_category=item.failure_category)
        for item in ordered
        if item.outcome == "failed" and item.failure_category is not None
    ]
    outcomes = {name: 0 for name in ("passed", "failed", "skipped", "xfailed", "xpassed")}
    for item in ordered:
        outcomes[item.outcome] += 1
    categories: dict[str, int] = {}
    for item in failures:
        key = item.failure_category.value
        categories[key] = categories.get(key, 0) + 1
    summary: dict[str, int | dict[str, int]] = {
        "total_tests": len(ordered),
        **outcomes,
        "failures_by_category": dict(sorted(categories.items())),
        "tests_with_traces": sum(bool(item.trace_ids) for item in ordered),
        "tests_without_traces": sum(not item.trace_ids for item in ordered),
    }
    return CiCorrelationReport(tests=ordered, failures=failures, summary=summary)


def write_test_evidence(
    evidence: CiTestEvidence, root: Path, worker_id: str
) -> Path:
    path = root / "workers" / worker_id / f"{evidence.test_id}.evidence.json"
    _write_json_atomic(path, evidence.model_dump(mode="json"))
    return path


def load_test_evidence(root: Path) -> list[CiTestEvidence]:
    return [
        CiTestEvidence.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((root / "workers").glob("*/*.evidence.json"))
    ]


def write_test_spans(
    spans: list[SpanEvidence], test_id: str, root: Path, worker_id: str
) -> Path:
    path = root / "workers" / worker_id / f"{test_id}.spans.json"
    _write_json_atomic(path, [span.model_dump(mode="json") for span in spans])
    return path


def load_test_spans(root: Path) -> list[SpanEvidence]:
    spans: list[SpanEvidence] = []
    for path in sorted((root / "workers").glob("*/*.spans.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        spans.extend(SpanEvidence.model_validate(item) for item in payload)
    return spans


def write_ci_reports(
    tests: list[CiTestEvidence],
    *,
    report_path: Path = CI_REPORT_PATH,
    failure_path: Path = FAILURE_REPORT_PATH,
    trace_path: Path = TRACE_REPORT_PATH,
) -> CiCorrelationReport:
    report = build_correlation_report(tests)
    _write_json_atomic(report_path, report.model_dump(mode="json"))
    _write_json_atomic(failure_path, report.summary)
    return report


def write_trace_report(spans: list[SpanEvidence], path: Path = TRACE_REPORT_PATH) -> None:
    grouped: dict[tuple[str, str, str, str], list[SpanEvidence]] = {}
    for span in spans:
        key = (span.trace_id, span.correlation_id, span.service_name, span.environment)
        grouped.setdefault(key, []).append(span)
    envelopes = [
        TraceEnvelope(
            trace_id=key[0],
            correlation_id=key[1],
            service_name=key[2],
            environment=key[3],
            spans=sorted(items, key=lambda item: (item.started_at, item.span_id)),
        )
        for key, items in sorted(grouped.items())
    ]
    _write_json_atomic(path, [item.model_dump(mode="json") for item in envelopes])


def worker_id_from_environment() -> str:
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "main")
    return worker_id if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", worker_id) else "worker"


def _attribute_values(attributes: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for item in attributes:
        value = item.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            values.extend(entry for entry in value if isinstance(entry, str))
    return _sorted_values(values)


def _sorted_values(values: Any) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def _artifact_refs(spans: list[SpanEvidence]) -> list[str]:
    refs = {"reports/observability/traces.json"} if spans else set()
    if any(span.operation_name == "ai.evaluation" for span in spans):
        refs.add("reports/ai_eval/phase10_report.json")
    return sorted(refs)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
