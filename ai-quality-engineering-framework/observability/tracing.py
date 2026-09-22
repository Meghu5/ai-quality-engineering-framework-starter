from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from types import TracebackType
from typing import Any

from observability.config import DEFAULT_OBSERVABILITY_SETTINGS, ObservabilitySettings
from observability.context import generate_span_id, get_context, use_context
from observability.exporters import InMemoryExporter, JsonEvidenceExporter, SpanExporter
from observability.models import FailureCategory, SpanEvidence, TraceStatus
from observability.redaction import sanitize_attributes


logger = logging.getLogger(__name__)


class Span:
    def __init__(
        self,
        facade: "TracingFacade",
        operation_name: str,
        *,
        operation_type: str,
        correlation_id: str | None,
        attributes: dict[str, Any] | None,
        new_trace: bool,
    ) -> None:
        self._facade = facade
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.correlation_id = correlation_id
        self.attributes = sanitize_attributes(attributes)
        self.new_trace = new_trace
        self.evidence: SpanEvidence | None = None
        self._scope = None
        self._started_at: datetime | None = None
        self._started_counter: float | None = None
        self._parent_span_id: str | None = None
        self._status = TraceStatus.UNSET
        self._failure_category: FailureCategory | None = None

    def __enter__(self) -> "Span":
        if not self._facade.enabled:
            return self
        try:
            parent = None if self.new_trace else get_context()
            self._parent_span_id = parent.span_id if parent else None
            self._scope = use_context(
                trace_id=None if self.new_trace else (parent.trace_id if parent else None),
                span_id=generate_span_id(),
                correlation_id=self.correlation_id
                or (parent.correlation_id if parent else None),
            )
            self._scope.__enter__()
            self._started_at = datetime.now(timezone.utc)
            self._started_counter = time.perf_counter()
        except Exception:
            logger.exception("Unable to start observability span")
            self._scope = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc is not None:
            self.record_exception(
                exc, self._failure_category or FailureCategory.UNKNOWN
            )
        elif self._status == TraceStatus.UNSET:
            self._status = TraceStatus.OK
        self.end()
        return False

    def set_attribute(self, name: str, value: Any) -> None:
        try:
            self.attributes.update(sanitize_attributes({name: value}))
        except Exception:
            logger.exception("Unable to set observability attribute")

    def record_exception(
        self,
        exception: BaseException,
        category: FailureCategory = FailureCategory.UNKNOWN,
    ) -> None:
        try:
            self._status = TraceStatus.ERROR
            self._failure_category = category
            self.attributes.update(
                sanitize_attributes({"exception_type": type(exception).__name__})
            )
        except Exception:
            logger.exception("Unable to record observability exception")

    def end(self) -> SpanEvidence | None:
        if not self._facade.enabled or self._started_at is None:
            return None
        try:
            ended_at = datetime.now(timezone.utc)
            duration_ms = max(
                0.0, (time.perf_counter() - (self._started_counter or 0.0)) * 1000
            )
            context = get_context()
            if context is None:
                return None
            self.evidence = SpanEvidence(
                trace_id=context.trace_id,
                span_id=context.span_id or generate_span_id(),
                parent_span_id=self._parent_span_id,
                correlation_id=context.correlation_id,
                operation_name=self.operation_name,
                operation_type=self.operation_type,
                started_at=self._started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                status=self._status,
                failure_category=self._failure_category,
                attributes=self.attributes,
                service_name=self._facade.settings.service_name,
                environment=self._facade.settings.environment,
            )
            self._facade._export(self.evidence)
            return self.evidence
        except Exception:
            logger.exception("Unable to finish observability span")
            return None
        finally:
            self._started_at = None
            if self._scope is not None:
                try:
                    self._scope.__exit__(None, None, None)
                except Exception:
                    logger.exception("Unable to reset observability context")
                self._scope = None


class TracingFacade:
    def __init__(
        self,
        settings: ObservabilitySettings = DEFAULT_OBSERVABILITY_SETTINGS,
        exporter: SpanExporter | None = None,
    ) -> None:
        self.settings = settings
        self.exporter = exporter

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and self.exporter is not None

    def start_trace(
        self,
        operation_name: str,
        *,
        operation_type: str = "internal",
        correlation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        return Span(
            self,
            operation_name,
            operation_type=operation_type,
            correlation_id=correlation_id,
            attributes=attributes,
            new_trace=True,
        )

    def start_span(
        self,
        operation_name: str,
        *,
        operation_type: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        return Span(
            self,
            operation_name,
            operation_type=operation_type,
            correlation_id=None,
            attributes=attributes,
            new_trace=False,
        )

    def get_current_trace(self) -> str | None:
        context = get_context()
        return context.trace_id if self.enabled and context else None

    def _export(self, evidence: SpanEvidence) -> None:
        try:
            if self.exporter is not None:
                self.exporter.export(evidence)
        except Exception:
            logger.exception("Observability exporter failed")


def create_tracing_facade(
    settings: ObservabilitySettings | None = None,
) -> TracingFacade:
    resolved = settings or ObservabilitySettings.from_env()
    exporter: SpanExporter | None = None
    if resolved.enabled:
        if resolved.exporter == "memory":
            exporter = InMemoryExporter()
        elif resolved.exporter == "json":
            exporter = JsonEvidenceExporter()
        elif resolved.exporter == "otlp":
            try:
                from observability.otel import OtlpSpanExporterAdapter

                exporter = OtlpSpanExporterAdapter(resolved)
            except Exception:
                logger.exception("Unable to initialize OTLP observability exporter")
    return TracingFacade(settings=resolved, exporter=exporter)
