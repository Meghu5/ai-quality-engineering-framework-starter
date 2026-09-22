from __future__ import annotations

import logging
from collections.abc import Sequence
from threading import Thread
from typing import Any

from observability.config import ObservabilitySettings
from observability.models import SpanEvidence, TraceStatus
from observability.redaction import sanitize_attributes


logger = logging.getLogger(__name__)


def normalize_otel_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if not attributes:
        return normalized
    supported = {
        str(key).strip().lower(): value
        for key, value in attributes.items()
        if _is_otel_scalar(value) or _is_safe_sequence(value)
    }
    for key, value in sanitize_attributes(supported).items():
        if _is_otel_scalar(value):
            normalized[key] = value
        elif _is_safe_sequence(value):
            normalized[key] = tuple(value)
    return normalized


def _is_otel_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float)) and not isinstance(value, bytes)


def _is_safe_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first_type = type(value[0])
    return first_type in {str, bool, int, float} and all(
        type(item) is first_type for item in value
    )


class OtlpSpanExporterAdapter:
    """Maps framework span evidence to OTEL while preserving framework IDs."""

    def __init__(
        self,
        settings: ObservabilitySettings,
        *,
        otel_exporter: Any | None = None,
        span_processor: Any | None = None,
    ) -> None:
        self.settings = settings
        self._closed = False
        if span_processor is not None:
            self._processor = span_processor
            return

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = otel_exporter or OTLPSpanExporter(
            endpoint=settings.otlp_endpoint,
            timeout=settings.otlp_timeout_seconds,
        )
        self._processor = BatchSpanProcessor(
            exporter,
            max_queue_size=256,
            schedule_delay_millis=200,
            max_export_batch_size=64,
            export_timeout_millis=settings.otlp_timeout_seconds * 1000,
        )

    def export(self, span: SpanEvidence) -> None:
        if self._closed:
            return
        try:
            self._processor.on_end(self._to_readable_span(span))
        except Exception:
            logger.exception("Unable to export OTLP span")

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        if self._closed:
            return True
        bounded_timeout = timeout_millis or int(self.settings.otlp_timeout_seconds * 1000)
        try:
            return bool(self._processor.force_flush(timeout_millis=bounded_timeout))
        except Exception:
            logger.exception("Unable to flush OTLP spans")
            return False

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []

        def stop_processor() -> None:
            try:
                self._processor.shutdown()
            except BaseException as exc:
                errors.append(exc)

        worker = Thread(target=stop_processor, name="otlp-shutdown", daemon=True)
        worker.start()
        worker.join(timeout=self.settings.otlp_timeout_seconds)
        if worker.is_alive():
            logger.warning("OTLP exporter shutdown exceeded the configured timeout")
        elif errors:
            logger.error(
                "Unable to shut down OTLP exporter (%s)", type(errors[0]).__name__
            )

    def _to_readable_span(self, span: SpanEvidence) -> Any:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import Event, ReadableSpan
        from opentelemetry.trace.status import Status, StatusCode

        context = trace.SpanContext(
            trace_id=int(span.trace_id, 16),
            span_id=int(span.span_id, 16),
            is_remote=False,
            trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
            trace_state=trace.TraceState(),
        )
        parent = None
        if span.parent_span_id:
            parent = trace.SpanContext(
                trace_id=int(span.trace_id, 16),
                span_id=int(span.parent_span_id, 16),
                is_remote=False,
                trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                trace_state=trace.TraceState(),
            )

        attributes = normalize_otel_attributes(span.attributes)
        attributes.update(
            {
                "ai.observability.operation_type": span.operation_type,
                "ai.observability.correlation_id": span.correlation_id,
            }
        )
        if span.failure_category is not None:
            attributes["ai.observability.failure_category"] = span.failure_category.value

        exception_type = attributes.pop("exception_type", None)
        events: Sequence[Any] = ()
        if exception_type:
            events = (
                Event(
                    "exception",
                    attributes={"exception.type": exception_type},
                    timestamp=int(span.ended_at.timestamp() * 1_000_000_000),
                ),
            )

        status_codes = {
            TraceStatus.OK: StatusCode.OK,
            TraceStatus.ERROR: StatusCode.ERROR,
            TraceStatus.UNSET: StatusCode.UNSET,
        }
        return ReadableSpan(
            name=span.operation_name,
            context=context,
            parent=parent,
            resource=Resource.create(
                {
                    "service.name": span.service_name,
                    "deployment.environment.name": span.environment,
                }
            ),
            attributes=attributes,
            events=events,
            status=Status(status_codes[span.status]),
            start_time=int(span.started_at.timestamp() * 1_000_000_000),
            end_time=int(span.ended_at.timestamp() * 1_000_000_000),
        )
