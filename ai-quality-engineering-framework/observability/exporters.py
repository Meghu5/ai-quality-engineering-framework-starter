from __future__ import annotations

import json
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Protocol

from observability.models import SpanEvidence, TraceEnvelope


logger = logging.getLogger(__name__)
DEFAULT_EVIDENCE_PATH = Path("reports") / "observability" / "traces.json"


class SpanExporter(Protocol):
    def export(self, span: SpanEvidence) -> None: ...


class InMemoryExporter:
    def __init__(self) -> None:
        self._spans: list[SpanEvidence] = []
        self._lock = Lock()

    def export(self, span: SpanEvidence) -> None:
        with self._lock:
            self._spans.append(span.model_copy(deep=True))

    @property
    def spans(self) -> list[SpanEvidence]:
        with self._lock:
            return deepcopy(self._spans)

    def envelopes(self) -> list[TraceEnvelope]:
        grouped: dict[tuple[str, str, str, str], list[SpanEvidence]] = defaultdict(list)
        for span in self.spans:
            key = (span.trace_id, span.correlation_id, span.service_name, span.environment)
            grouped[key].append(span)
        envelopes = []
        for key, spans in sorted(grouped.items()):
            trace_id, correlation_id, service_name, environment = key
            envelopes.append(
                TraceEnvelope(
                    trace_id=trace_id,
                    correlation_id=correlation_id,
                    service_name=service_name,
                    environment=environment,
                    spans=sorted(spans, key=lambda item: (item.started_at, item.span_id)),
                )
            )
        return envelopes

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


class JsonEvidenceExporter(InMemoryExporter):
    def __init__(self, path: Path = DEFAULT_EVIDENCE_PATH) -> None:
        super().__init__()
        self.path = path

    def export(self, span: SpanEvidence) -> None:
        super().export(span)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = [item.model_dump(mode="json") for item in self.envelopes()]
            self.path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            logger.exception("Unable to write observability evidence")
