from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FailureCategory(str, Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    PROVIDER = "provider"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    GROUNDING = "grounding"
    SAFETY = "safety"
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    TOOL = "tool"
    CONTRACT = "contract"
    PERFORMANCE = "performance"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class TraceStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    correlation_id: str
    operation_name: str = Field(min_length=1, max_length=200)
    operation_type: str = Field(default="internal", min_length=1, max_length=100)
    started_at: datetime
    ended_at: datetime
    duration_ms: float = Field(ge=0)
    status: TraceStatus = TraceStatus.UNSET
    failure_category: FailureCategory | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    service_name: str
    environment: str

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        return _validate_hex(value, 32, "trace_id")

    @field_validator("span_id", "parent_span_id")
    @classmethod
    def validate_span_id(cls, value: str | None) -> str | None:
        return _validate_hex(value, 16, "span_id") if value is not None else None

    @model_validator(mode="after")
    def validate_timing_and_status(self) -> "SpanEvidence":
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.failure_category is not None and self.status != TraceStatus.ERROR:
            raise ValueError("failure_category requires error status")
        return self


class TraceEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    correlation_id: str
    service_name: str
    environment: str
    spans: list[SpanEvidence] = Field(default_factory=list)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        return _validate_hex(value, 32, "trace_id")

    @model_validator(mode="after")
    def validate_relationships(self) -> "TraceEnvelope":
        for span in self.spans:
            if span.trace_id != self.trace_id:
                raise ValueError("every span must belong to the envelope trace")
            if span.correlation_id != self.correlation_id:
                raise ValueError("every span must use the envelope correlation ID")
        return self


def _validate_hex(value: str, length: int, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a {length}-character hexadecimal value")
    if set(normalized) == {"0"}:
        raise ValueError(f"{name} must not be all zeros")
    return normalized
