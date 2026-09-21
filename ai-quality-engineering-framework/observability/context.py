from __future__ import annotations

import secrets
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str | None
    correlation_id: str


_current_context: ContextVar[TraceContext | None] = ContextVar(
    "ai_observability_context", default=None
)


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def generate_span_id() -> str:
    return secrets.token_hex(8)


def generate_correlation_id() -> str:
    return secrets.token_hex(16)


def get_context() -> TraceContext | None:
    return _current_context.get()


def get_trace_id() -> str | None:
    context = get_context()
    return context.trace_id if context else None


def get_span_id() -> str | None:
    context = get_context()
    return context.span_id if context else None


def get_correlation_id() -> str | None:
    context = get_context()
    return context.correlation_id if context else None


def set_correlation_id(correlation_id: str | None = None) -> Token:
    current = get_context()
    value = _validate_correlation_id(correlation_id or generate_correlation_id())
    context = TraceContext(
        trace_id=current.trace_id if current else generate_trace_id(),
        span_id=current.span_id if current else None,
        correlation_id=value,
    )
    return _current_context.set(context)


def reset_context(token: Token) -> None:
    _current_context.reset(token)


def clear_context() -> None:
    _current_context.set(None)


@contextmanager
def use_context(
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    correlation_id: str | None = None,
) -> Iterator[TraceContext]:
    parent = get_context()
    context = TraceContext(
        trace_id=_validate_hex_id(
            trace_id or (parent.trace_id if parent else generate_trace_id()), 32, "trace_id"
        ),
        span_id=_validate_hex_id(span_id, 16, "span_id") if span_id else None,
        correlation_id=_validate_correlation_id(
            correlation_id
            or (parent.correlation_id if parent else generate_correlation_id())
        ),
    )
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


def _validate_hex_id(value: str, length: int, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != length or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a {length}-character hexadecimal value")
    if set(normalized) == {"0"}:
        raise ValueError(f"{name} must not be all zeros")
    return normalized


def _validate_correlation_id(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128:
        raise ValueError("correlation_id must contain 1 to 128 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        raise ValueError("correlation_id must not contain control characters")
    return normalized
