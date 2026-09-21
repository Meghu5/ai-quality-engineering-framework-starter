from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


REDACTED = "[REDACTED]"

SAFE_ATTRIBUTE_KEYS = {
    "case_id",
    "chunk_id",
    "context_count",
    "document_id",
    "evaluation_status",
    "exception_type",
    "latency_ms",
    "model_name",
    "operation",
    "prompt_length",
    "provider_name",
    "response_length",
    "status_code",
    "token_count",
}

_SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "cookie",
    "secret",
    "prompt",
    "context",
    "response",
    "content",
    "email",
    "phone",
    "passport",
    "card",
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        normalized = str(key).strip().lower()
        if normalized not in SAFE_ATTRIBUTE_KEYS:
            continue
        if _is_sensitive_key(normalized):
            continue
        sanitized[normalized] = _sanitize_value(value)
    return sanitized


def redact_text(value: str) -> str:
    redacted = _EMAIL.sub(REDACTED, value)
    redacted = _PHONE.sub(REDACTED, redacted)
    return _CARD.sub(REDACTED, redacted)


def _is_sensitive_key(key: str) -> bool:
    if key in {"prompt_length", "context_count", "response_length"}:
        return False
    return any(part in key for part in _SENSITIVE_KEY_PARTS)


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)[:512]
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value[:50]]
    return str(value)[:512]
