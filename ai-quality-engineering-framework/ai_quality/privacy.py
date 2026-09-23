from __future__ import annotations

import hashlib
from dataclasses import dataclass


SAFE_PROVIDER_LABELS = frozenset(
    {"deterministic", "http-json", "none", "provider-a", "real_http", "test-provider"}
)
SAFE_MODEL_LABELS = frozenset(
    {
        "airline-model-v1",
        "gpt-4o-mini",
        "none",
        "rule-based-airline-assistant",
        "test-model",
        "unconfigured",
    }
)


@dataclass(frozen=True, slots=True)
class _SafeIdentifier:
    value: str


def safe_provider_label(value: str) -> _SafeIdentifier:
    return _safe_registered_label(value, SAFE_PROVIDER_LABELS)


def safe_model_label(value: str) -> _SafeIdentifier:
    return _safe_registered_label(value, SAFE_MODEL_LABELS)


def safe_case_id(value: str) -> _SafeIdentifier:
    return _hashed_value("case", value)


def _safe_registered_label(value: str, allowed: frozenset[str]) -> _SafeIdentifier:
    normalized = (value or "none").strip()
    if normalized in allowed:
        return _SafeIdentifier(normalized)
    return _hashed_value("opaque", normalized)


def _hashed_value(prefix: str, value: str) -> _SafeIdentifier:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return _SafeIdentifier(f"{prefix}-{digest}")
