from __future__ import annotations

import os
from dataclasses import dataclass


def _enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ObservabilitySettings:
    enabled: bool = False
    service_name: str = "ai-quality-engineering-framework"
    environment: str = "test"
    exporter: str = "none"
    otlp_endpoint: str | None = None
    capture_content: bool = False
    capture_prompt: bool = False
    capture_context: bool = False

    @classmethod
    def from_env(cls) -> "ObservabilitySettings":
        capture_content = _enabled("AI_OBSERVABILITY_CAPTURE_CONTENT")
        return cls(
            enabled=_enabled("AI_OBSERVABILITY_ENABLED"),
            service_name=os.getenv(
                "AI_OBSERVABILITY_SERVICE_NAME", "ai-quality-engineering-framework"
            ).strip()
            or "ai-quality-engineering-framework",
            environment=os.getenv("AI_OBSERVABILITY_ENVIRONMENT", "test").strip()
            or "test",
            exporter=os.getenv("AI_OBSERVABILITY_EXPORTER", "none").strip().lower()
            or "none",
            otlp_endpoint=os.getenv("AI_OBSERVABILITY_OTLP_ENDPOINT") or None,
            capture_content=capture_content,
            capture_prompt=capture_content
            and _enabled("AI_OBSERVABILITY_CAPTURE_PROMPT"),
            capture_context=capture_content
            and _enabled("AI_OBSERVABILITY_CAPTURE_CONTEXT"),
        )


DEFAULT_OBSERVABILITY_SETTINGS = ObservabilitySettings.from_env()
