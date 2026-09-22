from __future__ import annotations

import os
from dataclasses import dataclass


VALID_EXPORTERS = frozenset({"none", "memory", "json", "otlp"})


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
    otlp_timeout_seconds: float = 5.0
    capture_content: bool = False
    capture_prompt: bool = False
    capture_context: bool = False

    def __post_init__(self) -> None:
        if self.otlp_timeout_seconds <= 0:
            raise ValueError("AI_OBSERVABILITY_OTLP_TIMEOUT_SECONDS must be greater than zero")
        if self.enabled and self.exporter == "otlp" and not self.otlp_endpoint:
            raise ValueError(
                "AI_OBSERVABILITY_OTLP_ENDPOINT is required when the OTLP exporter is enabled"
            )

    @classmethod
    def from_env(cls) -> "ObservabilitySettings":
        capture_content = _enabled("AI_OBSERVABILITY_CAPTURE_CONTENT")
        exporter = os.getenv("AI_OBSERVABILITY_EXPORTER", "none").strip().lower() or "none"
        if exporter not in VALID_EXPORTERS:
            raise ValueError(
                f"AI_OBSERVABILITY_EXPORTER must be one of: {', '.join(sorted(VALID_EXPORTERS))}"
            )
        timeout_value = os.getenv("AI_OBSERVABILITY_OTLP_TIMEOUT_SECONDS", "5").strip()
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as exc:
            raise ValueError(
                "AI_OBSERVABILITY_OTLP_TIMEOUT_SECONDS must be a number"
            ) from exc
        return cls(
            enabled=_enabled("AI_OBSERVABILITY_ENABLED"),
            service_name=os.getenv(
                "AI_OBSERVABILITY_SERVICE_NAME", "ai-quality-engineering-framework"
            ).strip()
            or "ai-quality-engineering-framework",
            environment=os.getenv("AI_OBSERVABILITY_ENVIRONMENT", "test").strip()
            or "test",
            exporter=exporter,
            otlp_endpoint=(os.getenv("AI_OBSERVABILITY_OTLP_ENDPOINT") or "").strip()
            or None,
            otlp_timeout_seconds=timeout_seconds,
            capture_content=capture_content,
            capture_prompt=capture_content
            and _enabled("AI_OBSERVABILITY_CAPTURE_PROMPT"),
            capture_context=capture_content
            and _enabled("AI_OBSERVABILITY_CAPTURE_CONTEXT"),
        )


DEFAULT_OBSERVABILITY_SETTINGS = ObservabilitySettings.from_env()
