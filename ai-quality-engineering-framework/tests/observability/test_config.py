from __future__ import annotations

import pytest

from observability.config import ObservabilitySettings


pytestmark = pytest.mark.observability


def test_observability_is_disabled_and_content_safe_by_default(monkeypatch):
    for name in (
        "AI_OBSERVABILITY_ENABLED",
        "AI_OBSERVABILITY_EXPORTER",
        "AI_OBSERVABILITY_CAPTURE_CONTENT",
        "AI_OBSERVABILITY_CAPTURE_PROMPT",
        "AI_OBSERVABILITY_CAPTURE_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = ObservabilitySettings.from_env()
    assert settings.enabled is False
    assert settings.exporter == "none"
    assert settings.capture_content is False
    assert settings.capture_prompt is False
    assert settings.capture_context is False


def test_prompt_and_context_capture_require_content_opt_in(monkeypatch):
    monkeypatch.setenv("AI_OBSERVABILITY_CAPTURE_PROMPT", "true")
    monkeypatch.setenv("AI_OBSERVABILITY_CAPTURE_CONTEXT", "true")
    assert ObservabilitySettings.from_env().capture_prompt is False
    assert ObservabilitySettings.from_env().capture_context is False

    monkeypatch.setenv("AI_OBSERVABILITY_CAPTURE_CONTENT", "true")
    settings = ObservabilitySettings.from_env()
    assert settings.capture_prompt is True
    assert settings.capture_context is True


def test_valid_otlp_configuration(monkeypatch):
    monkeypatch.setenv("AI_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AI_OBSERVABILITY_EXPORTER", "otlp")
    monkeypatch.setenv("AI_OBSERVABILITY_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setenv("AI_OBSERVABILITY_OTLP_TIMEOUT_SECONDS", "2.5")
    settings = ObservabilitySettings.from_env()
    assert settings.otlp_endpoint == "http://localhost:4318/v1/traces"
    assert settings.otlp_timeout_seconds == 2.5


def test_otlp_endpoint_is_required_only_when_enabled():
    ObservabilitySettings(enabled=False, exporter="otlp")
    with pytest.raises(ValueError, match="OTLP_ENDPOINT"):
        ObservabilitySettings(enabled=True, exporter="otlp")


@pytest.mark.parametrize("value", ["invalid", "", "0", "-1"])
def test_invalid_otlp_timeout(monkeypatch, value):
    monkeypatch.setenv("AI_OBSERVABILITY_OTLP_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="OTLP_TIMEOUT_SECONDS"):
        ObservabilitySettings.from_env()


def test_invalid_exporter_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_OBSERVABILITY_EXPORTER", "unknown")
    with pytest.raises(ValueError, match="AI_OBSERVABILITY_EXPORTER"):
        ObservabilitySettings.from_env()
