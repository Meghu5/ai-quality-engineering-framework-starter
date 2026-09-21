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
