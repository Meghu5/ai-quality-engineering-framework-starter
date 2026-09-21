from __future__ import annotations

import os
from dataclasses import dataclass


def _enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AIEvalSettings:
    ragas_enabled: bool
    deepeval_enabled: bool
    promptfoo_enabled: bool
    provider: str
    fail_on_optional_unavailable: bool = False

    @classmethod
    def from_env(cls) -> "AIEvalSettings":
        return cls(
            ragas_enabled=_enabled("AI_EVAL_RAGAS_ENABLED"),
            deepeval_enabled=_enabled("AI_EVAL_DEEPEVAL_ENABLED"),
            promptfoo_enabled=_enabled("AI_EVAL_PROMPTFOO_ENABLED"),
            provider=os.getenv("AI_EVAL_PROVIDER", "none").strip().lower() or "none",
            fail_on_optional_unavailable=_enabled("AI_EVAL_FAIL_ON_OPTIONAL_UNAVAILABLE"),
        )


DEFAULT_AI_EVAL_SETTINGS = AIEvalSettings.from_env()
