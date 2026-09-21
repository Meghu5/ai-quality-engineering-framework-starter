from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalProvider:
    name: str = "none"

    @property
    def is_configured(self) -> bool:
        return self.name not in {"", "none"}


def provider_from_name(name: str) -> EvalProvider:
    return EvalProvider(name=(name or "none").strip().lower())
