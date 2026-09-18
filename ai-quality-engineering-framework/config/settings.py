from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    api_timeout_seconds: float
    api_auth_token: str | None
    api_latency_threshold_seconds: float
    ui_base_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_base_url = os.getenv("API_BASE_URL") or os.getenv(
            "CHAT_BASE_URL", "http://localhost:8000"
        )
        ui_base_url = os.getenv("UI_BASE_URL") or os.getenv(
            "CHAT_BASE_URL", api_base_url
        )
        api_auth_token = os.getenv("API_AUTH_TOKEN")

        return cls(
            api_base_url=api_base_url.rstrip("/"),
            api_timeout_seconds=_get_float("API_TIMEOUT_SECONDS", 30.0),
            api_auth_token=api_auth_token or None,
            api_latency_threshold_seconds=_get_float(
                "API_LATENCY_THRESHOLD_SECONDS", 5.0
            ),
            ui_base_url=ui_base_url.rstrip("/"),
        )
