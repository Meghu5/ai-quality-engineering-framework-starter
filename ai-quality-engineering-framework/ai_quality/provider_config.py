from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit


DEFAULT_REAL_PROVIDER_TIMEOUT_SECONDS = 30.0
MAX_REAL_PROVIDER_TIMEOUT_SECONDS = 120.0
DEFAULT_REAL_PROVIDER_MAX_ATTEMPTS = 3
MAX_REAL_PROVIDER_MAX_ATTEMPTS = 5


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RealLLMProviderSettings:
    """Non-secret configuration for the optional HTTP LLM provider."""

    enabled: bool = False
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = DEFAULT_REAL_PROVIDER_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_REAL_PROVIDER_MAX_ATTEMPTS
    require_structured_output: bool = True

    @classmethod
    def from_env(
        cls, environment: dict[str, str] | None = None
    ) -> "RealLLMProviderSettings":
        env = environment if environment is not None else os.environ
        timeout_value = env.get(
            "AI_REAL_PROVIDER_TIMEOUT_SECONDS",
            str(DEFAULT_REAL_PROVIDER_TIMEOUT_SECONDS),
        ).strip()
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as exc:
            raise ValueError(
                "AI_REAL_PROVIDER_TIMEOUT_SECONDS must be a number"
            ) from exc
        max_attempts_value = env.get(
            "AI_REAL_PROVIDER_MAX_ATTEMPTS",
            str(DEFAULT_REAL_PROVIDER_MAX_ATTEMPTS),
        ).strip()
        try:
            max_attempts = int(max_attempts_value)
        except ValueError as exc:
            raise ValueError(
                "AI_REAL_PROVIDER_MAX_ATTEMPTS must be an integer"
            ) from exc
        settings = cls(
            enabled=_enabled(env.get("AI_REAL_PROVIDER_ENABLED")),
            base_url=env.get("AI_REAL_PROVIDER_BASE_URL", "").strip(),
            model=env.get("AI_REAL_PROVIDER_MODEL", "").strip(),
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            require_structured_output=_enabled(
                env.get("AI_REAL_PROVIDER_REQUIRE_STRUCTURED_OUTPUT", "true")
            ),
        )
        settings.validate()
        return settings

    def validate(self, *, api_key_present: bool | None = None) -> None:
        if not 0 < self.timeout_seconds <= MAX_REAL_PROVIDER_TIMEOUT_SECONDS:
            raise ValueError(
                "AI_REAL_PROVIDER_TIMEOUT_SECONDS must be greater than zero "
                f"and at most {MAX_REAL_PROVIDER_TIMEOUT_SECONDS:g}"
            )
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise ValueError("AI_REAL_PROVIDER_MAX_ATTEMPTS must be an integer")
        if not 1 <= self.max_attempts <= MAX_REAL_PROVIDER_MAX_ATTEMPTS:
            raise ValueError(
                "AI_REAL_PROVIDER_MAX_ATTEMPTS must be between 1 "
                f"and {MAX_REAL_PROVIDER_MAX_ATTEMPTS}"
            )
        if not self.enabled:
            return
        if not self.base_url:
            raise ValueError("AI_REAL_PROVIDER_BASE_URL is required when enabled")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("AI_REAL_PROVIDER_BASE_URL must be an absolute HTTP URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "AI_REAL_PROVIDER_BASE_URL must not contain credentials, a query, or a fragment"
            )
        is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not is_local:
            raise ValueError("AI_REAL_PROVIDER_BASE_URL must use HTTPS outside localhost")
        if not self.model:
            raise ValueError("AI_REAL_PROVIDER_MODEL is required when enabled")
        if api_key_present is False:
            raise ValueError("AI_REAL_PROVIDER_API_KEY is required when enabled")

    def safe_dict(self) -> dict[str, object]:
        return asdict(self)
