from __future__ import annotations

from dataclasses import dataclass

import httpx

from observability.models import FailureCategory


TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})


@dataclass(frozen=True)
class ProviderFailureMetadata:
    category: FailureCategory
    retryable: bool
    attempt: int
    max_attempts: int
    status_code: int | None = None

    def safe_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "status_code": self.status_code,
        }


def classify_http_failure(
    status_code: int, *, attempt: int, max_attempts: int
) -> ProviderFailureMetadata:
    if status_code == 401:
        category = FailureCategory.AUTHENTICATION
        retryable = False
    elif status_code == 403:
        category = FailureCategory.AUTHORIZATION
        retryable = False
    elif status_code == 429:
        category = FailureCategory.RATE_LIMIT
        retryable = True
    elif status_code in TRANSIENT_HTTP_STATUSES:
        category = FailureCategory.PROVIDER
        retryable = True
    elif 400 <= status_code < 500:
        category = FailureCategory.CONTRACT
        retryable = False
    else:
        category = FailureCategory.PROVIDER
        retryable = False
    return ProviderFailureMetadata(
        category=category,
        retryable=retryable,
        status_code=status_code,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def classify_transport_failure(
    exception: httpx.RequestError, *, attempt: int, max_attempts: int
) -> ProviderFailureMetadata:
    if isinstance(exception, httpx.TimeoutException):
        category = FailureCategory.TIMEOUT
        retryable = True
    elif isinstance(exception, httpx.NetworkError):
        category = FailureCategory.NETWORK
        retryable = True
    else:
        category = FailureCategory.NETWORK
        retryable = False
    return ProviderFailureMetadata(
        category=category,
        retryable=retryable,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def exponential_backoff(
    attempt: int, *, base_delay_seconds: float = 0.25, max_delay_seconds: float = 2.0
) -> float:
    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    return min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
