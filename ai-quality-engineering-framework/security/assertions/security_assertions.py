from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx


SENSITIVE_FIELD_NAMES = {
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
}

LEAK_INDICATORS = (
    "Traceback",
    "File \"",
    "C:\\",
    "/home/",
    "OperationalError",
    "DatabaseError",
    "SELECT ",
    "INSERT ",
    "UPDATE ",
    "DELETE ",
)


def assert_status(response: httpx.Response, expected_status: int) -> None:
    assert response.status_code == expected_status, (
        f"expected HTTP {expected_status}, got HTTP {response.status_code}: "
        f"{_safe_body(response)}"
    )


def assert_unauthorized(response: httpx.Response) -> None:
    assert_status(response, 401)
    assert_safe_error_response(response)


def assert_forbidden(response: httpx.Response) -> None:
    assert_status(response, 403)
    assert_safe_error_response(response)


def assert_bad_request(response: httpx.Response) -> None:
    assert response.status_code in {400, 413, 422}, (
        f"expected client validation failure, got HTTP {response.status_code}: "
        f"{_safe_body(response)}"
    )
    assert_safe_error_response(response)


def assert_safe_error_response(response: httpx.Response) -> None:
    body = _safe_body(response)
    assert "error" in body
    assert_no_internal_details(body)
    assert_no_sensitive_fields(response.json())


def assert_no_sensitive_fields(payload: Any) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert key.lower() not in SENSITIVE_FIELD_NAMES
            assert_no_sensitive_fields(value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_sensitive_fields(item)


def assert_no_internal_details(text: str) -> None:
    for indicator in LEAK_INDICATORS:
        assert indicator not in text


def assert_security_headers(response: httpx.Response) -> None:
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


def _safe_body(response: httpx.Response) -> str:
    try:
        return response.text
    except UnicodeDecodeError:
        return "<non-text response>"
