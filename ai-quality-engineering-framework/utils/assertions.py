from __future__ import annotations

from typing import Any, Iterable

import httpx


def assert_status_code(
    response: httpx.Response, expected_status_code: int, *, endpoint: str
) -> None:
    assert response.status_code == expected_status_code, (
        f"{endpoint}: expected status {expected_status_code}, "
        f"got {response.status_code}. Response body: {response.text}"
    )


def assert_json_response(response: httpx.Response, *, endpoint: str) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise AssertionError(
            f"{endpoint}: expected valid JSON response, got {response.text!r}"
        ) from exc

    assert isinstance(body, dict), (
        f"{endpoint}: expected JSON object, got {type(body).__name__}"
    )
    return body


def assert_required_fields(
    body: dict[str, Any], required_fields: Iterable[str], *, endpoint: str
) -> None:
    missing_fields = [field for field in required_fields if field not in body]
    assert not missing_fields, (
        f"{endpoint}: missing required field(s): {', '.join(missing_fields)}. "
        f"Actual fields: {sorted(body.keys())}"
    )


def assert_non_empty_string_field(
    body: dict[str, Any], field_name: str, *, endpoint: str
) -> None:
    value = body.get(field_name)
    assert isinstance(value, str), (
        f"{endpoint}: expected field {field_name!r} to be str, "
        f"got {type(value).__name__}"
    )
    assert value.strip(), f"{endpoint}: expected field {field_name!r} to be non-empty"


def assert_latency_within(
    response: httpx.Response, max_seconds: float, *, endpoint: str
) -> None:
    actual_seconds = response.elapsed.total_seconds()
    assert actual_seconds <= max_seconds, (
        f"{endpoint}: expected latency <= {max_seconds:.3f}s, "
        f"got {actual_seconds:.3f}s"
    )
