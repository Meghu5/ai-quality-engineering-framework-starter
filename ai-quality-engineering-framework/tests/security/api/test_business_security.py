from __future__ import annotations

import pytest

from security.assertions.security_assertions import assert_bad_request, assert_status


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


def test_duplicate_passenger_ids_cannot_create_booking(security_client, auth_headers):
    response = security_client.post(
        "/airline/bookings",
        headers=auth_headers,
        json={
            "flight_id": "FL-AIQ-100",
            "fare_id": "FARE-AIQ-100-E",
            "passenger_ids": ["PAX-SEC-0001", "PAX-SEC-0001"],
        },
    )

    assert_bad_request(response)


def test_invalid_state_transition_is_not_exposed_as_supported_operation(
    security_client,
    auth_headers,
):
    response = security_client.patch(
        "/airline/bookings/BKG-SEC-0001",
        headers=auth_headers,
        json={"status": "CANCELLED"},
    )

    assert_status(response, 405)


def test_rate_limit_probe_is_deterministic(security_client, auth_headers):
    statuses = [
        security_client.get(
            "/airline/security/rate-limit-probe",
            headers=auth_headers,
        ).status_code
        for _ in range(4)
    ]

    assert statuses == [200, 200, 200, 429]
