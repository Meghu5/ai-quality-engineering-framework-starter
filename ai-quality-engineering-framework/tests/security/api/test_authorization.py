from __future__ import annotations

import pytest

from security.assertions.security_assertions import assert_forbidden, assert_status


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


def test_user_cannot_read_another_passenger(security_client, auth_headers):
    response = security_client.get("/airline/passengers/PAX-SEC-0002", headers=auth_headers)

    assert_forbidden(response)


def test_user_cannot_read_another_booking(security_client, auth_headers):
    response = security_client.get("/airline/bookings/BKG-SEC-0002", headers=auth_headers)

    assert_forbidden(response)


def test_admin_can_read_cross_user_booking(security_client, admin_headers):
    response = security_client.get(
        "/airline/bookings/BKG-SEC-0002",
        headers=admin_headers,
    )

    assert_status(response, 200)
    assert response.json()["booking_id"] == "BKG-SEC-0002"


def test_booking_creation_cannot_reference_another_users_passenger(
    security_client,
    auth_headers,
):
    response = security_client.post(
        "/airline/bookings",
        headers=auth_headers,
        json={
            "flight_id": "FL-AIQ-100",
            "fare_id": "FARE-AIQ-100-E",
            "passenger_ids": ["PAX-SEC-0002"],
        },
    )

    assert_forbidden(response)
