from __future__ import annotations

import pytest

from security.assertions.security_assertions import (
    assert_no_internal_details,
    assert_no_sensitive_fields,
    assert_status,
)


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


def test_passenger_response_does_not_expose_sensitive_fields(security_client, auth_headers):
    response = security_client.get("/airline/passengers/PAX-SEC-0001", headers=auth_headers)

    assert_status(response, 200)
    assert_no_sensitive_fields(response.json())


def test_booking_response_is_limited_to_expected_business_fields(
    security_client,
    auth_headers,
):
    response = security_client.get("/airline/bookings/BKG-SEC-0001", headers=auth_headers)

    assert_status(response, 200)
    assert set(response.json()) == {
        "booking_id",
        "pnr",
        "status",
        "flight",
        "passengers",
        "fare_id",
    }
    assert_no_sensitive_fields(response.json())


def test_unknown_resource_error_does_not_leak_internals(security_client, auth_headers):
    response = security_client.get("/airline/bookings/BKG-NOT-FOUND", headers=auth_headers)

    assert response.status_code == 404
    assert_no_internal_details(response.text)
