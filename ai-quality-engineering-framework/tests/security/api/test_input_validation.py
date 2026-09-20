from __future__ import annotations

import pytest

from security.assertions.security_assertions import assert_bad_request


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "first_name": "",
            "last_name": "Stone",
            "passenger_type": "ADULT",
            "date_of_birth": "1988-04-12",
        },
        {
            "first_name": None,
            "last_name": "Stone",
            "passenger_type": "ADULT",
            "date_of_birth": "1988-04-12",
        },
        {
            "first_name": "Avery",
            "last_name": "Stone",
            "passenger_type": "ROBOT",
            "date_of_birth": "1988-04-12",
        },
        {
            "first_name": "Avery",
            "last_name": "Stone",
            "passenger_type": "ADULT",
            "date_of_birth": "not-a-date",
        },
    ],
)
def test_invalid_passenger_shapes_are_rejected(security_client, auth_headers, payload):
    response = security_client.post(
        "/airline/passengers",
        headers=auth_headers,
        json=payload,
    )

    assert_bad_request(response)


def test_unexpected_passenger_fields_are_rejected(security_client, auth_headers):
    response = security_client.post(
        "/airline/passengers",
        headers=auth_headers,
        json={
            "first_name": "Avery",
            "last_name": "Stone",
            "passenger_type": "ADULT",
            "date_of_birth": "1988-04-12",
            "is_admin": True,
        },
    )

    assert_bad_request(response)


def test_oversized_bounded_input_is_rejected(
    security_client,
    auth_headers,
    security_payloads,
):
    response = security_client.post(
        "/airline/passengers",
        headers=auth_headers,
        json={
            "first_name": security_payloads["bounded_oversized_string"],
            "last_name": "Stone",
            "passenger_type": "ADULT",
            "date_of_birth": "1988-04-12",
        },
    )

    assert_bad_request(response)


def test_flight_search_passenger_count_boundary_abuse_is_rejected(
    security_client,
    auth_headers,
):
    response = security_client.post(
        "/airline/flights/search",
        headers=auth_headers,
        json={
            "origin": "JFK",
            "destination": "LHR",
            "departure_date": "2099-06-15",
            "passengers": 100,
            "cabin": "ECONOMY",
            "trip_type": "ONE_WAY",
        },
    )

    assert_bad_request(response)
