from __future__ import annotations

import pytest

from security.assertions.security_assertions import (
    assert_bad_request,
    assert_no_internal_details,
)


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


@pytest.mark.parametrize("field_name", ["first_name", "last_name"])
def test_injection_like_passenger_names_are_safe_inputs(
    security_client,
    auth_headers,
    security_payloads,
    field_name,
):
    for payload in security_payloads["injection_strings"]:
        passenger = {
            "first_name": "Avery",
            "last_name": "Stone",
            "passenger_type": "ADULT",
            "date_of_birth": "1988-04-12",
            field_name: payload,
        }

        response = security_client.post(
            "/airline/passengers",
            headers=auth_headers,
            json=passenger,
        )

        assert response.status_code in {201, 422}
        assert_no_internal_details(response.text)


@pytest.mark.parametrize("resource_id", ["../../etc/passwd", "PAX-SEC-0002?owner=user-001"])
def test_path_manipulation_does_not_bypass_authorization(
    security_client,
    auth_headers,
    resource_id,
):
    response = security_client.get(f"/airline/passengers/{resource_id}", headers=auth_headers)

    assert response.status_code in {403, 404}
    assert_no_internal_details(response.text)


def test_malformed_json_returns_safe_validation_error(security_client, auth_headers):
    response = security_client.post(
        "/airline/passengers",
        headers={**auth_headers, "Content-Type": "application/json"},
        content="{not-json",
    )

    assert_bad_request(response)
