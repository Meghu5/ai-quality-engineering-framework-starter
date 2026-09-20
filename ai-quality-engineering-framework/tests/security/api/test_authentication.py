from __future__ import annotations

import pytest

from security.assertions.security_assertions import (
    assert_security_headers,
    assert_status,
    assert_unauthorized,
)
from security.config.security_config import SECURITY_IDENTITIES


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


def test_missing_authentication_is_rejected(security_client):
    response = security_client.get("/airline/passengers/PAX-SEC-0001")

    assert_unauthorized(response)
    assert_security_headers(response)


def test_malformed_authentication_is_rejected(security_client):
    response = security_client.get(
        "/airline/passengers/PAX-SEC-0001",
        headers={"Authorization": "Token valid-user-token"},
    )

    assert_unauthorized(response)


def test_invalid_token_is_rejected(security_client):
    response = security_client.get(
        "/airline/passengers/PAX-SEC-0001",
        headers={"Authorization": f"Bearer {SECURITY_IDENTITIES.invalid_token}"},
    )

    assert_unauthorized(response)


def test_expired_token_is_rejected_deterministically(security_client):
    response = security_client.get(
        "/airline/passengers/PAX-SEC-0001",
        headers={"Authorization": f"Bearer {SECURITY_IDENTITIES.expired_token}"},
    )

    assert_unauthorized(response)


def test_valid_token_can_access_owned_resource(security_client, auth_headers):
    response = security_client.get("/airline/passengers/PAX-SEC-0001", headers=auth_headers)

    assert_status(response, 200)
    assert response.json()["passenger_id"] == "PAX-SEC-0001"
