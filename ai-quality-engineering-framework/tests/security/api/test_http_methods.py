from __future__ import annotations

import pytest

from security.assertions.security_assertions import assert_status


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "OPTIONS"])
def test_unsupported_resource_methods_are_not_allowed(
    security_client,
    auth_headers,
    method,
):
    response = security_client.request(
        method,
        "/airline/passengers/PAX-SEC-0001",
        headers=auth_headers,
    )

    assert_status(response, 405)


def test_unknown_endpoint_returns_not_found_without_internals(security_client, auth_headers):
    response = security_client.get("/airline/admin/debug", headers=auth_headers)

    assert_status(response, 404)
    assert "Traceback" not in response.text
