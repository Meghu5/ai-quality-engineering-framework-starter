from __future__ import annotations

import pytest

from security.assertions.security_assertions import assert_security_headers


pytestmark = [pytest.mark.api, pytest.mark.airline, pytest.mark.security]


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/airline/passengers/PAX-SEC-0001",
        "/airline/bookings/BKG-SEC-0001",
    ],
)
def test_security_headers_are_present(security_client, auth_headers, path):
    headers = auth_headers if path.startswith("/airline/") else None
    response = security_client.get(path, headers=headers)

    assert_security_headers(response)
