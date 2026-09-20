from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from performance.apps.deterministic_airline_service import create_server
from security.config.security_config import SECURITY_IDENTITIES


SECURITY_DATA_DIR = Path(__file__).resolve().parents[3] / "security" / "data"


@pytest.fixture(scope="session")
def security_payloads() -> dict[str, Any]:
    with (SECURITY_DATA_DIR / "payloads.json").open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def security_identities() -> dict[str, Any]:
    with (SECURITY_DATA_DIR / "identities.json").open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def security_service_base_url():
    server = create_server("127.0.0.1", 0, security_enabled=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def security_client(security_service_base_url):
    with httpx.Client(base_url=security_service_base_url, timeout=5.0) as client:
        yield client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SECURITY_IDENTITIES.valid_user_token}"}


@pytest.fixture
def another_user_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SECURITY_IDENTITIES.another_user_token}"}


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SECURITY_IDENTITIES.admin_token}"}
