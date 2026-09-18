from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Mapping

import httpx


logger = logging.getLogger(__name__)


class ApiClient:
    """Reusable HTTPX client for service-specific API clients."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
        auth_token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        default_headers = {"Accept": "application/json"}
        if headers:
            default_headers.update(headers)
        if auth_token:
            default_headers["Authorization"] = f"Bearer {auth_token}"

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=default_headers,
            transport=transport,
        )

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        logger.info("Sending %s request to %s", method.upper(), endpoint)
        started_at = time.perf_counter()
        try:
            response = self._client.request(
                method=method,
                url=endpoint,
                params=params,
                json=json,
                headers=headers,
            )
        except httpx.HTTPError:
            logger.exception("Request failed for %s %s", method.upper(), endpoint)
            raise

        elapsed_seconds = time.perf_counter() - started_at
        if not hasattr(response, "_elapsed"):
            response._elapsed = timedelta(seconds=elapsed_seconds)

        logger.info(
            "Received %s from %s %s in %.3fs",
            response.status_code,
            method.upper(),
            endpoint,
            response.elapsed.total_seconds(),
        )
        return response

    def get(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self.request("GET", endpoint, params=params, headers=headers)

    def post(
        self,
        endpoint: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self.request(
            "POST", endpoint, params=params, json=json, headers=headers
        )

    def put(
        self,
        endpoint: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self.request("PUT", endpoint, params=params, json=json, headers=headers)

    def patch(
        self,
        endpoint: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self.request(
            "PATCH", endpoint, params=params, json=json, headers=headers
        )

    def delete(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        return self.request("DELETE", endpoint, params=params, headers=headers)

    def close(self) -> None:
        self._client.close()
