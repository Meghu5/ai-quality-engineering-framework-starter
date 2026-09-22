from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from observability.context import get_context
from observability.models import FailureCategory
from observability.tracing import TracingFacade, create_tracing_facade


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
        tracer: TracingFacade | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.tracer = tracer or create_tracing_facade()
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
        request_headers = dict(headers or {})
        correlation_id = _header_value(request_headers, "x-correlation-id")
        span_factory = (
            self.tracer.start_span if get_context() else self.tracer.start_trace
        )
        span_kwargs: dict[str, Any] = {
            "operation_type": "http.client",
            "attributes": {
                "http_method": method.upper(),
                **_safe_url_attributes(self.base_url, endpoint),
                "timeout_seconds": self.timeout,
            },
        }
        if span_factory == self.tracer.start_trace:
            span_kwargs["correlation_id"] = correlation_id

        with span_factory(f"http.{method.lower()}", **span_kwargs) as span:
            _inject_trace_headers(request_headers)
            try:
                response = self._client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=json,
                    headers=request_headers or None,
                )
            except httpx.HTTPError as exc:
                category = (
                    FailureCategory.TIMEOUT
                    if isinstance(exc, httpx.TimeoutException)
                    else FailureCategory.NETWORK
                )
                span.record_exception(exc, category)
                logger.exception("Request failed for %s %s", method.upper(), endpoint)
                raise

            span.set_attribute("status_code", response.status_code)
            span.set_attribute(
                "latency_ms", (time.perf_counter() - started_at) * 1000
            )

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


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    return next((value for key, value in headers.items() if key.lower() == name), None)


def _inject_trace_headers(headers: dict[str, str]) -> None:
    context = get_context()
    if context is None:
        return
    existing = {key.lower() for key in headers}
    if "x-trace-id" not in existing:
        headers["X-Trace-ID"] = context.trace_id
    if "x-correlation-id" not in existing:
        headers["X-Correlation-ID"] = context.correlation_id


def _safe_url_attributes(base_url: str, endpoint: str) -> dict[str, str]:
    base = urlsplit(base_url)
    path = urlsplit(endpoint).path or "/"
    return {"http_host": base.hostname or "", "http_path": path}
