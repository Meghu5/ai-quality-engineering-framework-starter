from __future__ import annotations

import httpx

from clients.api_client import ApiClient
from models.airline.flight import FlightSearchRequest


class FlightClient:
    endpoint = "/airline/flights/search"

    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client

    def search(self, request: FlightSearchRequest) -> httpx.Response:
        return self.api_client.post(
            self.endpoint,
            json=request.model_dump(mode="json"),
        )

    def search_raw(self, payload: dict) -> httpx.Response:
        return self.api_client.post(self.endpoint, json=payload)
