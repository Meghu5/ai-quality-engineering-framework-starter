from __future__ import annotations

import httpx

from clients.api_client import ApiClient


class AvailabilityClient:
    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client

    def get(self, flight_id: str) -> httpx.Response:
        return self.api_client.get(f"/airline/flights/{flight_id}/availability")
