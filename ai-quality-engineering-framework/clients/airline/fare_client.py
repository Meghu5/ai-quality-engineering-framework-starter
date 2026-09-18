from __future__ import annotations

import httpx

from clients.api_client import ApiClient


class FareClient:
    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client

    def get(self, fare_id: str) -> httpx.Response:
        return self.api_client.get(f"/airline/fares/{fare_id}")
