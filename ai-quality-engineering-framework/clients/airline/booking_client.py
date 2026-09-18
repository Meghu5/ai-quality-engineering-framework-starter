from __future__ import annotations

import httpx

from clients.api_client import ApiClient
from models.airline.booking import BookingRequest


class BookingClient:
    endpoint = "/airline/bookings"

    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client

    def create(self, request: BookingRequest) -> httpx.Response:
        return self.api_client.post(
            self.endpoint,
            json=request.model_dump(mode="json"),
        )

    def create_raw(self, payload: dict) -> httpx.Response:
        return self.api_client.post(self.endpoint, json=payload)
