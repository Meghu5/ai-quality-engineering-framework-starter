from __future__ import annotations

import httpx

from clients.api_client import ApiClient


DUFFEL_VERSION = "v2"


class DuffelClient:
    """Domain-oriented client for Duffel TEST/SANDBOX airline APIs."""

    offer_requests_endpoint = "/air/offer_requests"
    offers_endpoint = "/air/offers"
    orders_endpoint = "/air/orders"

    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client

    def create_offer_request(
        self,
        payload: dict,
        *,
        return_offers: bool = True,
    ) -> httpx.Response:
        return self.api_client.post(
            self.offer_requests_endpoint,
            params={"return_offers": str(return_offers).lower()},
            json={"data": payload},
        )

    def get_offers(
        self,
        *,
        offer_request_id: str,
        limit: int = 5,
        sort: str | None = None,
        max_connections: int | None = None,
    ) -> httpx.Response:
        params: dict[str, str | int] = {
            "offer_request_id": offer_request_id,
            "limit": limit,
        }
        if sort is not None:
            params["sort"] = sort
        if max_connections is not None:
            params["max_connections"] = max_connections

        return self.api_client.get(self.offers_endpoint, params=params)

    def get_offer(
        self,
        offer_id: str,
        *,
        return_available_services: bool = False,
    ) -> httpx.Response:
        return self.api_client.get(
            f"{self.offers_endpoint}/{offer_id}",
            params={
                "return_available_services": str(return_available_services).lower()
            },
        )

    def create_order(self, payload: dict) -> httpx.Response:
        return self.api_client.post(self.orders_endpoint, json={"data": payload})
