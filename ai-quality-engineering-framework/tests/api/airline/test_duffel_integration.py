from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from clients.airline import DUFFEL_VERSION, DuffelClient
from clients.api_client import ApiClient
from models.airline.duffel import (
    DuffelApiResponse,
    DuffelErrorResponse,
    DuffelListResponse,
    DuffelOffer,
    DuffelOfferRequest,
    DuffelOrder,
)
from utils.airline.duffel_mapping import (
    duffel_offer_to_fare,
    duffel_offer_to_flight,
)
from utils.airline_assertions import assert_valid_fare, assert_valid_flight
from utils.assertions import assert_json_response


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "airline" / "real_api"
DUFFEL_TOKEN_SKIP_REASON = "DUFFEL_ACCESS_TOKEN not configured"

pytestmark = [
    pytest.mark.api,
    pytest.mark.airline,
    pytest.mark.real_api,
]


@pytest.fixture(scope="session")
def duffel_scenarios() -> dict[str, Any]:
    with (DATA_DIR / "duffel_scenarios.json").open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def duffel_client(settings):
    if not settings.duffel_access_token:
        pytest.skip(DUFFEL_TOKEN_SKIP_REASON)

    client = ApiClient(
        settings.duffel_base_url,
        timeout=settings.duffel_timeout_seconds,
        auth_token=settings.duffel_access_token,
        headers={
            "Content-Type": "application/json",
            "Duffel-Version": DUFFEL_VERSION,
        },
    )
    yield DuffelClient(client)
    client.close()


@pytest.fixture
def offer_request_payload_factory():
    def build(scenario: dict[str, Any]) -> dict[str, Any]:
        departure_date = date.today() + timedelta(
            days=scenario["departure_days_from_today"]
        )
        return {
            "cabin_class": scenario["cabin_class"],
            "passengers": [{"type": scenario["passenger_type"]}],
            "slices": [
                {
                    "origin": scenario["origin"],
                    "destination": scenario["destination"],
                    "departure_date": departure_date.isoformat(),
                }
            ],
        }

    return build


@pytest.fixture
def duffel_offer_request(
    duffel_client,
    duffel_scenarios,
    offer_request_payload_factory,
) -> DuffelOfferRequest:
    response = duffel_client.create_offer_request(
        offer_request_payload_factory(duffel_scenarios["search_with_offers"])
    )
    assert response.status_code in {200, 201}, (
        "POST /air/offer_requests expected success, got "
        f"{response.status_code}: {response.text}"
    )
    body = assert_json_response(response, endpoint="/air/offer_requests")
    return DuffelOfferRequest.model_validate(DuffelApiResponse.model_validate(body).data)


@pytest.fixture
def duffel_offer(duffel_client, duffel_offer_request) -> DuffelOffer:
    offers = duffel_offer_request.offers
    if not offers:
        response = duffel_client.get_offers(
            offer_request_id=duffel_offer_request.id,
            limit=5,
            max_connections=1,
        )
        assert response.status_code == 200, (
            "GET /air/offers expected success, got "
            f"{response.status_code}: {response.text}"
        )
        offers = DuffelListResponse.model_validate(
            assert_json_response(response, endpoint="/air/offers")
        ).data

    assert offers, (
        "Duffel TEST/SANDBOX offer request returned no offers for the "
        "search_with_offers scenario"
    )
    return offers[0]


def test_duffel_real_flight_search_returns_offer_request_and_offer_structure(
    duffel_offer_request,
):
    assert duffel_offer_request.id.startswith("orq_")
    assert duffel_offer_request.live_mode is False
    assert duffel_offer_request.passengers
    assert duffel_offer_request.slices

    if duffel_offer_request.offers:
        first_offer = duffel_offer_request.offers[0]
        assert first_offer.id.startswith("off_")
        assert first_offer.slices
        assert first_offer.passengers
        assert first_offer.total_amount >= 0
        assert first_offer.total_currency


def test_duffel_offer_can_be_retrieved_and_mapped_to_airline_domain(
    duffel_client,
    duffel_offer,
    duffel_scenarios,
):
    response = duffel_client.get_offer(duffel_offer.id)
    assert response.status_code == 200, (
        f"GET /air/offers/{duffel_offer.id} expected success, got "
        f"{response.status_code}: {response.text}"
    )
    body = assert_json_response(response, endpoint=f"/air/offers/{duffel_offer.id}")
    offer = DuffelOffer.model_validate(DuffelApiResponse.model_validate(body).data)

    flight = duffel_offer_to_flight(offer)
    fare = duffel_offer_to_fare(offer)
    scenario = duffel_scenarios["search_with_offers"]

    assert_valid_flight(
        flight,
        origin=scenario["origin"],
        destination=scenario["destination"],
    )
    assert_valid_fare(fare)
    assert offer.owner is None or offer.owner.name or offer.owner.iata_code
    assert offer.slices[0].segments


def test_duffel_official_no_flights_scenario_returns_no_offers(
    duffel_client,
    duffel_scenarios,
    offer_request_payload_factory,
):
    scenario = duffel_scenarios["no_flights"]
    response = duffel_client.create_offer_request(
        offer_request_payload_factory(scenario)
    )
    assert response.status_code in {200, 201}, (
        "POST /air/offer_requests no-flights scenario expected success, got "
        f"{response.status_code}: {response.text}"
    )
    body = assert_json_response(response, endpoint="/air/offer_requests")
    offer_request = DuffelOfferRequest.model_validate(
        DuffelApiResponse.model_validate(body).data
    )

    assert offer_request.live_mode is False
    assert offer_request.offers == [], (
        "Official Duffel TEST/SANDBOX No Flights scenario PVD -> RAI "
        "should return no offers"
    )


def test_duffel_test_mode_hold_order_booking_flow(
    duffel_client,
    duffel_scenarios,
    offer_request_payload_factory,
):
    response = duffel_client.create_offer_request(
        offer_request_payload_factory(duffel_scenarios["hold_order"])
    )
    assert response.status_code in {200, 201}, (
        "POST /air/offer_requests hold-order scenario expected success, got "
        f"{response.status_code}: {response.text}"
    )
    offer_request = DuffelOfferRequest.model_validate(
        DuffelApiResponse.model_validate(
            assert_json_response(response, endpoint="/air/offer_requests")
        ).data
    )
    assert offer_request.live_mode is False
    assert offer_request.offers

    hold_offer = next(
        (
            offer
            for offer in offer_request.offers
            if offer.payment_requirements
            and offer.payment_requirements.requires_instant_payment is False
        ),
        None,
    )
    assert hold_offer is not None, (
        "Official Duffel TEST/SANDBOX Hold Orders scenario JFK -> EWR did "
        "not return a hold-order-capable offer"
    )

    order_response = duffel_client.create_order(
        {
            "selected_offers": [hold_offer.id],
            "type": "hold",
            "passengers": [
                {
                    "id": _required_passenger_id(hold_offer),
                    **duffel_scenarios["test_passenger"],
                }
            ],
        }
    )
    if order_response.status_code >= 400:
        error_body = assert_json_response(order_response, endpoint="/air/orders")
        errors = DuffelErrorResponse.model_validate(error_body)
        pytest.fail(
            "Duffel TEST/SANDBOX hold order creation failed: "
            f"{[error.model_dump() for error in errors.errors]}"
        )

    assert order_response.status_code in {200, 201}
    order = DuffelOrder.model_validate(
        DuffelApiResponse.model_validate(
            assert_json_response(order_response, endpoint="/air/orders")
        ).data
    )

    assert order.id.startswith("ord_")
    assert order.booking_reference or order.id
    assert order.passengers
    assert order.slices
    assert order.total_amount >= 0
    assert order.total_currency
    passenger = order.passengers[0]
    assert passenger.given_name == duffel_scenarios["test_passenger"]["given_name"]
    assert passenger.family_name == duffel_scenarios["test_passenger"]["family_name"]


def _required_passenger_id(offer: DuffelOffer) -> str:
    passenger = offer.passengers[0] if offer.passengers else None
    assert passenger and passenger.id, "Duffel offer did not include passenger ID"
    return passenger.id
