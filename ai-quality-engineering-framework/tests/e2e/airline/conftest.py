from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Page

from clients.airline import (
    AvailabilityClient,
    BookingClient,
    FareClient,
    FlightClient,
    PassengerClient,
)
from clients.api_client import ApiClient
from pages.airline import (
    ConfirmationPage,
    FarePage,
    FlightResultsPage,
    HomePage,
    PassengerPage,
    ReviewBookingPage,
)
from utils.airline.deterministic_api import (
    deterministic_airline_transport,
    load_airline_json,
)
from utils.airline.e2e_helpers import AirlineE2EContext, build_e2e_context


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "airline"
AIRLINE_UI_APP = PROJECT_ROOT / "test_apps" / "airline_ui" / "index.html"


@pytest.fixture(scope="session")
def airline_flights() -> list[dict]:
    return load_airline_json(DATA_DIR, "flights.json")


@pytest.fixture(scope="session")
def airline_fares() -> list[dict]:
    return load_airline_json(DATA_DIR, "fares.json")


@pytest.fixture(scope="session")
def passenger_payloads() -> list[dict]:
    return load_airline_json(DATA_DIR, "passengers.json")


@pytest.fixture
def airline_api_client(settings, airline_flights, airline_fares):
    client = ApiClient(
        settings.api_base_url,
        timeout=settings.api_timeout_seconds,
        auth_token=settings.api_auth_token,
        transport=deterministic_airline_transport(airline_flights, airline_fares),
    )
    yield client
    client.close()


@pytest.fixture
def flight_client(airline_api_client):
    return FlightClient(airline_api_client)


@pytest.fixture
def availability_client(airline_api_client):
    return AvailabilityClient(airline_api_client)


@pytest.fixture
def fare_client(airline_api_client):
    return FareClient(airline_api_client)


@pytest.fixture
def passenger_client(airline_api_client):
    return PassengerClient(airline_api_client)


@pytest.fixture
def booking_client(airline_api_client):
    return BookingClient(airline_api_client)


@pytest.fixture
def e2e_context(
    airline_flights,
    airline_fares,
    passenger_payloads,
) -> AirlineE2EContext:
    return build_e2e_context(
        flights=airline_flights,
        fares=airline_fares,
        passenger_payload=passenger_payloads[0],
    )


@pytest.fixture
def airline_home_page(page: Page) -> HomePage:
    page.set_default_timeout(5000)
    home_page = HomePage(page)
    home_page.open(AIRLINE_UI_APP)
    return home_page


@pytest.fixture
def flight_results_page(page: Page) -> FlightResultsPage:
    return FlightResultsPage(page)


@pytest.fixture
def fare_page(page: Page) -> FarePage:
    return FarePage(page)


@pytest.fixture
def passenger_page(page: Page) -> PassengerPage:
    return PassengerPage(page)


@pytest.fixture
def review_booking_page(page: Page) -> ReviewBookingPage:
    return ReviewBookingPage(page)


@pytest.fixture
def confirmation_page(page: Page) -> ConfirmationPage:
    return ConfirmationPage(page)
