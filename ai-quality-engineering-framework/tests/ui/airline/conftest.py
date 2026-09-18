from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page

from pages.airline import (
    ConfirmationPage,
    FarePage,
    FlightResultsPage,
    HomePage,
    PassengerPage,
    ReviewBookingPage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AIRLINE_UI_APP = PROJECT_ROOT / "test_apps" / "airline_ui" / "index.html"
UI_DATA_DIR = PROJECT_ROOT / "data" / "airline" / "ui"


def _load_json(name: str) -> dict[str, Any]:
    with (UI_DATA_DIR / name).open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def airline_ui_app_path() -> Path:
    return AIRLINE_UI_APP


@pytest.fixture(scope="session")
def search_cases() -> dict[str, Any]:
    return _load_json("search_cases.json")


@pytest.fixture(scope="session")
def passenger_cases() -> dict[str, Any]:
    return _load_json("passenger_cases.json")


@pytest.fixture(scope="session")
def booking_cases() -> dict[str, Any]:
    return _load_json("booking_cases.json")


@pytest.fixture
def airline_home_page(page: Page, airline_ui_app_path: Path) -> HomePage:
    page.set_default_timeout(5000)
    home_page = HomePage(page)
    home_page.open(airline_ui_app_path)
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


@pytest.fixture
def searched_flight_results(
    airline_home_page: HomePage,
    flight_results_page: FlightResultsPage,
    search_cases: dict[str, Any],
) -> FlightResultsPage:
    valid_search = search_cases["valid_one_way"]
    airline_home_page.search(
        origin=valid_search["origin"],
        destination=valid_search["destination"],
        departure_date=valid_search["departure_date"],
        passengers=valid_search["passengers"],
    )
    return flight_results_page


@pytest.fixture
def selected_flight_fare_page(
    searched_flight_results: FlightResultsPage,
    fare_page: FarePage,
    booking_cases: dict[str, Any],
) -> FarePage:
    booking = booking_cases["standard_booking"]
    searched_flight_results.select_flight(booking["flight_id"])
    searched_flight_results.continue_to_fares()
    return fare_page


@pytest.fixture
def passenger_entry_page(
    selected_flight_fare_page: FarePage,
    passenger_page: PassengerPage,
    booking_cases: dict[str, Any],
) -> PassengerPage:
    booking = booking_cases["standard_booking"]
    selected_flight_fare_page.select_fare(booking["fare_id"])
    selected_flight_fare_page.continue_to_passenger()
    return passenger_page
