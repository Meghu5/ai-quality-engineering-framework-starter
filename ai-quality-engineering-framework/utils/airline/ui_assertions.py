from __future__ import annotations

import re

from playwright.sync_api import expect

from pages.airline.confirmation_page import ConfirmationPage
from pages.airline.fare_page import FarePage
from pages.airline.flight_results_page import FlightResultsPage
from pages.airline.review_booking_page import ReviewBookingPage


def assert_search_results_for_route(
    results_page: FlightResultsPage,
    *,
    origin: str,
    destination: str,
) -> None:
    expected_route = f"{origin} to {destination}"
    expect(results_page.results_view).to_be_visible()
    expect(results_page.route).to_have_text(expected_route)
    expect(results_page.results).to_contain_text(expected_route)


def assert_selected_flight(
    fare_page: FarePage,
    *,
    flight_id: str,
    origin: str,
    destination: str,
) -> None:
    expect(fare_page.fare_view).to_be_visible()
    expect(fare_page.flight_summary).to_contain_text(flight_id)
    expect(fare_page.flight_summary).to_contain_text(f"{origin} to {destination}")


def assert_review_summary(
    review_page: ReviewBookingPage,
    *,
    route: str,
    flight_id: str,
    fare_name: str,
    passenger_name: str,
) -> None:
    expect(review_page.review_view).to_be_visible()
    expect(review_page.itinerary).to_contain_text(route)
    expect(review_page.flight).to_contain_text(flight_id)
    expect(review_page.fare).to_contain_text(fare_name)
    expect(review_page.passenger).to_contain_text(passenger_name)
    expect(review_page.total).to_contain_text("AED")


def assert_confirmation(
    confirmation_page: ConfirmationPage,
    *,
    pnr_pattern: str,
    itinerary: str,
    passenger_name: str,
    fare_name: str,
) -> None:
    expect(confirmation_page.confirmation_view).to_be_visible()
    expect(confirmation_page.message).to_have_text("Your booking is confirmed.")
    pnr = confirmation_page.pnr_text()
    assert re.fullmatch(pnr_pattern, pnr), (
        f"Expected PNR to match {pnr_pattern}, got {pnr}"
    )
    expect(confirmation_page.itinerary).to_have_text(itinerary)
    expect(confirmation_page.passenger).to_have_text(passenger_name)
    expect(confirmation_page.fare).to_contain_text(fare_name)
