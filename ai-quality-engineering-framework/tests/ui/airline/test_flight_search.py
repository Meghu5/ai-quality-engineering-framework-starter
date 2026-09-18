import pytest
from playwright.sync_api import expect

from utils.airline.ui_assertions import (
    assert_search_results_for_route,
    assert_selected_flight,
)


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.flight
def test_valid_one_way_flight_search(
    airline_home_page,
    flight_results_page,
    search_cases,
):
    search = search_cases["valid_one_way"]

    airline_home_page.search(
        origin=search["origin"],
        destination=search["destination"],
        departure_date=search["departure_date"],
        passengers=search["passengers"],
    )

    assert_search_results_for_route(
        flight_results_page,
        origin=search["origin"],
        destination=search["destination"],
    )


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.flight
@pytest.mark.parametrize(
    "case_name",
    [
        "missing_origin",
        "missing_destination",
        "same_origin_destination",
        "past_departure",
        "invalid_passenger_count",
    ],
)
def test_flight_search_validation_errors(airline_home_page, search_cases, case_name):
    search = search_cases[case_name]

    airline_home_page.search(
        origin=search["origin"],
        destination=search["destination"],
        departure_date=search["departure_date"],
        passengers=search["passengers"],
    )

    expect(airline_home_page.error).to_have_text(search["expected_error"])


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.flight
def test_select_available_flight(
    searched_flight_results,
    fare_page,
    search_cases,
    booking_cases,
):
    search = search_cases["valid_one_way"]
    booking = booking_cases["standard_booking"]

    searched_flight_results.select_flight(booking["flight_id"])
    searched_flight_results.continue_to_fares()

    assert_selected_flight(
        fare_page,
        flight_id=booking["flight_id"],
        origin=search["origin"],
        destination=search["destination"],
    )


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.flight
def test_continue_without_selecting_flight_shows_validation(searched_flight_results):
    searched_flight_results.continue_to_fares()

    expect(searched_flight_results.error).to_have_text("Select a flight to continue")
