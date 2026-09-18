import pytest

from utils.airline.ui_assertions import assert_confirmation, assert_review_summary


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.e2e
@pytest.mark.booking
def test_airline_booking_journey_generates_pnr_and_final_itinerary(
    airline_home_page,
    flight_results_page,
    fare_page,
    passenger_page,
    review_booking_page,
    confirmation_page,
    search_cases,
    passenger_cases,
    booking_cases,
):
    search = search_cases["valid_one_way"]
    passenger = passenger_cases["valid_adult"]
    booking = booking_cases["standard_booking"]
    passenger_name = f"{passenger['first_name']} {passenger['last_name']}"

    airline_home_page.search(
        origin=search["origin"],
        destination=search["destination"],
        departure_date=search["departure_date"],
        passengers=search["passengers"],
    )

    flight_results_page.select_flight(booking["flight_id"])
    flight_results_page.continue_to_fares()

    fare_page.select_fare(booking["fare_id"])
    fare_page.continue_to_passenger()

    passenger_page.enter_passenger(
        first_name=passenger["first_name"],
        last_name=passenger["last_name"],
        date_of_birth=passenger["date_of_birth"],
        passenger_type=passenger["passenger_type"],
    )
    passenger_page.continue_to_review()

    assert_review_summary(
        review_booking_page,
        route=search["expected_route"],
        flight_id=booking["flight_id"],
        fare_name=booking["fare_name"],
        passenger_name=passenger_name,
    )

    review_booking_page.confirm_booking()

    assert_confirmation(
        confirmation_page,
        pnr_pattern=booking["expected_pnr_pattern"],
        itinerary=booking["expected_itinerary"],
        passenger_name=passenger_name,
        fare_name=booking["fare_name"],
    )
