import pytest
from playwright.sync_api import expect


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.fare
def test_select_valid_fare(
    selected_flight_fare_page,
    passenger_page,
    booking_cases,
):
    booking = booking_cases["standard_booking"]

    selected_flight_fare_page.select_fare(booking["fare_id"])
    selected_flight_fare_page.continue_to_passenger()

    expect(passenger_page.passenger_view).to_be_visible()


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.fare
def test_continue_without_selecting_fare_shows_validation(selected_flight_fare_page):
    selected_flight_fare_page.continue_to_passenger()

    expect(selected_flight_fare_page.error).to_have_text("Select a fare to continue")


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.passenger
def test_valid_passenger_details_reach_review(
    passenger_entry_page,
    review_booking_page,
    passenger_cases,
):
    passenger = passenger_cases["valid_adult"]

    passenger_entry_page.enter_passenger(
        first_name=passenger["first_name"],
        last_name=passenger["last_name"],
        date_of_birth=passenger["date_of_birth"],
        passenger_type=passenger["passenger_type"],
    )
    passenger_entry_page.continue_to_review()

    expect(review_booking_page.review_view).to_be_visible()
    expect(review_booking_page.passenger).to_contain_text(
        f"{passenger['first_name']} {passenger['last_name']}"
    )


@pytest.mark.ui
@pytest.mark.airline
@pytest.mark.passenger
@pytest.mark.parametrize(
    "case_name",
    ["missing_first_name", "missing_last_name", "invalid_name"],
)
def test_passenger_validation_errors(
    passenger_entry_page,
    passenger_cases,
    case_name,
):
    passenger = passenger_cases[case_name]

    passenger_entry_page.enter_passenger(
        first_name=passenger["first_name"],
        last_name=passenger["last_name"],
        date_of_birth=passenger["date_of_birth"],
        passenger_type=passenger["passenger_type"],
    )
    passenger_entry_page.continue_to_review()

    expect(passenger_entry_page.error).to_have_text(passenger["expected_error"])
