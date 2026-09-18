import pytest

from utils.airline.e2e_helpers import (
    assert_booking_matches_api_and_ui,
    complete_booking_via_ui,
    create_booking_via_api,
)


@pytest.mark.e2e
@pytest.mark.airline
@pytest.mark.booking
def test_api_prepared_booking_is_validated_in_ui(
    passenger_client,
    booking_client,
    e2e_context,
    airline_home_page,
    flight_results_page,
    fare_page,
    passenger_page,
    review_booking_page,
    confirmation_page,
):
    api_booking = create_booking_via_api(
        passenger_client=passenger_client,
        booking_client=booking_client,
        context=e2e_context,
    )

    ui_booking = complete_booking_via_ui(
        context=e2e_context,
        home_page=airline_home_page,
        flight_results_page=flight_results_page,
        fare_page=fare_page,
        passenger_page=passenger_page,
        review_page=review_booking_page,
        confirmation_page=confirmation_page,
    )

    assert_booking_matches_api_and_ui(
        context=e2e_context,
        api_booking=api_booking,
        ui_booking=ui_booking,
    )
