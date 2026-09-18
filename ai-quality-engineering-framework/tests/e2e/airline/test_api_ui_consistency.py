import pytest
from playwright.sync_api import expect

from models.airline.fare import FareResponse
from models.airline.flight import FlightSearchResponse
from utils.airline.e2e_helpers import (
    assert_booking_matches_api_and_ui,
    assert_fare_consistent,
    assert_flight_consistent,
    assert_passenger_consistent,
    complete_booking_via_ui,
    create_booking_via_api,
    fare_display_name,
)
from utils.assertions import assert_json_response, assert_status_code


@pytest.mark.e2e
@pytest.mark.airline
@pytest.mark.flight
def test_api_ui_flight_consistency(
    flight_client,
    e2e_context,
    airline_home_page,
    flight_results_page,
):
    response = flight_client.search(e2e_context.search_request)
    assert_status_code(response, 200, endpoint="/airline/flights/search")
    search_response = FlightSearchResponse.model_validate(
        assert_json_response(response, endpoint="/airline/flights/search")
    )

    airline_home_page.search(
        origin=e2e_context.flight.origin,
        destination=e2e_context.flight.destination,
        departure_date=e2e_context.search_request.departure_date.isoformat(),
        passengers=e2e_context.passengers,
    )

    matching_flight_ids = {flight.flight_id for flight in search_response.flights}
    assert e2e_context.flight.flight_id in matching_flight_ids
    expect(flight_results_page.results).to_contain_text(e2e_context.flight.flight_id)
    expect(flight_results_page.results).to_contain_text(e2e_context.route)


@pytest.mark.e2e
@pytest.mark.airline
@pytest.mark.fare
def test_api_ui_fare_consistency(
    fare_client,
    e2e_context,
    airline_home_page,
    flight_results_page,
    fare_page,
):
    response = fare_client.get(e2e_context.fare.fare_id)
    assert_status_code(
        response,
        200,
        endpoint=f"/airline/fares/{e2e_context.fare.fare_id}",
    )
    api_fare = FareResponse.model_validate(
        assert_json_response(
            response,
            endpoint=f"/airline/fares/{e2e_context.fare.fare_id}",
        )
    )

    airline_home_page.search(
        origin=e2e_context.flight.origin,
        destination=e2e_context.flight.destination,
        departure_date=e2e_context.search_request.departure_date.isoformat(),
        passengers=e2e_context.passengers,
    )
    flight_results_page.select_flight(e2e_context.flight.flight_id)
    flight_results_page.continue_to_fares()

    expect(fare_page.fare_options).to_contain_text(api_fare.fare_id)
    expect(fare_page.fare_options).to_contain_text(api_fare.currency)
    expect(fare_page.fare_options).to_contain_text(f"{api_fare.total_amount:.2f}")
    expect(fare_page.fare_options).to_contain_text(fare_display_name(api_fare))


@pytest.mark.e2e
@pytest.mark.airline
@pytest.mark.passenger
@pytest.mark.booking
def test_api_ui_passenger_pnr_and_itinerary_consistency(
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

    assert_flight_consistent(api_booking=api_booking, ui_booking=ui_booking)
    assert_fare_consistent(api_fare=e2e_context.fare, ui_booking=ui_booking)
    assert_passenger_consistent(api_booking=api_booking, ui_booking=ui_booking)
    assert_booking_matches_api_and_ui(
        context=e2e_context,
        api_booking=api_booking,
        ui_booking=ui_booking,
    )
