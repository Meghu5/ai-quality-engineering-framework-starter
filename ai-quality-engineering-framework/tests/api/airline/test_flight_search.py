import pytest

from models.airline.flight import FlightSearchResponse
from utils.airline_assertions import assert_valid_flight
from utils.assertions import assert_json_response, assert_status_code


ENDPOINT = "/airline/flights/search"


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.flight
def test_flight_search_returns_matching_available_flights(
    flight_client,
    flight_search_request,
):
    response = flight_client.search(flight_search_request)

    assert_status_code(response, 200, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    search_response = FlightSearchResponse.model_validate(body)

    assert search_response.flights, f"{ENDPOINT}: expected at least one matching flight"
    for flight in search_response.flights:
        assert_valid_flight(
            flight,
            origin=flight_search_request.origin,
            destination=flight_search_request.destination,
        )
        assert flight.cabin == flight_search_request.cabin


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.flight
@pytest.mark.parametrize(
    ("payload_update", "expected_error"),
    [
        ({"origin": "ZZZ"}, "origin airport is not supported"),
        ({"origin": "JFK", "destination": "JFK"}, "origin and destination must be different"),
        ({"departure_date": "2000-01-01"}, "departure date must not be in the past"),
        ({"passengers": 0}, "greater than or equal to 1"),
        ({"passengers": 10}, "less than or equal to 9"),
        ({"cabin": "MOON_DECK"}, "Input should be"),
    ],
)
def test_flight_search_rejects_invalid_business_requests(
    flight_client,
    flight_search_request,
    payload_update,
    expected_error,
):
    payload = flight_search_request.model_dump(mode="json")
    payload.update(payload_update)

    response = flight_client.search_raw(payload)

    assert_status_code(response, 422, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    assert expected_error in body["error"], (
        f"{ENDPOINT}: expected error containing {expected_error!r}, got {body}"
    )
