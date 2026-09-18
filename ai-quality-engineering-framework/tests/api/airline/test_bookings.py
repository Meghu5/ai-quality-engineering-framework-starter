import pytest

from models.airline.booking import BookingRequest, BookingResponse
from utils.airline_assertions import assert_valid_booking
from utils.assertions import assert_json_response, assert_status_code


ENDPOINT = "/airline/bookings"


@pytest.fixture
def created_passenger_id(passenger_client, passenger_request) -> str:
    response = passenger_client.create(passenger_request)
    assert_status_code(response, 201, endpoint="/airline/passengers")
    return response.json()["passenger_id"]


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.booking
def test_booking_creation_generates_pnr_and_links_passenger_and_itinerary(
    booking_client,
    created_passenger_id,
):
    request = BookingRequest(
        flight_id="FL-AIQ-100",
        fare_id="FARE-AIQ-100-E",
        passenger_ids=[created_passenger_id],
    )

    response = booking_client.create(request)

    assert_status_code(response, 201, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    booking = BookingResponse.model_validate(body)
    assert_valid_booking(booking)
    assert booking.flight.flight_id == request.flight_id
    assert booking.fare_id == request.fare_id
    assert [passenger.passenger_id for passenger in booking.passengers] == request.passenger_ids


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.booking
def test_booking_rejects_unknown_passenger(booking_client):
    response = booking_client.create_raw(
        {
            "flight_id": "FL-AIQ-100",
            "fare_id": "FARE-AIQ-100-E",
            "passenger_ids": ["PAX-9999"],
        }
    )

    assert_status_code(response, 404, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    assert body["error"] == "passenger not found"


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.booking
def test_booking_rejects_duplicate_passenger_ids(booking_client, created_passenger_id):
    response = booking_client.create_raw(
        {
            "flight_id": "FL-AIQ-100",
            "fare_id": "FARE-AIQ-100-E",
            "passenger_ids": [created_passenger_id, created_passenger_id],
        }
    )

    assert_status_code(response, 422, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    assert "passenger_ids must be unique" in body["error"]


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.booking
def test_booking_rejects_unavailable_flight(booking_client, created_passenger_id):
    response = booking_client.create_raw(
        {
            "flight_id": "FL-AIQ-200",
            "fare_id": "FARE-AIQ-200-B",
            "passenger_ids": [created_passenger_id],
        }
    )

    assert_status_code(response, 409, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    assert body["error"] == "flight is unavailable"
