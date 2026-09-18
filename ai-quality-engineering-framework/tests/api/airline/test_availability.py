import pytest

from models.airline.availability import AvailabilityResponse
from utils.airline_assertions import assert_valid_availability
from utils.assertions import assert_json_response, assert_status_code


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.availability
def test_availability_returns_valid_seat_count(availability_client):
    endpoint = "/airline/flights/FL-AIQ-100/availability"

    response = availability_client.get("FL-AIQ-100")

    assert_status_code(response, 200, endpoint=endpoint)
    body = assert_json_response(response, endpoint=endpoint)
    availability = AvailabilityResponse.model_validate(body)
    assert_valid_availability(availability)
    assert availability.available_seats > 0


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.availability
def test_availability_handles_unavailable_flight(availability_client):
    endpoint = "/airline/flights/FL-AIQ-200/availability"

    response = availability_client.get("FL-AIQ-200")

    assert_status_code(response, 200, endpoint=endpoint)
    body = assert_json_response(response, endpoint=endpoint)
    availability = AvailabilityResponse.model_validate(body)
    assert_valid_availability(availability)
    assert availability.available_seats == 0
    assert availability.is_available is False


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.availability
def test_availability_returns_not_found_for_unknown_flight(availability_client):
    endpoint = "/airline/flights/FL-UNKNOWN/availability"

    response = availability_client.get("FL-UNKNOWN")

    assert_status_code(response, 404, endpoint=endpoint)
    body = assert_json_response(response, endpoint=endpoint)
    assert body["error"] == "flight not found"
