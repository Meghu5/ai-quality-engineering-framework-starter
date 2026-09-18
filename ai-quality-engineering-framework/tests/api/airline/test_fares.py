import pytest

from models.airline.fare import FareResponse
from utils.airline_assertions import assert_valid_fare
from utils.assertions import assert_json_response, assert_status_code


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.fare
def test_fare_returns_consistent_pricing(fare_client):
    endpoint = "/airline/fares/FARE-AIQ-100-E"

    response = fare_client.get("FARE-AIQ-100-E")

    assert_status_code(response, 200, endpoint=endpoint)
    body = assert_json_response(response, endpoint=endpoint)
    fare = FareResponse.model_validate(body)
    assert_valid_fare(fare)


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.fare
def test_fare_returns_not_found_for_unknown_fare(fare_client):
    endpoint = "/airline/fares/FARE-UNKNOWN"

    response = fare_client.get("FARE-UNKNOWN")

    assert_status_code(response, 404, endpoint=endpoint)
    body = assert_json_response(response, endpoint=endpoint)
    assert body["error"] == "fare not found"
