import pytest

from models.airline.passenger import PassengerResponse
from utils.assertions import assert_json_response, assert_status_code


ENDPOINT = "/airline/passengers"


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.passenger
def test_passenger_creation_returns_passenger_identifier(
    passenger_client,
    passenger_request,
):
    response = passenger_client.create(passenger_request)

    assert_status_code(response, 201, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    passenger = PassengerResponse.model_validate(body)
    assert passenger.passenger_id.startswith("PAX-")
    assert passenger.first_name == passenger_request.first_name
    assert passenger.last_name == passenger_request.last_name


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.passenger
def test_passenger_creation_rejects_missing_required_name(
    passenger_client,
    passenger_request,
):
    payload = passenger_request.model_dump(mode="json")
    payload.pop("last_name")

    response = passenger_client.create_raw(payload)

    assert_status_code(response, 422, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    assert "last_name" in body["error"]


@pytest.mark.api
@pytest.mark.airline
@pytest.mark.passenger
def test_passenger_creation_rejects_invalid_passenger_type(
    passenger_client,
    passenger_request,
):
    payload = passenger_request.model_dump(mode="json")
    payload["passenger_type"] = "VIP"

    response = passenger_client.create_raw(payload)

    assert_status_code(response, 422, endpoint=ENDPOINT)
    body = assert_json_response(response, endpoint=ENDPOINT)
    assert "Input should be" in body["error"]
