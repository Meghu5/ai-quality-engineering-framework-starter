from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from clients.airline import BookingClient, FlightClient, PassengerClient
from models.airline.booking import BookingRequest
from models.airline.flight import FlightSearchRequest
from models.airline.passenger import PassengerRequest


SUPPORTED_AIRPORTS = {"JFK", "LHR", "LAX", "NRT"}


def load_airline_json(data_dir: Path, name: str) -> list[dict[str, Any]]:
    with (data_dir / name).open(encoding="utf-8") as file:
        return json.load(file)


def _json_body(request: httpx.Request) -> dict[str, Any]:
    try:
        body = json.loads(request.content.decode("utf-8"))
    except ValueError:
        raise ValueError("malformed JSON")

    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def _error(status_code: int, message: str) -> httpx.Response:
    return httpx.Response(status_code, json={"error": message})


def deterministic_airline_transport(
    flights: list[dict[str, Any]],
    fares: list[dict[str, Any]],
) -> httpx.MockTransport:
    fare_by_id = {fare["fare_id"]: fare for fare in fares}
    flight_by_id = {flight["flight_id"]: flight for flight in flights}
    passenger_by_id: dict[str, dict[str, Any]] = {}

    def search_flights(request: httpx.Request) -> httpx.Response:
        try:
            body = _json_body(request)
            search = FlightSearchRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return _error(422, str(exc))

        if search.origin not in SUPPORTED_AIRPORTS:
            return _error(422, "origin airport is not supported")
        if search.destination not in SUPPORTED_AIRPORTS:
            return _error(422, "destination airport is not supported")
        if search.departure_date < date.today():
            return _error(422, "departure date must not be in the past")

        matching_flights = [
            {
                key: value
                for key, value in flight.items()
                if key != "available_seats"
            }
            for flight in flights
            if flight["origin"] == search.origin
            and flight["destination"] == search.destination
            and flight["departure_at"].startswith(search.departure_date.isoformat())
            and flight["cabin"] == search.cabin
            and flight["available_seats"] >= search.passengers
        ]
        return httpx.Response(200, json={"flights": matching_flights})

    def get_availability(flight_id: str) -> httpx.Response:
        flight = flight_by_id.get(flight_id)
        if not flight:
            return _error(404, "flight not found")

        available_seats = flight["available_seats"]
        return httpx.Response(
            200,
            json={
                "flight_id": flight_id,
                "cabin": flight["cabin"],
                "available_seats": available_seats,
                "is_available": available_seats > 0,
            },
        )

    def get_fare(fare_id: str) -> httpx.Response:
        fare = fare_by_id.get(fare_id)
        if not fare:
            return _error(404, "fare not found")
        return httpx.Response(200, json=fare)

    def create_passenger(request: httpx.Request) -> httpx.Response:
        try:
            body = _json_body(request)
            passenger = PassengerRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return _error(422, str(exc))

        passenger_id = f"PAX-{len(passenger_by_id) + 1:04d}"
        response_body = passenger.model_dump(mode="json")
        response_body["passenger_id"] = passenger_id
        passenger_by_id[passenger_id] = response_body
        return httpx.Response(201, json=response_body)

    def create_booking(request: httpx.Request) -> httpx.Response:
        try:
            body = _json_body(request)
            booking = BookingRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return _error(422, str(exc))

        flight = flight_by_id.get(booking.flight_id)
        if not flight:
            return _error(404, "flight not found")

        fare = fare_by_id.get(booking.fare_id)
        if not fare or fare["flight_id"] != booking.flight_id:
            return _error(404, "fare not found for flight")

        missing_passengers = [
            passenger_id
            for passenger_id in booking.passenger_ids
            if passenger_id not in passenger_by_id
        ]
        if missing_passengers:
            return _error(404, "passenger not found")

        if flight["available_seats"] < len(booking.passenger_ids):
            return _error(409, "flight is unavailable")

        flight_body = {
            key: value
            for key, value in flight.items()
            if key != "available_seats"
        }
        return httpx.Response(
            201,
            json={
                "booking_id": "BKG-AIQ-0001",
                "pnr": "AIQ7K2",
                "status": "CONFIRMED",
                "flight": flight_body,
                "passengers": [
                    passenger_by_id[passenger_id]
                    for passenger_id in booking.passenger_ids
                ],
                "fare_id": booking.fare_id,
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if request.method == "POST" and path == FlightClient.endpoint:
            return search_flights(request)
        if (
            request.method == "GET"
            and path.startswith("/airline/flights/")
            and path.endswith("/availability")
        ):
            flight_id = path.removeprefix("/airline/flights/").removesuffix(
                "/availability"
            )
            return get_availability(flight_id)
        if request.method == "GET" and path.startswith("/airline/fares/"):
            return get_fare(path.removeprefix("/airline/fares/"))
        if request.method == "POST" and path == PassengerClient.endpoint:
            return create_passenger(request)
        if request.method == "POST" and path == BookingClient.endpoint:
            return create_booking(request)

        return _error(404, "endpoint not found")

    return httpx.MockTransport(handler)
