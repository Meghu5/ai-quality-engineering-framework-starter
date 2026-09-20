from __future__ import annotations

import json
import threading
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


class DeterministicAirlineBackend:
    """Reusable deterministic airline API test double.

    This backend powers pytest MockTransport and the local HTTP performance target.
    It is not a production airline backend.
    """

    def __init__(
        self,
        flights: list[dict[str, Any]],
        fares: list[dict[str, Any]],
    ) -> None:
        self.flights = flights
        self.fares = fares
        self.fare_by_id = {fare["fare_id"]: fare for fare in fares}
        self.flight_by_id = {flight["flight_id"]: flight for flight in flights}
        self.passenger_by_id: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def handle(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        method = method.upper()

        if method == "POST" and path == FlightClient.endpoint:
            return self.search_flights(body or {})
        if (
            method == "GET"
            and path.startswith("/airline/flights/")
            and path.endswith("/availability")
        ):
            flight_id = path.removeprefix("/airline/flights/").removesuffix(
                "/availability"
            )
            return self.get_availability(flight_id)
        if method == "GET" and path.startswith("/airline/fares/"):
            return self.get_fare(path.removeprefix("/airline/fares/"))
        if method == "POST" and path == PassengerClient.endpoint:
            return self.create_passenger(body or {})
        if method == "POST" and path == BookingClient.endpoint:
            return self.create_booking(body or {})

        return 404, {"error": "endpoint not found"}

    def search_flights(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            search = FlightSearchRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return 422, {"error": str(exc)}

        if search.origin not in SUPPORTED_AIRPORTS:
            return 422, {"error": "origin airport is not supported"}
        if search.destination not in SUPPORTED_AIRPORTS:
            return 422, {"error": "destination airport is not supported"}
        if search.departure_date < date.today():
            return 422, {"error": "departure date must not be in the past"}

        matching_flights = [
            {
                key: value
                for key, value in flight.items()
                if key != "available_seats"
            }
            for flight in self.flights
            if flight["origin"] == search.origin
            and flight["destination"] == search.destination
            and flight["departure_at"].startswith(search.departure_date.isoformat())
            and flight["cabin"] == search.cabin
            and flight["available_seats"] >= search.passengers
        ]
        return 200, {"flights": matching_flights}

    def get_availability(self, flight_id: str) -> tuple[int, dict[str, Any]]:
        flight = self.flight_by_id.get(flight_id)
        if not flight:
            return 404, {"error": "flight not found"}

        available_seats = flight["available_seats"]
        return (
            200,
            {
                "flight_id": flight_id,
                "cabin": flight["cabin"],
                "available_seats": available_seats,
                "is_available": available_seats > 0,
            },
        )

    def get_fare(self, fare_id: str) -> tuple[int, dict[str, Any]]:
        fare = self.fare_by_id.get(fare_id)
        if not fare:
            return 404, {"error": "fare not found"}
        return 200, fare

    def create_passenger(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            passenger = PassengerRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return 422, {"error": str(exc)}

        with self._lock:
            passenger_id = f"PAX-{len(self.passenger_by_id) + 1:04d}"
            response_body = passenger.model_dump(mode="json")
            response_body["passenger_id"] = passenger_id
            self.passenger_by_id[passenger_id] = response_body
        return 201, response_body

    def create_booking(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            booking = BookingRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return 422, {"error": str(exc)}

        flight = self.flight_by_id.get(booking.flight_id)
        if not flight:
            return 404, {"error": "flight not found"}

        fare = self.fare_by_id.get(booking.fare_id)
        if not fare or fare["flight_id"] != booking.flight_id:
            return 404, {"error": "fare not found for flight"}

        missing_passengers = [
            passenger_id
            for passenger_id in booking.passenger_ids
            if passenger_id not in self.passenger_by_id
        ]
        if missing_passengers:
            return 404, {"error": "passenger not found"}

        if flight["available_seats"] < len(booking.passenger_ids):
            return 409, {"error": "flight is unavailable"}

        flight_body = {
            key: value
            for key, value in flight.items()
            if key != "available_seats"
        }
        return (
            201,
            {
                "booking_id": "BKG-AIQ-0001",
                "pnr": "AIQ7K2",
                "status": "CONFIRMED",
                "flight": flight_body,
                "passengers": [
                    self.passenger_by_id[passenger_id]
                    for passenger_id in booking.passenger_ids
                ],
                "fare_id": booking.fare_id,
            },
        )


def deterministic_airline_transport(
    flights: list[dict[str, Any]],
    fares: list[dict[str, Any]],
) -> httpx.MockTransport:
    backend = DeterministicAirlineBackend(flights, fares)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            body = _json_body(request) if request.content else None
        except ValueError as exc:
            return _error(422, str(exc))

        status_code, response_body = backend.handle(
            method=request.method,
            path=request.url.path,
            body=body,
        )
        return httpx.Response(status_code, json=response_body)

    return httpx.MockTransport(handler)
