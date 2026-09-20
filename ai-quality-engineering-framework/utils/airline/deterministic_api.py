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
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
SECURITY_IDENTITIES = {
    "valid-user-token": {
        "subject": "user-001",
        "role": "traveler",
        "passenger_ids": {"PAX-SEC-0001"},
        "booking_ids": {"BKG-SEC-0001"},
    },
    "another-user-token": {
        "subject": "user-002",
        "role": "traveler",
        "passenger_ids": {"PAX-SEC-0002"},
        "booking_ids": {"BKG-SEC-0002"},
    },
    "admin-test-token": {
        "subject": "admin-001",
        "role": "admin",
        "passenger_ids": set(),
        "booking_ids": set(),
    },
}
EXPIRED_SECURITY_TOKENS = {"expired-test-token"}


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
        *,
        security_enabled: bool = False,
    ) -> None:
        self.flights = flights
        self.fares = fares
        self.security_enabled = security_enabled
        self.fare_by_id = {fare["fare_id"]: fare for fare in fares}
        self.flight_by_id = {flight["flight_id"]: flight for flight in flights}
        self.passenger_by_id: dict[str, dict[str, Any]] = {}
        self.booking_by_id: dict[str, dict[str, Any]] = {}
        self.resource_owner_by_id: dict[str, str] = {}
        self.rate_limit_counts_by_subject: dict[str, int] = {}
        self._lock = threading.Lock()
        if security_enabled:
            self._seed_security_resources()

    def handle(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        method = method.upper()
        identity: dict[str, Any] | None = None

        if self.security_enabled and path.startswith("/airline/"):
            status_code, response_body, identity = self._authenticate(headers or {})
            if status_code != 200:
                return status_code, response_body

        if method != "GET" and path.startswith("/airline/passengers/"):
            return 405, {"error": "method not allowed"}
        if method != "GET" and path.startswith("/airline/bookings/"):
            return 405, {"error": "method not allowed"}

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
            return self.create_passenger(body or {}, identity=identity)
        if method == "POST" and path == BookingClient.endpoint:
            return self.create_booking(body or {}, identity=identity)
        if method == "GET" and path.startswith("/airline/passengers/"):
            return self.get_passenger(
                path.removeprefix("/airline/passengers/"),
                identity=identity,
            )
        if method == "GET" and path.startswith("/airline/bookings/"):
            return self.get_booking(
                path.removeprefix("/airline/bookings/"),
                identity=identity,
            )
        if method == "GET" and path == "/airline/security/rate-limit-probe":
            return self.rate_limit_probe(identity=identity)

        return 404, {"error": "endpoint not found"}

    def search_flights(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        unexpected_fields = set(body) - set(FlightSearchRequest.model_fields)
        if self.security_enabled and unexpected_fields:
            return 422, {"error": "unexpected request field"}

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

    def create_passenger(
        self,
        body: dict[str, Any],
        *,
        identity: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        unexpected_fields = set(body) - set(PassengerRequest.model_fields)
        if self.security_enabled and unexpected_fields:
            return 422, {"error": "unexpected request field"}
        if self.security_enabled and any(
            isinstance(value, str) and len(value) > 256 for value in body.values()
        ):
            return 413, {"error": "request field exceeds maximum length"}

        try:
            passenger = PassengerRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return 422, {"error": str(exc)}

        with self._lock:
            passenger_id = f"PAX-{len(self.passenger_by_id) + 1:04d}"
            response_body = passenger.model_dump(mode="json")
            response_body["passenger_id"] = passenger_id
            self.passenger_by_id[passenger_id] = response_body
            if identity:
                self.resource_owner_by_id[passenger_id] = identity["subject"]
        return 201, response_body

    def create_booking(
        self,
        body: dict[str, Any],
        *,
        identity: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        unexpected_fields = set(body) - set(BookingRequest.model_fields)
        if self.security_enabled and unexpected_fields:
            return 422, {"error": "unexpected request field"}

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

        if identity and identity["role"] != "admin":
            unauthorized_passengers = [
                passenger_id
                for passenger_id in booking.passenger_ids
                if not self._can_access(passenger_id, identity)
            ]
            if unauthorized_passengers:
                return 403, {"error": "access denied"}

        if flight["available_seats"] < len(booking.passenger_ids):
            return 409, {"error": "flight is unavailable"}

        flight_body = {
            key: value
            for key, value in flight.items()
            if key != "available_seats"
        }
        response_body = {
            "booking_id": "BKG-AIQ-0001",
            "pnr": "AIQ7K2",
            "status": "CONFIRMED",
            "flight": flight_body,
            "passengers": [
                self.passenger_by_id[passenger_id]
                for passenger_id in booking.passenger_ids
            ],
            "fare_id": booking.fare_id,
        }
        if identity:
            self.booking_by_id[response_body["booking_id"]] = response_body
            self.resource_owner_by_id[response_body["booking_id"]] = identity["subject"]
        return 201, response_body

    def get_passenger(
        self,
        passenger_id: str,
        *,
        identity: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        passenger = self.passenger_by_id.get(passenger_id)
        if not passenger:
            return 404, {"error": "passenger not found"}
        if identity and not self._can_access(passenger_id, identity):
            return 403, {"error": "access denied"}
        return 200, passenger

    def get_booking(
        self,
        booking_id: str,
        *,
        identity: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        booking = self.booking_by_id.get(booking_id)
        if not booking:
            return 404, {"error": "booking not found"}
        if identity and not self._can_access(booking_id, identity):
            return 403, {"error": "access denied"}
        return 200, booking

    def rate_limit_probe(
        self,
        *,
        identity: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        subject = identity["subject"] if identity else "anonymous"
        self.rate_limit_counts_by_subject[subject] = (
            self.rate_limit_counts_by_subject.get(subject, 0) + 1
        )
        if self.rate_limit_counts_by_subject[subject] > 3:
            return 429, {"error": "rate limit exceeded"}
        return 200, {"status": "ok"}

    def _authenticate(
        self,
        headers: dict[str, str],
    ) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        auth_header = headers.get("authorization", "")
        if not auth_header:
            return 401, {"error": "authentication required"}, None
        if not auth_header.startswith("Bearer "):
            return 401, {"error": "invalid authorization header"}, None

        token = auth_header.removeprefix("Bearer ").strip()
        if token in EXPIRED_SECURITY_TOKENS:
            return 401, {"error": "token expired or invalid"}, None

        identity = SECURITY_IDENTITIES.get(token)
        if not identity:
            return 401, {"error": "token expired or invalid"}, None
        return 200, {}, identity

    def _can_access(self, resource_id: str, identity: dict[str, Any]) -> bool:
        if identity["role"] == "admin":
            return True
        return self.resource_owner_by_id.get(resource_id) == identity["subject"]

    def _seed_security_resources(self) -> None:
        self.passenger_by_id.update(
            {
                "PAX-SEC-0001": {
                    "first_name": "Avery",
                    "last_name": "Stone",
                    "passenger_type": "ADULT",
                    "date_of_birth": "1988-04-12",
                    "passenger_id": "PAX-SEC-0001",
                },
                "PAX-SEC-0002": {
                    "first_name": "Mira",
                    "last_name": "Chen",
                    "passenger_type": "ADULT",
                    "date_of_birth": "1991-02-03",
                    "passenger_id": "PAX-SEC-0002",
                },
            }
        )
        for token, identity in SECURITY_IDENTITIES.items():
            if token == "admin-test-token":
                continue
            for passenger_id in identity["passenger_ids"]:
                self.resource_owner_by_id[passenger_id] = identity["subject"]

        flight_body = {
            key: value
            for key, value in self.flights[0].items()
            if key != "available_seats"
        }
        self.booking_by_id.update(
            {
                "BKG-SEC-0001": {
                    "booking_id": "BKG-SEC-0001",
                    "pnr": "AIQ7K2",
                    "status": "CONFIRMED",
                    "flight": flight_body,
                    "passengers": [self.passenger_by_id["PAX-SEC-0001"]],
                    "fare_id": self.fares[0]["fare_id"],
                },
                "BKG-SEC-0002": {
                    "booking_id": "BKG-SEC-0002",
                    "pnr": "AIQ9M4",
                    "status": "CONFIRMED",
                    "flight": flight_body,
                    "passengers": [self.passenger_by_id["PAX-SEC-0002"]],
                    "fare_id": self.fares[0]["fare_id"],
                },
            }
        )
        for token, identity in SECURITY_IDENTITIES.items():
            if token == "admin-test-token":
                continue
            for booking_id in identity["booking_ids"]:
                self.resource_owner_by_id[booking_id] = identity["subject"]


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
            headers={key.lower(): value for key, value in request.headers.items()},
        )
        return httpx.Response(status_code, json=response_body)

    return httpx.MockTransport(handler)
