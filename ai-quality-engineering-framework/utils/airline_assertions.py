from __future__ import annotations

from decimal import Decimal

from models.airline.availability import AvailabilityResponse
from models.airline.booking import BookingResponse
from models.airline.common import BookingStatus
from models.airline.fare import FareResponse
from models.airline.flight import Flight


def assert_valid_flight(flight: Flight, *, origin: str, destination: str) -> None:
    assert flight.origin == origin, (
        f"{flight.flight_id}: expected origin {origin}, got {flight.origin}"
    )
    assert flight.destination == destination, (
        f"{flight.flight_id}: expected destination {destination}, got {flight.destination}"
    )
    assert flight.arrival_at > flight.departure_at, (
        f"{flight.flight_id}: arrival must be after departure"
    )
    assert flight.duration_minutes > 0, (
        f"{flight.flight_id}: duration must be positive"
    )


def assert_valid_availability(availability: AvailabilityResponse) -> None:
    assert availability.available_seats >= 0, (
        f"{availability.flight_id}: available seats must be non-negative"
    )
    assert availability.is_available == (availability.available_seats > 0), (
        f"{availability.flight_id}: availability flag does not match seat count"
    )


def assert_valid_fare(fare: FareResponse) -> None:
    assert fare.currency.strip(), f"{fare.fare_id}: currency must be present"
    assert fare.fare_class.strip(), f"{fare.fare_id}: fare class must be present"
    assert fare.base_amount >= Decimal("0"), f"{fare.fare_id}: base amount is negative"
    assert fare.taxes >= Decimal("0"), f"{fare.fare_id}: taxes are negative"
    assert fare.total_amount == fare.base_amount + fare.taxes, (
        f"{fare.fare_id}: total must equal base amount plus taxes"
    )


def assert_valid_booking(booking: BookingResponse) -> None:
    assert booking.status in {BookingStatus.CONFIRMED, BookingStatus.HELD}, (
        f"{booking.booking_id}: unexpected booking status {booking.status}"
    )
    assert len(booking.pnr) == 6 and booking.pnr.isalnum(), (
        f"{booking.booking_id}: expected six-character alphanumeric PNR, got {booking.pnr}"
    )
    assert booking.passengers, f"{booking.booking_id}: booking has no passengers"
    assert booking.flight.flight_id, f"{booking.booking_id}: booking has no flight"
