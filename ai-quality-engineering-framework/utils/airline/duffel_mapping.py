from __future__ import annotations

from decimal import Decimal

from models.airline.common import Cabin
from models.airline.fare import FareResponse
from models.airline.flight import Flight
from models.airline.duffel import DuffelOffer, DuffelOrder


def duffel_offer_to_flight(offer: DuffelOffer) -> Flight:
    first_slice = _first(offer.slices, "offer slices")
    first_segment = _first(first_slice.segments, "offer segments")
    origin = first_segment.origin.iata_code or first_slice.origin.iata_code
    destination = first_segment.destination.iata_code or first_slice.destination.iata_code
    carrier = (
        first_segment.marketing_carrier
        or first_segment.operating_carrier
        or offer.owner
    )
    carrier_code = carrier.iata_code if carrier and carrier.iata_code else "NA"
    flight_number = first_segment.marketing_carrier_flight_number or "UNKNOWN"

    return Flight(
        flight_id=first_segment.id,
        carrier_code=carrier_code,
        flight_number=flight_number,
        origin=_required(origin, "origin IATA code"),
        destination=_required(destination, "destination IATA code"),
        departure_at=first_segment.departing_at,
        arrival_at=first_segment.arriving_at,
        duration_minutes=_duration_minutes(first_segment.duration),
        cabin=Cabin.ECONOMY,
        fare_id=offer.id,
    )


def duffel_offer_to_fare(offer: DuffelOffer) -> FareResponse:
    base_amount = offer.base_amount or Decimal("0")
    taxes = offer.tax_amount
    if taxes is None:
        taxes = offer.total_amount - base_amount
    if taxes < 0:
        taxes = Decimal("0")

    return FareResponse(
        fare_id=offer.id,
        flight_id=duffel_offer_to_flight(offer).flight_id,
        currency=offer.total_currency,
        fare_class="DUFFEL",
        base_amount=base_amount,
        taxes=taxes,
        total_amount=offer.total_amount,
    )


def duffel_order_to_reference(order: DuffelOrder) -> str:
    return order.booking_reference or order.id


def _duration_minutes(duration: str | None) -> int:
    if not duration:
        return 1
    value = duration.removeprefix("P").removeprefix("T")
    hours = 0
    minutes = 0
    if "H" in value:
        hours_part, value = value.split("H", 1)
        hours = int(hours_part or 0)
    if "M" in value:
        minutes_part = value.split("M", 1)[0]
        minutes = int(minutes_part or 0)
    return max((hours * 60) + minutes, 1)


def _first(values: list, name: str):
    assert values, f"Duffel response did not include {name}"
    return values[0]


def _required(value: str | None, name: str) -> str:
    assert value, f"Duffel response did not include {name}"
    return value
