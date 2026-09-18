from __future__ import annotations

from enum import StrEnum


class Cabin(StrEnum):
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"


class PassengerType(StrEnum):
    ADULT = "ADULT"
    CHILD = "CHILD"
    INFANT = "INFANT"


class TripType(StrEnum):
    ONE_WAY = "ONE_WAY"
    ROUND_TRIP = "ROUND_TRIP"


class BookingStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    HELD = "HELD"
    CANCELLED = "CANCELLED"
