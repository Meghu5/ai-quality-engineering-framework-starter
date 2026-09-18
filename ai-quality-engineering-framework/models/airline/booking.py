from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from models.airline.common import BookingStatus
from models.airline.flight import Flight
from models.airline.passenger import PassengerResponse


class BookingRequest(BaseModel):
    flight_id: str = Field(..., min_length=1)
    fare_id: str = Field(..., min_length=1)
    passenger_ids: list[str] = Field(..., min_length=1)

    @field_validator("passenger_ids")
    @classmethod
    def passenger_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("passenger_ids must be unique")
        return value


class BookingResponse(BaseModel):
    booking_id: str = Field(..., min_length=1)
    pnr: str = Field(..., min_length=6, max_length=6)
    status: BookingStatus
    flight: Flight
    passengers: list[PassengerResponse] = Field(..., min_length=1)
    fare_id: str = Field(..., min_length=1)
