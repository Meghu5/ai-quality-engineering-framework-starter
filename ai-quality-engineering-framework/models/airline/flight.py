from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from models.airline.common import Cabin, TripType


class FlightSearchRequest(BaseModel):
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    departure_date: date
    passengers: int = Field(..., ge=1, le=9)
    cabin: Cabin = Cabin.ECONOMY
    trip_type: TripType = TripType.ONE_WAY

    @field_validator("origin", "destination")
    @classmethod
    def airport_codes_must_be_uppercase(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("airport code must be a three-letter IATA code")
        return normalized

    @model_validator(mode="after")
    def origin_and_destination_must_differ(self) -> "FlightSearchRequest":
        if self.origin == self.destination:
            raise ValueError("origin and destination must be different")
        return self


class Flight(BaseModel):
    flight_id: str = Field(..., min_length=1)
    carrier_code: str = Field(..., min_length=2, max_length=3)
    flight_number: str = Field(..., min_length=1)
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    departure_at: datetime
    arrival_at: datetime
    duration_minutes: int = Field(..., gt=0)
    cabin: Cabin
    fare_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def arrival_must_be_after_departure(self) -> "Flight":
        if self.arrival_at <= self.departure_at:
            raise ValueError("arrival_at must be after departure_at")
        return self


class FlightSearchResponse(BaseModel):
    flights: list[Flight]
