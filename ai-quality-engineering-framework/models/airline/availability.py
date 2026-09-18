from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from models.airline.common import Cabin


class AvailabilityResponse(BaseModel):
    flight_id: str = Field(..., min_length=1)
    cabin: Cabin
    available_seats: int = Field(..., ge=0)
    is_available: bool

    @model_validator(mode="after")
    def availability_must_match_seat_count(self) -> "AvailabilityResponse":
        if self.is_available != (self.available_seats > 0):
            raise ValueError("is_available must match available_seats")
        return self
