from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

from models.airline.common import PassengerType


class PassengerRequest(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    passenger_type: PassengerType = PassengerType.ADULT
    date_of_birth: date

    @field_validator("first_name", "last_name")
    @classmethod
    def names_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("passenger names must not be blank")
        return value.strip()


class PassengerResponse(PassengerRequest):
    passenger_id: str = Field(..., min_length=1)
