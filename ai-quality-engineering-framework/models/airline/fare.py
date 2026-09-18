from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field, field_validator, model_validator


class FareResponse(BaseModel):
    fare_id: str = Field(..., min_length=1)
    flight_id: str = Field(..., min_length=1)
    currency: str = Field(..., min_length=3, max_length=3)
    fare_class: str = Field(..., min_length=1)
    base_amount: Decimal = Field(..., ge=0)
    taxes: Decimal = Field(..., ge=0)
    total_amount: Decimal = Field(..., ge=0)

    @field_validator("currency")
    @classmethod
    def currency_must_be_uppercase(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency must be a three-letter ISO code")
        return normalized

    @model_validator(mode="after")
    def total_must_equal_base_plus_taxes(self) -> "FareResponse":
        if self.total_amount != self.base_amount + self.taxes:
            raise ValueError("total_amount must equal base_amount plus taxes")
        return self
