from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DuffelApiResponse(BaseModel):
    data: Any


class DuffelAirport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    iata_code: str | None = None
    name: str | None = None


class DuffelCarrier(BaseModel):
    model_config = ConfigDict(extra="ignore")

    iata_code: str | None = None
    name: str | None = None


class DuffelSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    origin: DuffelAirport
    destination: DuffelAirport
    departing_at: datetime
    arriving_at: datetime
    marketing_carrier: DuffelCarrier | None = None
    operating_carrier: DuffelCarrier | None = None
    marketing_carrier_flight_number: str | None = None
    duration: str | None = None


class DuffelSlice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    origin: DuffelAirport
    destination: DuffelAirport
    segments: list[DuffelSegment] = Field(default_factory=list)
    duration: str | None = None


class DuffelPassenger(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    type: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    born_on: str | None = None
    gender: str | None = None
    title: str | None = None


class DuffelPaymentRequirements(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requires_instant_payment: bool | None = None


class DuffelOffer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    owner: DuffelCarrier | None = None
    slices: list[DuffelSlice] = Field(default_factory=list)
    passengers: list[DuffelPassenger] = Field(default_factory=list)
    total_amount: Decimal
    total_currency: str
    base_amount: Decimal | None = None
    base_currency: str | None = None
    tax_amount: Decimal | None = None
    tax_currency: str | None = None
    expires_at: datetime | None = None
    payment_requirements: DuffelPaymentRequirements | None = None


class DuffelOfferRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    live_mode: bool | None = None
    offers: list[DuffelOffer] = Field(default_factory=list)
    passengers: list[DuffelPassenger] = Field(default_factory=list)
    slices: list[DuffelSlice] = Field(default_factory=list)


class DuffelListResponse(BaseModel):
    data: list[DuffelOffer] = Field(default_factory=list)


class DuffelOrder(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    booking_reference: str | None = None
    owner: DuffelCarrier | None = None
    slices: list[DuffelSlice] = Field(default_factory=list)
    passengers: list[DuffelPassenger] = Field(default_factory=list)
    total_amount: Decimal
    total_currency: str
    base_amount: Decimal | None = None
    base_currency: str | None = None
    tax_amount: Decimal | None = None
    tax_currency: str | None = None
    cancelled_at: datetime | None = None


class DuffelError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    title: str | None = None
    message: str | None = None
    code: str | None = None


class DuffelErrorResponse(BaseModel):
    errors: list[DuffelError] = Field(default_factory=list)
