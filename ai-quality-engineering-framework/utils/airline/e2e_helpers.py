from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from playwright.sync_api import expect

from clients.airline import BookingClient, PassengerClient
from models.airline.booking import BookingRequest, BookingResponse
from models.airline.common import Cabin, PassengerType, TripType
from models.airline.fare import FareResponse
from models.airline.flight import Flight, FlightSearchRequest
from models.airline.passenger import PassengerRequest, PassengerResponse
from pages.airline import (
    ConfirmationPage,
    FarePage,
    FlightResultsPage,
    HomePage,
    PassengerPage,
    ReviewBookingPage,
)
from utils.airline.ui_assertions import assert_review_summary
from utils.airline_assertions import assert_valid_booking
from utils.assertions import assert_json_response, assert_status_code


@dataclass(frozen=True)
class AirlineE2EContext:
    flight: Flight
    fare: FareResponse
    passenger: PassengerRequest
    passengers: int = 1

    @property
    def search_request(self) -> FlightSearchRequest:
        return FlightSearchRequest(
            origin=self.flight.origin,
            destination=self.flight.destination,
            departure_date=self.flight.departure_at.date(),
            passengers=self.passengers,
            cabin=self.flight.cabin,
            trip_type=TripType.ONE_WAY,
        )

    @property
    def route(self) -> str:
        return f"{self.flight.origin} to {self.flight.destination}"

    @property
    def itinerary(self) -> str:
        return f"{self.route} | {self.flight.flight_id}"

    @property
    def passenger_name(self) -> str:
        return f"{self.passenger.first_name} {self.passenger.last_name}"

    @property
    def fare_name(self) -> str:
        return fare_display_name(self.fare)


@dataclass(frozen=True)
class UiBookingSnapshot:
    pnr: str
    itinerary: str
    passenger: str
    fare: str


def fare_display_name(fare: FareResponse) -> str:
    names_by_class = {
        "E": "Economy",
        "W": "Premium Economy",
        "J": "Business",
    }
    return names_by_class.get(fare.fare_class, fare.fare_class)


def build_e2e_context(
    *,
    flights: list[dict],
    fares: list[dict],
    passenger_payload: dict,
) -> AirlineE2EContext:
    available_flight = next(
        flight
        for flight in flights
        if flight["available_seats"] > 0 and flight["origin"] == "JFK"
    )
    matching_fare = next(
        fare
        for fare in fares
        if fare["fare_id"] == available_flight["fare_id"]
    )
    passenger = PassengerRequest(
        first_name=passenger_payload["first_name"],
        last_name=passenger_payload["last_name"],
        passenger_type=PassengerType(passenger_payload["passenger_type"]),
        date_of_birth=date.fromisoformat(passenger_payload["date_of_birth"]),
    )
    return AirlineE2EContext(
        flight=Flight.model_validate(
            {
                key: value
                for key, value in available_flight.items()
                if key != "available_seats"
            }
        ),
        fare=FareResponse.model_validate(matching_fare),
        passenger=passenger,
    )


def create_booking_via_api(
    *,
    passenger_client: PassengerClient,
    booking_client: BookingClient,
    context: AirlineE2EContext,
) -> BookingResponse:
    passenger_response = passenger_client.create(context.passenger)
    assert_status_code(passenger_response, 201, endpoint="/airline/passengers")
    passenger_body = assert_json_response(
        passenger_response,
        endpoint="/airline/passengers",
    )
    passenger = PassengerResponse.model_validate(passenger_body)

    booking_response = booking_client.create(
        BookingRequest(
            flight_id=context.flight.flight_id,
            fare_id=context.fare.fare_id,
            passenger_ids=[passenger.passenger_id],
        )
    )
    assert_status_code(booking_response, 201, endpoint="/airline/bookings")
    booking_body = assert_json_response(
        booking_response,
        endpoint="/airline/bookings",
    )
    booking = BookingResponse.model_validate(booking_body)
    assert_valid_booking(booking)
    return booking


def complete_booking_via_ui(
    *,
    context: AirlineE2EContext,
    home_page: HomePage,
    flight_results_page: FlightResultsPage,
    fare_page: FarePage,
    passenger_page: PassengerPage,
    review_page: ReviewBookingPage,
    confirmation_page: ConfirmationPage,
) -> UiBookingSnapshot:
    home_page.search(
        origin=context.flight.origin,
        destination=context.flight.destination,
        departure_date=context.search_request.departure_date.isoformat(),
        passengers=context.passengers,
    )
    expect(flight_results_page.results_view).to_be_visible()

    flight_results_page.select_flight(context.flight.flight_id)
    flight_results_page.continue_to_fares()
    expect(fare_page.fare_view).to_be_visible()

    fare_page.select_fare(context.fare.fare_id)
    fare_page.continue_to_passenger()
    expect(passenger_page.passenger_view).to_be_visible()

    passenger_page.enter_passenger(
        first_name=context.passenger.first_name,
        last_name=context.passenger.last_name,
        date_of_birth=context.passenger.date_of_birth.isoformat(),
        passenger_type=context.passenger.passenger_type,
    )
    passenger_page.continue_to_review()

    assert_review_summary(
        review_page,
        route=context.route,
        flight_id=context.flight.flight_id,
        fare_name=context.fare_name,
        passenger_name=context.passenger_name,
        currency=context.fare.currency,
    )

    review_page.confirm_booking()
    expect(confirmation_page.confirmation_view).to_be_visible()
    return UiBookingSnapshot(
        pnr=confirmation_page.pnr.inner_text(),
        itinerary=confirmation_page.itinerary.inner_text(),
        passenger=confirmation_page.passenger.inner_text(),
        fare=confirmation_page.fare.inner_text(),
    )


def assert_flight_consistent(
    *,
    api_booking: BookingResponse,
    ui_booking: UiBookingSnapshot,
) -> None:
    assert api_booking.flight.flight_id in ui_booking.itinerary, (
        "UI itinerary does not contain API flight id "
        f"{api_booking.flight.flight_id}: {ui_booking.itinerary}"
    )
    route = f"{api_booking.flight.origin} to {api_booking.flight.destination}"
    assert route in ui_booking.itinerary, (
        f"UI itinerary does not contain API route {route}: {ui_booking.itinerary}"
    )


def assert_fare_consistent(
    *,
    api_fare: FareResponse,
    ui_booking: UiBookingSnapshot,
) -> None:
    assert api_fare.fare_id in ui_booking.fare, (
        f"UI fare does not contain API fare id {api_fare.fare_id}: {ui_booking.fare}"
    )
    assert api_fare.currency in ui_booking.fare, (
        f"UI fare does not contain API currency {api_fare.currency}: {ui_booking.fare}"
    )
    expected_total = _money(api_fare.total_amount)
    assert expected_total in ui_booking.fare, (
        f"UI fare does not contain API total {expected_total}: {ui_booking.fare}"
    )


def assert_passenger_consistent(
    *,
    api_booking: BookingResponse,
    ui_booking: UiBookingSnapshot,
) -> None:
    api_passenger = api_booking.passengers[0]
    expected_name = f"{api_passenger.first_name} {api_passenger.last_name}"
    assert ui_booking.passenger == expected_name, (
        f"UI passenger {ui_booking.passenger!r} did not match API passenger "
        f"{expected_name!r}"
    )


def assert_pnr_consistent(
    *,
    api_booking: BookingResponse,
    ui_booking: UiBookingSnapshot,
) -> None:
    assert ui_booking.pnr == api_booking.pnr, (
        f"UI PNR {ui_booking.pnr!r} did not match API PNR {api_booking.pnr!r}"
    )


def assert_itinerary_consistent(
    *,
    context: AirlineE2EContext,
    ui_booking: UiBookingSnapshot,
) -> None:
    assert ui_booking.itinerary == context.itinerary, (
        f"UI itinerary {ui_booking.itinerary!r} did not match expected "
        f"{context.itinerary!r}"
    )


def assert_booking_matches_api_and_ui(
    *,
    context: AirlineE2EContext,
    api_booking: BookingResponse,
    ui_booking: UiBookingSnapshot,
) -> None:
    assert_flight_consistent(api_booking=api_booking, ui_booking=ui_booking)
    assert_fare_consistent(api_fare=context.fare, ui_booking=ui_booking)
    assert_passenger_consistent(api_booking=api_booking, ui_booking=ui_booking)
    assert_pnr_consistent(api_booking=api_booking, ui_booking=ui_booking)
    assert_itinerary_consistent(context=context, ui_booking=ui_booking)


def _money(amount: Decimal) -> str:
    return f"{amount:.2f}"
