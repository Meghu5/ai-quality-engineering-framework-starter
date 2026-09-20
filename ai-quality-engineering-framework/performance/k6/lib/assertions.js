import { check } from 'k6';

import { json } from './client.js';
import { expectedBooking, expectedFare, expectedFlight } from './data.js';

export function validateHealth(response) {
  return check(response, {
    'health status is 200': (res) => res.status === 200,
    'health response is ok': (res) => json(res).status === 'ok',
  });
}

export function validateFlightSearch(response) {
  const body = json(response);
  const flight = body.flights && body.flights[0];
  return check(response, {
    'flight search status is 200': (res) => res.status === 200,
    'flight search has flights': () => Array.isArray(body.flights) && body.flights.length > 0,
    'flight search returns expected flight': () => flight && flight.flight_id === expectedFlight.flight_id,
    'flight search route is correct': () =>
      flight && flight.origin === expectedFlight.origin && flight.destination === expectedFlight.destination,
  });
}

export function validateAvailability(response) {
  const body = json(response);
  return check(response, {
    'availability status is 200': (res) => res.status === 200,
    'availability flight id is correct': () => body.flight_id === expectedFlight.flight_id,
    'availability seats are numeric': () => Number.isInteger(body.available_seats),
    'availability flag matches seats': () => body.is_available === body.available_seats > 0,
  });
}

export function validateFare(response) {
  const body = json(response);
  return check(response, {
    'fare status is 200': (res) => res.status === 200,
    'fare id is correct': () => body.fare_id === expectedFare.fare_id,
    'fare flight id is correct': () => body.flight_id === expectedFare.flight_id,
    'fare currency is correct': () => body.currency === expectedFare.currency,
    'fare total amount exists': () => Number(body.total_amount) > 0,
  });
}

export function validatePassenger(response) {
  const body = json(response);
  return check(response, {
    'passenger status is 201': (res) => res.status === 201,
    'passenger id generated': () => typeof body.passenger_id === 'string' && body.passenger_id.startsWith('PAX-'),
  });
}

export function validateBooking(response) {
  const body = json(response);
  const passenger = body.passengers && body.passengers[0];
  return check(response, {
    'booking status is 201': (res) => res.status === 201,
    'booking id is correct': () => body.booking_id === expectedBooking.booking_id,
    'booking pnr is correct': () => body.pnr === expectedBooking.pnr,
    'booking confirmed': () => body.status === expectedBooking.status,
    'booking flight is correct': () => body.flight && body.flight.flight_id === expectedFlight.flight_id,
    'booking fare is correct': () => body.fare_id === expectedFare.fare_id,
    'booking passenger is linked': () => passenger && passenger.passenger_id,
  });
}
