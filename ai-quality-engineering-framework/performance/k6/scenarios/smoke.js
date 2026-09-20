import { Rate } from 'k6/metrics';
import { sleep } from 'k6';

import { get, json, post } from '../lib/client.js';
import { expectedFare, expectedFlight, passenger, searchRequest } from '../lib/data.js';
import {
  validateAvailability,
  validateBooking,
  validateFare,
  validateFlightSearch,
  validateHealth,
  validatePassenger,
} from '../lib/assertions.js';
import { optionsFor, scenarioThresholds, smokeThresholds } from '../lib/thresholds.js';

export const flightSearchSuccess = new Rate('flight_search_success');
export const availabilitySuccess = new Rate('availability_success');
export const fareSuccess = new Rate('fare_success');
export const bookingSuccess = new Rate('booking_success');

export const options = optionsFor('smoke', {
  ...smokeThresholds,
  ...scenarioThresholds,
});

export default function () {
  validateHealth(get('/health', { scenario: 'smoke', step: 'health' }));

  const searchResponse = post('/airline/flights/search', searchRequest, {
    scenario: 'smoke',
    step: 'flight_search',
  });
  flightSearchSuccess.add(validateFlightSearch(searchResponse));

  const availabilityResponse = get(
    `/airline/flights/${expectedFlight.flight_id}/availability`,
    { scenario: 'smoke', step: 'availability' },
  );
  availabilitySuccess.add(validateAvailability(availabilityResponse));

  const fareResponse = get(`/airline/fares/${expectedFare.fare_id}`, {
    scenario: 'smoke',
    step: 'fare',
  });
  fareSuccess.add(validateFare(fareResponse));

  const passengerResponse = post('/airline/passengers', passenger, {
    scenario: 'smoke',
    step: 'passenger',
  });
  const passengerIsValid = validatePassenger(passengerResponse);

  const bookingResponse = post(
    '/airline/bookings',
    {
      flight_id: expectedFlight.flight_id,
      fare_id: expectedFare.fare_id,
      passenger_ids: [json(passengerResponse).passenger_id],
    },
    { scenario: 'smoke', step: 'booking' },
  );
  bookingSuccess.add(passengerIsValid && validateBooking(bookingResponse));
  sleep(1);
}
