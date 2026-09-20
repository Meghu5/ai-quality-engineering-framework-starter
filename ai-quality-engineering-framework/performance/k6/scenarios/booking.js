import { Rate } from 'k6/metrics';
import { sleep } from 'k6';

import { post } from '../lib/client.js';
import { json } from '../lib/client.js';
import { expectedFare, expectedFlight, passenger } from '../lib/data.js';
import {
  validateBooking,
  validatePassenger,
} from '../lib/assertions.js';
import { localThresholds, optionsFor, scenarioThresholds } from '../lib/thresholds.js';

export const bookingSuccess = new Rate('booking_success');

export const options = optionsFor(__ENV.K6_PROFILE || 'load', {
  ...localThresholds,
  ...scenarioThresholds,
});

export default function () {
  const passengerResponse = post('/airline/passengers', passenger, {
    scenario: 'booking',
    step: 'passenger',
  });
  const passengerIsValid = validatePassenger(passengerResponse);
  const passengerId = json(passengerResponse).passenger_id;

  const bookingResponse = post(
    '/airline/bookings',
    {
      flight_id: expectedFlight.flight_id,
      fare_id: expectedFare.fare_id,
      passenger_ids: [passengerId],
    },
    {
      scenario: 'booking',
      step: 'booking',
    },
  );

  bookingSuccess.add(passengerIsValid && validateBooking(bookingResponse));
  sleep(1);
}
