import { Rate } from 'k6/metrics';
import { sleep } from 'k6';

import { get } from '../lib/client.js';
import { expectedFlight } from '../lib/data.js';
import { validateAvailability } from '../lib/assertions.js';
import { localThresholds, optionsFor, scenarioThresholds } from '../lib/thresholds.js';

export const availabilitySuccess = new Rate('availability_success');

export const options = optionsFor(__ENV.K6_PROFILE || 'load', {
  ...localThresholds,
  ...scenarioThresholds,
});

export default function () {
  const response = get(`/airline/flights/${expectedFlight.flight_id}/availability`, {
    scenario: 'availability',
  });
  availabilitySuccess.add(validateAvailability(response));
  sleep(1);
}
