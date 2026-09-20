import { Rate } from 'k6/metrics';
import { sleep } from 'k6';

import { get } from '../lib/client.js';
import { expectedFare } from '../lib/data.js';
import { validateFare } from '../lib/assertions.js';
import { localThresholds, optionsFor, scenarioThresholds } from '../lib/thresholds.js';

export const fareSuccess = new Rate('fare_success');

export const options = optionsFor(__ENV.K6_PROFILE || 'load', {
  ...localThresholds,
  ...scenarioThresholds,
});

export default function () {
  const response = get(`/airline/fares/${expectedFare.fare_id}`, {
    scenario: 'fare',
  });
  fareSuccess.add(validateFare(response));
  sleep(1);
}
