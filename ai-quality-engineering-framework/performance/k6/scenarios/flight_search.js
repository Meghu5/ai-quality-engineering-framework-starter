import { Rate } from 'k6/metrics';
import { sleep } from 'k6';

import { post } from '../lib/client.js';
import { searchRequest } from '../lib/data.js';
import { validateFlightSearch } from '../lib/assertions.js';
import { localThresholds, optionsFor, scenarioThresholds } from '../lib/thresholds.js';

export const flightSearchSuccess = new Rate('flight_search_success');

export const options = optionsFor(__ENV.K6_PROFILE || 'load', {
  ...localThresholds,
  ...scenarioThresholds,
});

export default function () {
  const response = post('/airline/flights/search', searchRequest, {
    scenario: 'flight_search',
  });
  flightSearchSuccess.add(validateFlightSearch(response));
  sleep(1);
}
