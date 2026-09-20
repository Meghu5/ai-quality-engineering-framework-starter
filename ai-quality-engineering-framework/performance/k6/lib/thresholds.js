export const smokeThresholds = {
  http_req_failed: ['rate<0.01'],
  http_req_duration: ['p(95)<750'],
  checks: ['rate>0.99'],
};

export const localThresholds = {
  http_req_failed: ['rate<0.01'],
  http_req_duration: ['p(90)<500', 'p(95)<750', 'p(99)<1200'],
  checks: ['rate>0.99'],
};

export const scenarioThresholds = {
  flight_search_success: ['rate>0.99'],
  availability_success: ['rate>0.99'],
  fare_success: ['rate>0.99'],
  booking_success: ['rate>0.99'],
};

export const profiles = {
  smoke: {
    vus: 1,
    duration: __ENV.K6_SMOKE_DURATION || '1m',
  },
  load: {
    stages: [
      { duration: '30s', target: 10 },
      { duration: '1m', target: 10 },
      { duration: '30s', target: 0 },
    ],
  },
  stress: {
    stages: [
      { duration: '30s', target: 10 },
      { duration: '30s', target: 20 },
      { duration: '30s', target: 30 },
      { duration: '30s', target: 0 },
    ],
  },
  spike: {
    stages: [
      { duration: '10s', target: 1 },
      { duration: '10s', target: 25 },
      { duration: '30s', target: 25 },
      { duration: '10s', target: 0 },
    ],
  },
  soak: {
    stages: [
      { duration: '1m', target: 5 },
      { duration: __ENV.K6_SOAK_DURATION || '10m', target: 5 },
      { duration: '1m', target: 0 },
    ],
  },
};

export function optionsFor(profileName = 'smoke', thresholds = localThresholds) {
  const profile = profiles[profileName] || profiles.smoke;
  return {
    ...profile,
    thresholds,
    noConnectionReuse: false,
  };
}
