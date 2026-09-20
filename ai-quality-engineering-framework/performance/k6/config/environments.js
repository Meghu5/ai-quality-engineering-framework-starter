export const environments = {
  local: {
    baseUrl: __ENV.AIRLINE_PERF_BASE_URL || 'http://127.0.0.1:8001',
  },
};

export function currentEnvironment() {
  const name = __ENV.AIRLINE_PERF_ENV || 'local';
  return environments[name] || environments.local;
}
