# k6 Airline Performance Testing

Phase 6 adds k6 performance tests for deterministic airline API workflows.

These scripts target a local deterministic HTTP test double by default. They do not call Duffel, do not require `DUFFEL_ACCESS_TOKEN`, and do not create real airline bookings.

## Install k6

Windows with winget:

```powershell
winget install k6.k6
```

Windows with Chocolatey:

```powershell
choco install k6
```

macOS with Homebrew:

```bash
brew install k6
```

Linux:

Follow the official package instructions for your distribution:

```text
https://grafana.com/docs/k6/latest/set-up/install-k6/
```

Verify installation:

```powershell
k6 version
```

## Start the Local Performance Target

From `ai-quality-engineering-framework/`:

```powershell
python -m performance.apps.deterministic_airline_service --host 127.0.0.1 --port 8001
```

Health check:

```powershell
curl http://127.0.0.1:8001/health
```

This service is a deterministic test double aligned to the Phase 2 airline API contract:

- `POST /airline/flights/search`
- `GET /airline/flights/{flight_id}/availability`
- `GET /airline/fares/{fare_id}`
- `POST /airline/passengers`
- `POST /airline/bookings`

It is not a production airline backend.

## Run k6 Scenarios

Smoke:

```powershell
k6 run performance/k6/scenarios/smoke.js
```

Flight search:

```powershell
k6 run -e K6_PROFILE=load performance/k6/scenarios/flight_search.js
```

Availability:

```powershell
k6 run -e K6_PROFILE=load performance/k6/scenarios/availability.js
```

Fare:

```powershell
k6 run -e K6_PROFILE=load performance/k6/scenarios/fare.js
```

Booking:

```powershell
k6 run -e K6_PROFILE=load performance/k6/scenarios/booking.js
```

Override the target:

```powershell
k6 run -e AIRLINE_PERF_BASE_URL=http://127.0.0.1:8001 performance/k6/scenarios/smoke.js
```

## Profiles

Profiles are demonstration load shapes for a local deterministic service:

- `smoke`: 1 VU, short validation/baseline run
- `load`: ramp to 10 VUs, hold, ramp down
- `stress`: gradual increase beyond normal demo load
- `spike`: rapid jump to 25 VUs
- `soak`: sustained 5 VUs, explicitly invoked only

Use `K6_PROFILE` for non-smoke scripts:

```powershell
k6 run -e K6_PROFILE=spike performance/k6/scenarios/booking.js
```

Use `K6_SOAK_DURATION` to control soak duration:

```powershell
k6 run -e K6_PROFILE=soak -e K6_SOAK_DURATION=30m performance/k6/scenarios/booking.js
```

## Thresholds and Metrics

Thresholds are engineering demonstration thresholds, not production SLAs:

- `http_req_failed`: error rate gate
- `http_req_duration`: p90/p95/p99 latency gates
- `checks`: business validation gate
- `flight_search_success`
- `availability_success`
- `fare_success`
- `booking_success`

Percentiles matter because averages can hide bad tail latency. p95 and p99 make slow outliers visible, which is critical for customer-facing booking flows.

## Separation From Other Test Layers

Pytest API/UI/E2E tests validate functionality and orchestration.

k6 validates performance behavior, throughput, latency distribution, error rate, and business success rate under load.

Duffel real API tests remain isolated behind `real_api` and `DUFFEL_ACCESS_TOKEN`. Normal performance tests do not call Duffel.
