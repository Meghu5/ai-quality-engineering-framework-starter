# AI Airline Quality Engineering Framework

Enterprise-grade portfolio project for building a scalable quality engineering framework around airline APIs, chatbot experiences, and future AI testing layers.

## Current Scope
Phase 1 provides the deterministic API automation foundation:

- Environment-driven settings
- Reusable HTTPX API client
- Chat-specific API client
- Pydantic request/response models
- Reusable assertion helpers
- Pytest fixtures with clean client lifecycle
- Deterministic `/chat` API tests using mocked HTTP responses
- HTML report generation for API evidence

Phase 2 adds a realistic airline-domain API automation layer:

- Airline service clients built on the reusable `ApiClient`
- Pydantic models for flight search, availability, fares, passengers, and bookings
- Deterministic airline test data
- Business-rule assertions for flights, fares, availability, and booking/PNR validation
- Positive and negative API tests for meaningful airline scenarios

Future suites for UI, LLM, RAG, agents, and AI security are intentionally skipped until their phases are implemented.

## Prerequisites
- Python 3.14
- Git
- PowerShell or another terminal

Playwright is listed for future UI work. Browser installation is only needed when UI tests are enabled:

```powershell
python -m playwright install chromium
```

## Setup
From this project directory:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

## Environment Variables
`.env.example` documents supported configuration:

```text
API_BASE_URL=http://localhost:8000
CHAT_BASE_URL=http://localhost:8000
UI_BASE_URL=http://localhost:8000
API_TIMEOUT_SECONDS=30
API_LATENCY_THRESHOLD_SECONDS=5
API_AUTH_TOKEN=
```

Do not commit real secrets. `API_AUTH_TOKEN` is optional and is only sent when configured.

## Deterministic Local Test Strategy
Phase 1 and Phase 2 API tests do not depend on a live production service. They use `httpx.MockTransport` to stub responses inside the test process. This keeps:

- `python -m pytest -m api -v`
- `python -m pytest -m "api and airline" -v`
- `python -m pytest -m api --html=reports/api-report.html --self-contained-html`

reproducible on a local machine and in CI.

The mocks are only test transports. They are not airline backend or chatbot implementations.

## Running Tests
Collect tests:

```powershell
python -m pytest --collect-only -q
```

Run Phase 1 chat API tests:

```powershell
python -m pytest -m "api and not airline" -v
```

Run Phase 2 airline API tests:

```powershell
python -m pytest -m "api and airline" -v
```

Run all deterministic API tests:

```powershell
python -m pytest -m api -v
```

Generate a Phase 1 HTML report:

```powershell
python -m pytest -m api --html=reports/api-report.html --self-contained-html
```

Generate a Phase 2 airline API HTML report:

```powershell
python -m pytest -m "api and airline" --html=reports/airline-api-report.html --self-contained-html
```

Run all currently available tests:

```powershell
python -m pytest -v
```

Future-phase tests are collected but skipped until their implementations exist.

`pytest.ini` disables the legacy `pytest-html-reporter` plugin if it is installed globally because it is incompatible with pytest 9. HTML reports are generated with the project dependency `pytest-html`.

## Test Markers
- `api`: deterministic API tests
- `airline`: airline-domain API tests
- `flight`: flight search API tests
- `availability`: flight availability API tests
- `fare`: fare and pricing API tests
- `passenger`: passenger API tests
- `booking`: booking and PNR API tests
- `ui`: Playwright UI tests
- `llm`: semantic LLM quality tests
- `rag`: RAG evaluation tests
- `agents`: agent/tool-call tests
- `security`: defensive AI security tests

## Architecture
Phase 1 establishes this client pattern:

```text
ApiClient
  -> ChatClient
```

Phase 2 extends the same pattern for airline-domain APIs:

```text
ApiClient
  -> ChatClient
  -> Airline clients
       -> FlightClient
       -> AvailabilityClient
       -> FareClient
       -> PassengerClient
       -> BookingClient
```

The current deterministic airline flow is:

```text
flight search -> availability -> fare -> passenger -> booking / PNR
```

Current limitations:

- No real airline backend integration
- No payment, ticketing, seats, baggage, check-in, refunds, or loyalty APIs yet
- No UI/mobile/E2E integration yet
- No LLM/RAG/agent implementation yet

## Roadmap
1. Base API client, fixtures, deterministic API gates
2. Airline API clients, domain models, and business-rule validations
3. Playwright UI flow
4. Mobile/Appium testing
5. Airline business-domain E2E scenarios
6. LLM semantic evaluation
7. RAG retrieval and groundedness
8. Agent/tool validation
9. Kafka and event-driven tests
10. Database and contract testing
11. Security and performance testing
12. Docker, Kubernetes, Jenkins, observability, and quality gates
