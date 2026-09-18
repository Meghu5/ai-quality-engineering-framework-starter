# AI Airline Quality Engineering Framework

Enterprise-grade portfolio project for building a scalable quality engineering framework around airline APIs, chatbot experiences, and future AI testing layers.

## Current Scope
Phase 1 focuses on a deterministic API automation foundation:

- Environment-driven settings
- Reusable HTTPX API client
- Chat-specific API client
- Pydantic request/response models
- Reusable assertion helpers
- Pytest fixtures with clean client lifecycle
- Deterministic `/chat` API tests using mocked HTTP responses
- HTML report generation for API evidence

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
Phase 1 API tests do not depend on a live production service. They use `httpx.MockTransport` to stub `/chat` responses inside the test process. This keeps:

- `python -m pytest -m api -v`
- `python -m pytest -m api --html=reports/api-report.html --self-contained-html`

reproducible on a local machine and in CI.

The mock is only a test transport. It is not an airline backend or chatbot implementation.

## Running Tests
Collect tests:

```powershell
python -m pytest --collect-only -q
```

Run Phase 1 API tests:

```powershell
python -m pytest -m api -v
```

Generate a Phase 1 HTML report:

```powershell
python -m pytest -m api --html=reports/api-report.html --self-contained-html
```

Run all currently available tests:

```powershell
python -m pytest -v
```

Future-phase tests are collected but skipped until their implementations exist.

`pytest.ini` disables the legacy `pytest-html-reporter` plugin if it is installed globally because it is incompatible with pytest 9. HTML reports are generated with the project dependency `pytest-html`.

## Test Markers
- `api`: deterministic API tests
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

Later phases can add service-specific clients without changing the base contract:

```text
ApiClient
  -> FlightClient
  -> BookingClient
  -> PassengerClient
  -> PaymentClient
  -> ChatClient
```

## Roadmap
1. Base API client, fixtures, deterministic API gates
2. Playwright UI flow
3. Mobile/Appium testing
4. Airline business-domain E2E scenarios
5. LLM semantic evaluation
6. RAG retrieval and groundedness
7. Agent/tool validation
8. Kafka and event-driven tests
9. Database and contract testing
10. Security and performance testing
11. Docker, Kubernetes, Jenkins, observability, and quality gates
