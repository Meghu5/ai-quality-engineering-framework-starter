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

Phase 3 adds deterministic airline UI automation:

- A local static airline booking application rendered in the browser
- Playwright page objects for search, flight selection, fares, passengers, review, and confirmation
- Deterministic UI test data
- End-to-end booking validation with PNR and itinerary assertions

Phase 4 adds API + UI E2E orchestration:

- API-prepared airline state validated through the browser journey
- Browser-created airline bookings validated against API-domain responses
- Cross-layer consistency checks for flight, fare, passenger, PNR, and itinerary values
- Shared deterministic API transport and E2E helpers for reproducible orchestration

Phase 5 adds real airline API integration through Duffel TEST/SANDBOX mode:

- Duffel client methods for offer requests, offers, and orders
- Pydantic models for the Duffel response fields consumed by tests
- Mapping from Duffel responses into existing airline-domain models
- Opt-in real API tests guarded by `DUFFEL_ACCESS_TOKEN`

Phase 6 adds k6 performance testing:

- Local deterministic airline HTTP test double for k6 execution
- k6 smoke, load, stress, spike, and soak profiles
- Reusable k6 client, data, assertions, thresholds, and business metrics
- CI smoke performance gate that does not require Duffel credentials

Phase 7 adds deterministic API security testing:

- Authentication and authorization checks against the local airline HTTP target
- Safe injection-like and malformed input payloads treated only as strings
- BOLA-style passenger and booking access checks using synthetic identities
- Security header, HTTP method, data exposure, and abuse-case regression tests
- Optional OWASP ZAP baseline workflow against the local target only

Phase 8 adds deterministic AI/LLM quality engineering:

- Airline chatbot contract with structured intent, entities, actions, grounding, and safety fields
- Deterministic local LLM provider for CI-safe quality gates
- Prompt registry with versioned airline assistant prompts
- 25-case golden dataset for airline AI behavior
- Reusable evaluators for intent, entities, relevance, groundedness, safety, PII, prompt injection, hallucination, structured output, and tool/action validation
- AI quality thresholds and JSON/HTML reporting

Phase 9 adds deterministic RAG quality engineering:

- Airline policy knowledge base with document metadata and versioning
- Deterministic document loading, validation, and chunking
- Local lexical retrieval abstraction with deterministic sparse embeddings
- Hit@K, Recall@K, Precision@K, MRR, context relevance, and context completeness gates
- Grounded answer, citation, hallucination, prompt-injection, and PII-protection evaluators
- RAG quality report and CI workflow without external vector databases or paid APIs

Future agent suites are intentionally skipped until their phase is implemented.

## Prerequisites
- Python 3.14
- Git
- PowerShell or another terminal

Playwright powers the Phase 3 UI suite. Install Chromium before running UI tests locally:

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
DUFFEL_BASE_URL=https://api.duffel.com
DUFFEL_ACCESS_TOKEN=
DUFFEL_TIMEOUT_SECONDS=30
```

Do not commit real secrets. `API_AUTH_TOKEN` and `DUFFEL_ACCESS_TOKEN` are optional and are only sent when configured.

## Deterministic Local Test Strategy
Phase 1 and Phase 2 API tests do not depend on a live production service. They use `httpx.MockTransport` to stub responses inside the test process. This keeps:

- `python -m pytest -m api -v`
- `python -m pytest -m "api and airline" -v`
- `python -m pytest -m api --html=reports/api-report.html --self-contained-html`

reproducible on a local machine and in CI.

The mocks are only test transports. They are not airline backend or chatbot implementations.

Phase 3 UI tests do not depend on a live airline website. They exercise a local static HTML/JavaScript demo app under `test_apps/airline_ui/` using Playwright.

Phase 4 E2E tests intentionally combine the deterministic airline API clients with the local Playwright UI. They do not use a production airline site, external API, database, or web server.

## Phase 5 Real Airline API Integration
Phase 5 introduces Duffel TEST/SANDBOX API coverage without replacing deterministic regression tests.

Deterministic mode uses `httpx.MockTransport` for fast, offline regression. It is intended for normal local runs and CI:

```powershell
python -m pytest -m "api and not real_api" -v
```

Real API mode uses Duffel TEST/SANDBOX APIs for external contract validation:

```powershell
python -m pytest -m real_api -v
```

If `DUFFEL_ACCESS_TOKEN` is not configured, real API tests skip cleanly before making network calls. This project uses Duffel TEST/SANDBOX mode only. It does not create real production airline bookings.

Official Duffel documentation used by this integration:

- https://duffel.com/docs/api/overview/making-requests
- https://duffel.com/docs/api/offer-requests
- https://duffel.com/docs/api/offers
- https://duffel.com/docs/api/orders
- https://duffel.com/docs/api/overview/test-mode
- https://duffel.com/docs/api/overview/test-your-integration

Supported Phase 5 scenarios:

- Real offer request search through `POST /air/offer_requests`
- Offer retrieval through `GET /air/offers` and `GET /air/offers/{id}`
- Official Duffel no-flights test scenario: `PVD -> RAI`
- Official Duffel hold-order test scenario: `JFK -> EWR`
- Test-mode hold order creation through `POST /air/orders` when the returned offer supports hold orders

Known real API limitations:

- External sandbox availability is controlled by Duffel and airline sandboxes
- Tests validate structure and business invariants, not unstable live prices
- Real API tests require a Duffel test token that starts with `duffel_test_`

## Phase 6 k6 Performance Testing
Phase 6 introduces a separate k6 performance layer for airline API flows.

The default performance target is a local deterministic HTTP service:

```powershell
python -m performance.apps.deterministic_airline_service --host 127.0.0.1 --port 8001
```

This service reuses the deterministic airline API behavior and data. It is not a production airline backend and does not call Duffel.

Install k6 and verify it:

```powershell
k6 version
```

See detailed install/run instructions:

```text
performance/k6/README.md
```

Run the k6 smoke scenario:

```powershell
k6 run performance/k6/scenarios/smoke.js
```

Run a specific profile:

```powershell
k6 run -e K6_PROFILE=load performance/k6/scenarios/flight_search.js
k6 run -e K6_PROFILE=stress performance/k6/scenarios/booking.js
k6 run -e K6_PROFILE=spike performance/k6/scenarios/availability.js
k6 run -e K6_PROFILE=soak -e K6_SOAK_DURATION=30m performance/k6/scenarios/fare.js
```

The performance layer measures HTTP timing plus business success metrics:

- `flight_search_success`
- `availability_success`
- `fare_success`
- `booking_success`

Thresholds include error rate, checks, and p90/p95/p99 latency. They are engineering demonstration thresholds for the deterministic local service, not production SLAs.

## Phase 7 Security Testing and API Security
Phase 7 introduces a safe security testing layer for the airline API framework. It does not probe Duffel, production systems, public airline APIs, or any external target.

The security test target is the same deterministic local HTTP service used by Phase 6, started in explicit security mode:

```powershell
python -m performance.apps.deterministic_airline_service --host 127.0.0.1 --port 8001 --security-mode
```

Security mode enables a minimal deterministic authentication and authorization model using synthetic tokens:

- `valid-user-token`
- `another-user-token`
- `admin-test-token`
- `invalid-test-token`
- `expired-test-token`

These are not secrets and are never sent to an external service.

Phase 7 covers:

- Broken authentication: missing, malformed, invalid, and expired tokens
- Broken object level authorization: cross-user passenger and booking access
- Broken function level authorization: unsupported state-transition methods
- Broken object property level authorization: unexpected fields and excessive response fields
- Unrestricted resource consumption: bounded oversized input and deterministic rate-limit probe
- Security misconfiguration: required local security headers
- Unrestricted access to sensitive business flows: booking with another user's passenger
- Injection resilience: SQL-like, XSS-like, command-like, traversal-like, and template-like strings as inert inputs
- Error-message hygiene: no stack traces, filesystem paths, or database details

OWASP API Security categories are mapped as interview-ready coverage, not a claim of full production assurance. SSRF, unsafe third-party API consumption, and external inventory discovery are intentionally out of scope because Phase 7 has no external target.

Run the Phase 7 tests:

```powershell
python -m pytest -m "security and api" -v
```

Generate a Phase 7 HTML report:

```powershell
python -m pytest -m "security and api" -v --html=reports/security-api-report.html --self-contained-html
```

ZAP baseline scanning is optional and passive/bounded. CI only runs it when the repository variable `RUN_ZAP_BASELINE` is set to `true`; the target remains `http://127.0.0.1:8001`.

Phase distinction:

- Phase 2: functional deterministic airline API validation and negative business-rule tests
- Phase 5: opt-in Duffel TEST/SANDBOX integration, skipped without `DUFFEL_ACCESS_TOKEN`
- Phase 6: k6 performance testing against the local deterministic service
- Phase 7: API security testing against the local deterministic service in security mode

## Phase 8 AI / LLM Quality Engineering
Phase 8 introduces an AI quality layer for airline chatbot and assistant behavior. AI quality testing is different from traditional API testing because the test target may produce natural-language and structured outputs that must be checked for intent, relevance, grounding, safety, and data protection rather than only status codes and schemas.

The default provider is deterministic and local:

```text
DeterministicLLMProvider
  -> airline chatbot contract
  -> golden dataset
  -> deterministic evaluators
  -> quality thresholds
```

It does not call OpenAI, Anthropic, Gemini, Duffel, or any external airline system. It is a controlled provider for reproducible tests and CI gates, not a real LLM.

Phase 8 structure:

```text
ai_quality/
  providers.py
  models.py
  prompt_registry.py
  dataset.py
  evaluators.py
  thresholds.py

ai/prompts/
  airline_assistant_v1.txt
  airline_assistant_v2.txt

data/ai/
  golden_cases.json

tests/ai/
  test_intent.py
  test_entities.py
  test_structured_output.py
  test_groundedness.py
  test_relevance.py
  test_safety.py
  test_pii.py
  test_prompt_injection.py
  test_hallucination.py
  test_tool_actions.py
  test_quality_gate.py
```

The golden dataset contains 25 deterministic cases covering flight search, flight status, booking, cancellation, baggage, check-in, seat selection, fares, refunds, general help, unsupported requests, ambiguity, missing information, invalid airports/dates, multiple entities, multi-turn state, safety-sensitive prompts, prompt injection, synthetic PII, hallucination traps, unsupported policy questions, tool/action requests, and structured output.

Prompt versions are loaded through `PromptRegistry`, so tests can identify which prompt version produced the response. Current prompt versions:

- `airline_assistant_v1`: default deterministic quality gate prompt
- `airline_assistant_v2`: reserved traceable prompt variant for future comparison

Quality thresholds are centralized in `ai_quality/thresholds.py`:

- intent accuracy >= 0.95
- entity accuracy >= 0.95
- structured output validity = 1.00
- relevance score >= 0.95
- groundedness score >= 0.95
- safety pass rate = 1.00
- PII protection rate = 1.00
- prompt injection pass rate = 1.00
- hallucination pass rate >= 0.95

These are project quality gates for deterministic tests, not industry certification.

Run the Phase 8 AI quality tests:

```powershell
python -m pytest -m "ai" -v
```

Generate a Phase 8 HTML report:

```powershell
python -m pytest -m "ai" -v --html=reports/ai-quality-report.html --self-contained-html
```

The quality gate also writes:

```text
reports/ai-quality-report.json
```

Optional real LLM integration is represented by a provider interface and disabled adapter seam. It is not wired into the default suite and must be marked separately with `real_llm` when implemented. Phase 8 does not implement full RAG, vector search, LangChain, LangGraph, MCP, autonomous agents, or production LLM evaluation; those remain future phases.

## Phase 9 RAG Quality Engineering
Phase 9 introduces deterministic Retrieval-Augmented Generation quality testing for airline policy answers. RAG quality matters because a system can have a valid prompt and still fail if ingestion, chunking, retrieval, ranking, context assembly, citations, or grounded generation are weak.

The default Phase 9 stack is local and deterministic:

```text
data/rag/documents.json
  -> DocumentLoader
  -> DeterministicChunker
  -> DeterministicSparseEmbeddingProvider
  -> LexicalRetriever
  -> RagContextBuilder
  -> DeterministicRagPipeline
  -> RagEvaluator
  -> RagQualityReport
```

It does not require OpenAI, Anthropic, Gemini, FAISS, Chroma, pgvector, Elasticsearch/OpenSearch, Azure AI Search, Pinecone, or any external vector database.

Phase 9 validates:

- ingestion and document metadata
- malformed and duplicate documents
- chunk size, overlap, ordering, and metadata propagation
- deterministic retrieval and ranking
- Hit@1, Hit@3, Hit@5, Recall@K, Precision@K, and MRR
- context relevance and completeness
- groundedness against retrieved context
- citation existence, validity, and context alignment
- hallucination traps and contradictory claims
- prompt injection inside retrieved documents
- synthetic PII protection
- empty, malformed, unrelated, short, and long query failure modes
- stale versus current document version behavior

The deterministic knowledge base contains airline policy documents for baggage, check-in, seats, cancellations, refunds, fares, name correction, passenger types, special assistance, minors, sports equipment, dangerous goods, disruptions, travel documentation, loyalty, transfers, security test data, and privacy test data.

Run Phase 9 RAG tests:

```powershell
python -m pytest -m "rag" -v
python -m pytest -m "rag_retrieval" -v
python -m pytest -m "rag_quality" -v
python -m pytest -m "rag_security" -v
```

The RAG quality gate writes:

```text
reports/rag/rag_quality_report.json
```

Production RAG systems may later plug in real embedding providers or vector stores such as FAISS, Chroma, pgvector, Elasticsearch/OpenSearch, Azure AI Search, or Pinecone. Phase 9 deliberately keeps the default test suite dependency-light and deterministic so it can run locally and in CI.

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
python -m pytest -m "api and airline and not real_api" -v
```

Run all deterministic API tests:

```powershell
python -m pytest -m "api and not real_api" -v
```

Run Phase 3 airline UI tests:

```powershell
python -m pytest -m "ui and airline" -v
```

Run Phase 4 airline API + UI E2E tests:

```powershell
python -m pytest -m e2e -v
```

Run Phase 5 Duffel TEST/SANDBOX API tests:

```powershell
python -m pytest -m real_api -v
```

Run UI tests headed for debugging:

```powershell
python -m pytest -m "ui and airline" -v --headed
```

Generate a Phase 1 HTML report:

```powershell
python -m pytest -m api --html=reports/api-report.html --self-contained-html
```

Generate a Phase 2 airline API HTML report:

```powershell
python -m pytest -m "api and airline" --html=reports/airline-api-report.html --self-contained-html
```

Generate a Phase 3 airline UI HTML report:

```powershell
python -m pytest -m "ui and airline" -v --html=reports/airline-ui-report.html --self-contained-html
```

Generate a Phase 4 airline E2E HTML report:

```powershell
python -m pytest -m e2e -v --html=reports/airline-e2e-report.html --self-contained-html
```

Generate a Phase 5 Duffel TEST/SANDBOX API HTML report:

```powershell
python -m pytest -m real_api -v --html=reports/duffel-real-api-report.html --self-contained-html
```

Run the Phase 6 local deterministic performance service:

```powershell
python -m performance.apps.deterministic_airline_service --host 127.0.0.1 --port 8001
```

Run the Phase 6 k6 smoke test:

```powershell
k6 run performance/k6/scenarios/smoke.js
```

Run Phase 7 API security tests:

```powershell
python -m pytest -m "security and api" -v
```

Run Phase 8 AI quality tests:

```powershell
python -m pytest -m "ai" -v
```

Run Phase 9 RAG quality tests:

```powershell
python -m pytest -m "rag" -v
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
- `real_api`: real external airline API integration tests
- `flight`: flight search API tests
- `availability`: flight availability API tests
- `fare`: fare and pricing API tests
- `passenger`: passenger API tests
- `booking`: booking and PNR API tests
- `e2e`: end-to-end workflow tests
- `ui`: Playwright UI tests
- `llm`: semantic LLM quality tests
- `ai`: deterministic AI quality tests
- `real_llm`: optional real LLM provider tests
- `rag`: RAG evaluation tests
- `rag_retrieval`: RAG retrieval and ranking tests
- `rag_quality`: RAG answer quality and groundedness tests
- `rag_security`: RAG prompt-injection and PII tests
- `agents`: agent/tool-call tests
- `security`: defensive API and AI security tests

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
       -> DuffelClient
```

The current deterministic airline flow is:

```text
flight search -> availability -> fare -> passenger -> booking / PNR
```

Current limitations:

- No real airline backend integration
- No payment, ticketing, seats, baggage, check-in, refunds, or loyalty APIs yet
- UI automation uses a local deterministic demo app, not a production airline site
- No mobile/Appium integration yet
- No autonomous agent implementation yet; AI and RAG coverage is deterministic

Phase 3 UI structure:

```text
test_apps/airline_ui/
  index.html
  app.js
  styles.css

pages/airline/
  HomePage
  FlightResultsPage
  FarePage
  PassengerPage
  ReviewBookingPage
  ConfirmationPage
```

The local UI booking flow is:

```text
search -> flight results -> fare selection -> passenger details -> review -> confirmation / PNR
```

Phase 4 orchestrates both layers without making either layer depend on the other:

```text
API -> passenger/booking creation -> browser booking journey -> consistency assertions
UI  -> browser booking journey -> API booking creation -> consistency assertions
```

The shared deterministic E2E contract validates:

- origin and destination
- flight identifier
- fare identifier, currency, and total
- passenger name
- PNR
- final itinerary

Phase 5 keeps the same `ApiClient` foundation and adds a real-provider path:

```text
ApiClient
  -> DuffelClient
      -> Duffel TEST/SANDBOX REST API
      -> Duffel Pydantic models
      -> existing airline-domain models
      -> business assertions
```

Phase 6 keeps performance testing separate from pytest and real API integration:

```text
k6
  -> local deterministic airline HTTP service
      -> deterministic airline backend
      -> Phase 2 airline contract
      -> k6 thresholds and business metrics
```

Phase 7 reuses the same deterministic HTTP service with explicit security mode:

```text
pytest security suite
  -> local deterministic airline HTTP service --security-mode
      -> deterministic airline backend
      -> synthetic identity model
      -> reusable security assertions
```

Phase 8 adds a deterministic AI quality layer:

```text
tests/ai
  -> DeterministicLLMProvider
      -> airline chatbot response contract
      -> prompt registry
      -> golden dataset
      -> evaluators and thresholds
      -> AI quality report
```

Phase 9 adds deterministic RAG quality gates:

```text
tests/rag
  -> airline policy documents
      -> loader and metadata validation
      -> chunker
      -> local sparse retrieval
      -> context builder
      -> RAG answer and citations
      -> RAG evaluators and thresholds
      -> RAG quality report
```

CI runs API and UI automation separately:

- `.github/workflows/phase-1-api-tests.yml`: Phase 1 and Phase 2 deterministic API suites, excluding `real_api`
- `.github/workflows/phase-3-ui-tests.yml`: Phase 3 deterministic Playwright UI suite
- `.github/workflows/phase-4-e2e-tests.yml`: Phase 4 deterministic API + UI E2E suite
- `.github/workflows/phase-5-real-api-tests.yml`: Phase 5 Duffel TEST/SANDBOX API suite, requiring the `DUFFEL_ACCESS_TOKEN` GitHub secret
- `.github/workflows/phase-6-performance-tests.yml`: Phase 6 k6 smoke performance suite against the local deterministic service
- `.github/workflows/phase-7-security-tests.yml`: Phase 7 deterministic API security suite against the local service in security mode, plus optional ZAP baseline when explicitly enabled
- `.github/workflows/phase-8-ai-quality-tests.yml`: Phase 8 deterministic AI quality suite with HTML and JSON report artifacts
- `.github/workflows/phase-9-rag-quality-tests.yml`: Phase 9 deterministic RAG retrieval, quality, and security gates with report artifacts

## Roadmap
1. Base API client, fixtures, deterministic API gates
2. Airline API clients, domain models, and business-rule validations
3. Playwright airline UI flow
4. Airline API + UI E2E orchestration
5. Real airline API integration through Duffel TEST/SANDBOX
6. k6 performance testing
7. Security testing and API security
8. AI/LLM quality engineering
9. RAG retrieval and groundedness
10. Agent/tool validation
11. Mobile/Appium testing
12. Kafka and event-driven tests
13. Database and contract testing
14. Observability testing
15. Docker, Kubernetes, Jenkins, observability, and quality gates
