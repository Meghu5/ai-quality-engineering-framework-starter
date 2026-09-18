import pytest
import httpx
from pydantic import ValidationError

from clients.api_client import ApiClient
from clients.chat_client import ChatClient
from models.chat import ChatResponse
from utils.assertions import (
    assert_json_response,
    assert_latency_within,
    assert_non_empty_string_field,
    assert_required_fields,
    assert_status_code,
)


CHAT_ENDPOINT = "/chat"


def _mock_chat_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or request.url.path != CHAT_ENDPOINT:
            return httpx.Response(404, json={"error": "not found"})

        try:
            payload = request.read()
            body = httpx.Response(200, content=payload).json()
        except ValueError:
            return httpx.Response(400, json={"error": "malformed JSON"})

        if not isinstance(body, dict) or "prompt" not in body:
            return httpx.Response(422, json={"error": "prompt is required"})

        prompt = body["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            return httpx.Response(422, json={"error": "prompt must not be blank"})

        return httpx.Response(
            200,
            json={"answer": f"Mock airline assistant response for: {prompt}"},
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def deterministic_chat_client(settings):
    api_client = ApiClient(
        settings.api_base_url,
        timeout=settings.api_timeout_seconds,
        auth_token=settings.api_auth_token,
        transport=_mock_chat_transport(),
    )
    client = ChatClient(api_client)
    yield client
    client.close()


@pytest.mark.api
def test_chat_api_returns_valid_response(deterministic_chat_client, settings):
    response = deterministic_chat_client.ask("What is Playwright?")

    assert_status_code(response, 200, endpoint=CHAT_ENDPOINT)
    body = assert_json_response(response, endpoint=CHAT_ENDPOINT)
    assert_required_fields(body, ["answer"], endpoint=CHAT_ENDPOINT)
    assert_non_empty_string_field(body, "answer", endpoint=CHAT_ENDPOINT)
    ChatResponse.model_validate(body)
    assert_latency_within(
        response,
        settings.api_latency_threshold_seconds,
        endpoint=CHAT_ENDPOINT,
    )


@pytest.mark.api
def test_chat_api_rejects_empty_prompt_before_request(deterministic_chat_client):
    with pytest.raises(ValidationError, match="prompt must not be blank"):
        deterministic_chat_client.ask("   ")


@pytest.mark.api
def test_chat_api_rejects_missing_prompt_field(deterministic_chat_client):
    response = deterministic_chat_client.post_raw({})

    assert_status_code(response, 422, endpoint=CHAT_ENDPOINT)
    body = assert_json_response(response, endpoint=CHAT_ENDPOINT)
    assert body["error"] == "prompt is required", (
        f"{CHAT_ENDPOINT}: expected missing prompt error, got {body}"
    )


@pytest.mark.api
def test_chat_api_rejects_malformed_request_body(deterministic_chat_client):
    response = deterministic_chat_client.post_raw({"prompt": 123})

    assert_status_code(response, 422, endpoint=CHAT_ENDPOINT)
    body = assert_json_response(response, endpoint=CHAT_ENDPOINT)
    assert body["error"] == "prompt must not be blank", (
        f"{CHAT_ENDPOINT}: expected malformed prompt error, got {body}"
    )


@pytest.mark.api
def test_chat_client_surfaces_timeout_errors(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout", request=request)

    api_client = ApiClient(
        settings.api_base_url,
        timeout=0.001,
        transport=httpx.MockTransport(handler),
    )
    client = ChatClient(api_client)

    try:
        with pytest.raises(httpx.TimeoutException, match="simulated timeout"):
            client.ask("Will this timeout?")
    finally:
        client.close()
