import pytest


@pytest.mark.api
def test_chat_api_returns_valid_response(chat_client):
    response = chat_client.ask("What is Playwright?")
    assert response.status_code == 200, response.text

    body = response.json()
    assert "answer" in body
    assert body["answer"].strip()
    assert response.elapsed.total_seconds() < 5
