import pytest

from clients.api_client import ApiClient
from clients.chat_client import ChatClient
from config.settings import Settings


@pytest.fixture(scope="session")
def settings():
    return Settings.from_env()


@pytest.fixture(scope="session")
def api_client(settings):
    client = ApiClient(
        settings.api_base_url,
        timeout=settings.api_timeout_seconds,
        auth_token=settings.api_auth_token,
    )
    yield client
    client.close()


@pytest.fixture(scope="session")
def chat_client(api_client):
    return ChatClient(api_client)


@pytest.fixture
def browser_page(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    yield page
    browser.close()
