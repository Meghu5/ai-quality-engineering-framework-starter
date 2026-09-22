import pytest

from clients.api_client import ApiClient
from clients.chat_client import ChatClient
from config.settings import Settings
from observability.config import ObservabilitySettings
from observability.context import clear_context
from observability.exporters import InMemoryExporter
from observability.tracing import TracingFacade


@pytest.fixture(autouse=True)
def isolated_observability_context():
    clear_context()
    yield
    clear_context()


@pytest.fixture
def observability_exporter():
    return InMemoryExporter()


@pytest.fixture
def observability_tracer(observability_exporter):
    return TracingFacade(
        settings=ObservabilitySettings(enabled=True, exporter="memory"),
        exporter=observability_exporter,
    )


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
