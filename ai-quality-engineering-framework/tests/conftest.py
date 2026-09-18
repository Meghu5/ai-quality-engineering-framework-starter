import os
import pytest
from dotenv import load_dotenv
from clients.chat_client import ChatClient

load_dotenv()


@pytest.fixture(scope="session")
def chat_base_url():
    return os.getenv("CHAT_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def chat_client(chat_base_url):
    client = ChatClient(chat_base_url)
    yield client
    client.close()


@pytest.fixture
def browser_page(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    yield page
    browser.close()
