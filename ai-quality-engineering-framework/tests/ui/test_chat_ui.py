import os
import pytest
from pages.chatbot_page import ChatbotPage


@pytest.mark.ui
def test_chatbot_end_to_end(browser_page):
    base_url = os.getenv("CHAT_BASE_URL", "http://localhost:8000")
    browser_page.goto(base_url)

    chatbot = ChatbotPage(browser_page)
    answer = chatbot.ask("What is Playwright?")
    assert answer.strip()
