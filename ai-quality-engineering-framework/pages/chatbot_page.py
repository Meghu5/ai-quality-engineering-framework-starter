class ChatbotPage:
    def __init__(self, page):
        self.page = page
        self.prompt_box = page.get_by_test_id("chat-input")
        self.send_button = page.get_by_test_id("send-message")
        self.answer = page.get_by_test_id("assistant-answer")

    def ask(self, prompt: str) -> str:
        self.prompt_box.fill(prompt)
        self.send_button.click()
        self.answer.wait_for(state="visible")
        return self.answer.inner_text()
