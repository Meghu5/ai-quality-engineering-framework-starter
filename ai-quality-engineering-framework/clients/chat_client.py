import httpx


class ChatClient:
    """Reusable REST client for the application under test."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def ask(self, prompt: str) -> httpx.Response:
        return self.client.post(f"{self.base_url}/chat", json={"prompt": prompt})

    def close(self) -> None:
        self.client.close()
