from __future__ import annotations

import httpx
from pydantic import ValidationError

from clients.api_client import ApiClient
from models.chat import ChatRequest


class ChatClient:
    """Chat-specific API client built on the reusable API client."""

    endpoint = "/chat"

    def __init__(self, api_client: ApiClient) -> None:
        self.api_client = api_client

    def ask(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        context: dict | None = None,
        metadata: dict | None = None,
    ) -> httpx.Response:
        request = ChatRequest(
            prompt=prompt,
            session_id=session_id,
            context=context,
            metadata=metadata,
        )
        return self.api_client.post(self.endpoint, json=request.model_dump(exclude_none=True))

    def post_raw(self, payload: dict | None) -> httpx.Response:
        return self.api_client.post(self.endpoint, json=payload)

    def ask_with_client_validation_error(self, prompt: str) -> ValidationError:
        try:
            ChatRequest(prompt=prompt)
        except ValidationError as exc:
            return exc

        raise AssertionError("Expected chat request validation to fail")

    def close(self) -> None:
        self.api_client.close()
