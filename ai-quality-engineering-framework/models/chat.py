from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    """Current chat request contract with room for future chatbot context."""

    prompt: str = Field(..., min_length=1)
    message: str | None = None
    session_id: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    """Current chat response contract."""

    model_config = ConfigDict(extra="allow")

    answer: str = Field(..., min_length=1)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be blank")
        return value
