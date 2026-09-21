from __future__ import annotations

import pytest

from observability.redaction import REDACTED, redact_text, sanitize_attributes


pytestmark = pytest.mark.observability


def test_sensitive_and_raw_content_keys_are_not_emitted():
    attributes = sanitize_attributes(
        {
            "Authorization": "Bearer secret",
            "api_key": "key",
            "password": "password",
            "access_token": "token",
            "cookie": "session=x",
            "prompt": "raw prompt",
            "rag_context": "raw context",
            "model_response": "raw response",
        }
    )
    assert attributes == {}


def test_safe_metadata_remains_available():
    attributes = sanitize_attributes(
        {
            "case_id": "case-1",
            "document_id": "policy-1",
            "chunk_id": "policy-1-0001",
            "prompt_length": 22,
            "context_count": 3,
            "response_length": 81,
            "unknown_attribute": "discard me",
        }
    )
    assert attributes == {
        "case_id": "case-1",
        "document_id": "policy-1",
        "chunk_id": "policy-1-0001",
        "prompt_length": 22,
        "context_count": 3,
        "response_length": 81,
    }


def test_pii_inside_allowed_string_values_is_redacted():
    attributes = sanitize_attributes(
        {"case_id": "email user@example.com phone +1 (202) 555-0112"}
    )
    assert "user@example.com" not in attributes["case_id"]
    assert "555-0112" not in attributes["case_id"]
    assert REDACTED in attributes["case_id"]


def test_payment_card_text_is_redacted():
    assert "4111" not in redact_text("card 4111 1111 1111 1111")
