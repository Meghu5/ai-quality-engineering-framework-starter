from __future__ import annotations

import json

import pytest

from ai_quality.prompt_registry import (
    Prompt,
    PromptRegistry,
    RegisteredPrompt,
    registered_prompt_identity,
)


def test_resolve_returns_immutable_registry_owned_prompt():
    registry = PromptRegistry()
    registered = registry.resolve("airline_assistant", "v1")

    assert isinstance(registered, RegisteredPrompt)
    assert registered.prompt == registry.get("airline_assistant", "v1")
    with pytest.raises(AttributeError, match="immutable"):
        registered.prompt = Prompt("other", "v1", "other", "other")
    with pytest.raises(AttributeError):
        registered.prompt.text = "replacement"


def test_metadata_equality_cannot_manufacture_registered_authority():
    registry = PromptRegistry()
    first = registry.resolve("airline_assistant", "v1")
    second = registry.resolve("airline_assistant", "v1")
    public = Prompt(
        name=first.prompt.name,
        version=first.prompt.version,
        purpose=first.prompt.purpose,
        text=first.prompt.text,
    )

    assert first is not second
    assert registered_prompt_identity(first) == ("airline_assistant", "v1")
    assert registered_prompt_identity(second) == ("airline_assistant", "v1")
    assert registered_prompt_identity(public) == ("unregistered", "0")
    with pytest.raises(TypeError, match="resolved by PromptRegistry"):
        RegisteredPrompt(public, object())


def test_registered_capability_is_not_serialized_or_exposed_by_repr():
    registered = PromptRegistry().resolve("airline_assistant", "v1")
    public_projection = {
        "name": registered.prompt.name,
        "version": registered.prompt.version,
    }
    serialized = json.dumps(public_projection, sort_keys=True)

    assert "capability" not in serialized
    assert "capability" not in repr(registered).lower()
    assert registered.prompt.text not in repr(registered)
    assert registered.prompt.purpose not in repr(registered)


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("airline_assistant", "v3"),
        ("AIRLINE_ASSISTANT", "v1"),
        ("unknown", "v1"),
    ],
)
def test_invalid_resolution_fails_deterministically(name, version):
    with pytest.raises(KeyError, match="registration not found"):
        PromptRegistry().resolve(name, version)
