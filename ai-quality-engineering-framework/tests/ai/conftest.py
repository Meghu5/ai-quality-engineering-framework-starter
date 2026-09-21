from __future__ import annotations

import pytest

from ai_quality.dataset import load_golden_cases
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.providers import DeterministicLLMProvider


@pytest.fixture(scope="session")
def ai_golden_cases():
    return load_golden_cases()


@pytest.fixture(scope="session")
def ai_prompt():
    return PromptRegistry().get("airline_assistant", "v1")


@pytest.fixture(scope="session")
def deterministic_llm_provider():
    return DeterministicLLMProvider()


@pytest.fixture(scope="session")
def ai_responses(ai_golden_cases, ai_prompt, deterministic_llm_provider):
    return [
        deterministic_llm_provider.generate_structured(
            case.user_input,
            prompt=ai_prompt,
            case=case,
            context=case.context,
        )
        for case in ai_golden_cases
    ]
