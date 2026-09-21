from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatbotAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_ids: list[str] = Field(default_factory=list)
    grounded: bool = True


class SafetyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe: bool = True
    refusal: bool = False
    pii_detected: bool = False
    pii_redacted: bool = True
    prompt_injection_detected: bool = False


class AirlineAssistantResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    response: str
    entities: dict[str, Any] = Field(default_factory=dict)
    actions: list[ChatbotAction] = Field(default_factory=list)
    grounding: GroundingResult = Field(default_factory=GroundingResult)
    safety: SafetyResult = Field(default_factory=SafetyResult)
    prompt_version: str


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    user_input: str
    context: str | None = None
    expected_intent: str
    expected_entities: dict[str, Any] = Field(default_factory=dict)
    required_action: ChatbotAction | None = None
    reference_answer_terms: list[str] = Field(default_factory=list)
    required_context_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    expect_refusal: bool = False
    expect_pii: bool = False
    expected_missing_info: list[str] = Field(default_factory=list)
    structured_output_required: bool = True
    turns: list[str] = Field(default_factory=list)


class EvaluationCheck(BaseModel):
    name: str
    passed: bool
    details: str = ""


class QualityReport(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    intent_accuracy: float
    entity_accuracy: float
    structured_output_validity: float
    relevance_score: float
    groundedness_score: float
    safety_pass_rate: float
    pii_protection_rate: float
    prompt_injection_pass_rate: float
    hallucination_pass_rate: float
    overall_passed: bool
