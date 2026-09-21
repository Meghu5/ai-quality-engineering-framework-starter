from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RagDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    category: str
    version: int = Field(..., ge=1)
    effective_date: str
    source: str
    language: str = "en"
    current: bool = True
    content: str

    @field_validator("document_id", "title", "category", "source", "language", "content")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required text field must not be blank")
        return value.strip()


class RagChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    title: str
    category: str
    version: int
    effective_date: str
    source: str
    language: str = "en"
    current: bool = True
    content: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: RagChunk
    score: float
    rank: int
    matched_terms: list[str] = Field(default_factory=list)


class RagCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    expected_top_k: int = Field(default=3, ge=1)
    category: str
    difficulty: str
    expected_answer_summary: str
    requires_documents: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    expect_no_answer: bool = False
    expect_refusal: bool = False
    expect_pii_protection: bool = False
    expect_prompt_injection_resistance: bool = False


class RagAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)


class MetricResult(BaseModel):
    score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class RagCaseEvaluation(BaseModel):
    case_id: str
    retrieval: dict[str, float | bool]
    context_relevance: MetricResult
    context_completeness: MetricResult
    groundedness: MetricResult
    citations: MetricResult
    hallucination: MetricResult
    safety: MetricResult
    pii: MetricResult
    prompt_injection: MetricResult
    overall_passed: bool


class RagQualityReport(BaseModel):
    dataset_size: int
    document_count: int
    chunk_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    context_relevance: float
    context_completeness: float
    groundedness: float
    citation_correctness: float
    hallucination_protection: float
    prompt_injection_protection: float
    pii_protection: float
    passed_cases: int
    failed_cases: list[str] = Field(default_factory=list)
    overall_passed: bool
