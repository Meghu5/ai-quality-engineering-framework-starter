from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ExecutionStatus = Literal[
    "executed",
    "not_executed",
    "skipped",
    "unavailable",
    "provider_required",
    "failed",
]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    metric: str
    case_id: str
    score: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    reason: str = ""
    provider: str = "none"
    execution_status: ExecutionStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrameworkExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    installed: bool
    enabled: bool
    provider: str = "none"
    status: ExecutionStatus
    version: str | None = None
    results: list[EvaluationResult] = Field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrossFrameworkComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept: str
    available_frameworks: list[str]
    note: str
    scores: dict[str, float] = Field(default_factory=dict)


class Phase10Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: dict[str, Any]
    frameworks: list[FrameworkExecutionReport]
    comparisons: list[CrossFrameworkComparison] = Field(default_factory=list)
    overall_passed: bool
