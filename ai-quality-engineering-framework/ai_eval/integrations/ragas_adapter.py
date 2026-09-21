from __future__ import annotations

from typing import Any

from ai_eval.config import AIEvalSettings
from ai_eval.models import EvaluationResult, FrameworkExecutionReport
from ai_eval.package_detection import is_package_available, package_version
from ai_eval.provider import provider_from_name
from ai_eval.thresholds import DEFAULT_AI_EVAL_THRESHOLDS, AIEvalThresholds
from rag_quality.pipeline import DeterministicRagPipeline, load_rag_cases


class RagasAdapter:
    framework = "ragas"

    def __init__(
        self,
        *,
        settings: AIEvalSettings,
        thresholds: AIEvalThresholds = DEFAULT_AI_EVAL_THRESHOLDS,
    ) -> None:
        self.settings = settings
        self.thresholds = thresholds

    def build_dataset(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        pipeline = DeterministicRagPipeline()
        rows: list[dict[str, Any]] = []
        for case in load_rag_cases()[:limit]:
            answer = pipeline.answer_case(case)
            rows.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "answer": answer.answer,
                    "contexts": [item.chunk.content for item in answer.retrieved_context],
                    "ground_truth": case.expected_answer_summary,
                    "expected_document_ids": case.expected_document_ids,
                    "retrieved_document_ids": [
                        item.chunk.document_id for item in answer.retrieved_context
                    ],
                }
            )
        return rows

    def evaluate(self) -> FrameworkExecutionReport:
        installed = is_package_available("ragas")
        version = package_version("ragas")
        if not installed:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=False,
                enabled=self.settings.ragas_enabled,
                provider=self.settings.provider,
                status="unavailable",
                version=version,
                reason="Ragas is not installed. Install optional AI evaluation dependencies to execute it.",
            )
        if not self.settings.ragas_enabled:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=True,
                enabled=False,
                provider=self.settings.provider,
                status="skipped",
                version=version,
                reason="Ragas is installed but AI_EVAL_RAGAS_ENABLED is false.",
            )

        provider = provider_from_name(self.settings.provider)
        if not provider.is_configured:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=True,
                enabled=True,
                provider=provider.name,
                status="provider_required",
                version=version,
                reason=(
                    "Ragas metrics such as faithfulness and answer relevance require "
                    "a configured judge/embedding provider; default provider is none."
                ),
            )

        return FrameworkExecutionReport(
            framework=self.framework,
            installed=True,
            enabled=True,
            provider=provider.name,
            status="provider_required",
            version=version,
            reason=(
                "Provider abstraction is configured, but no real Ragas provider adapter "
                "is implemented in Phase 10 to avoid paid API requirements."
            ),
        )
    def normalize_metric(
        self,
        *,
        case_id: str,
        metric: str,
        score: float,
        threshold: float,
        provider: str,
    ) -> EvaluationResult:
        return EvaluationResult(
            framework=self.framework,
            metric=metric,
            case_id=case_id,
            score=score,
            threshold=threshold,
            passed=score >= threshold,
            provider=provider,
            execution_status="executed",
        )
