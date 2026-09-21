from __future__ import annotations

from typing import Any

from ai_eval.config import AIEvalSettings
from ai_eval.models import EvaluationResult, FrameworkExecutionReport
from ai_eval.package_detection import is_package_available, package_version
from ai_eval.provider import provider_from_name
from ai_eval.thresholds import DEFAULT_AI_EVAL_THRESHOLDS, AIEvalThresholds
from ai_quality.dataset import load_golden_cases
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.providers import DeterministicLLMProvider
from rag_quality.pipeline import DeterministicRagPipeline, load_rag_cases


class DeepEvalAdapter:
    framework = "deepeval"

    def __init__(
        self,
        *,
        settings: AIEvalSettings,
        thresholds: AIEvalThresholds = DEFAULT_AI_EVAL_THRESHOLDS,
    ) -> None:
        self.settings = settings
        self.thresholds = thresholds

    def build_ai_test_cases(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        cases = load_golden_cases()[:limit]
        prompt = PromptRegistry().get("airline_assistant", "v1")
        provider = DeterministicLLMProvider()
        rows = []
        for case in cases:
            response = provider.generate_structured(
                case.user_input,
                prompt=prompt,
                case=case,
                context=case.context,
            )
            rows.append(
                {
                    "case_id": case.id,
                    "input": case.user_input,
                    "actual_output": response.response,
                    "expected_output": " ".join(case.reference_answer_terms),
                    "context": [case.context] if case.context else [],
                    "retrieval_context": [case.context] if case.context else [],
                }
            )
        return rows

    def build_rag_test_cases(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        pipeline = DeterministicRagPipeline()
        rows = []
        for case in load_rag_cases()[:limit]:
            answer = pipeline.answer_case(case)
            rows.append(
                {
                    "case_id": case.case_id,
                    "input": case.question,
                    "actual_output": answer.answer,
                    "expected_output": case.expected_answer_summary,
                    "context": [item.chunk.content for item in answer.retrieved_context],
                    "retrieval_context": [item.chunk.content for item in answer.retrieved_context],
                }
            )
        return rows

    def evaluate(self) -> FrameworkExecutionReport:
        installed = is_package_available("deepeval")
        version = package_version("deepeval")
        if not installed:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=False,
                enabled=self.settings.deepeval_enabled,
                provider=self.settings.provider,
                status="unavailable",
                version=version,
                reason="DeepEval is not installed. Install optional AI evaluation dependencies to execute it.",
            )
        if not self.settings.deepeval_enabled:
            return FrameworkExecutionReport(
                framework=self.framework,
                installed=True,
                enabled=False,
                provider=self.settings.provider,
                status="skipped",
                version=version,
                reason="DeepEval is installed but AI_EVAL_DEEPEVAL_ENABLED is false.",
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
                    "DeepEval metrics such as AnswerRelevancy and Faithfulness "
                    "require an LLM judge; default provider is none."
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
                "Provider abstraction is configured, but no real DeepEval judge adapter "
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
