from __future__ import annotations

from ai_quality.dataset import load_golden_cases
from ai_quality.evaluators import QualityGateEvaluator
from ai_quality.prompt_registry import PromptRegistry
from ai_quality.providers import DeterministicLLMProvider
from rag_quality.pipeline import DeterministicRagPipeline, RagReportBuilder, load_rag_cases


def build_phase8_report() -> dict:
    cases = load_golden_cases()
    prompt = PromptRegistry().get("airline_assistant", "v1")
    provider = DeterministicLLMProvider()
    responses = [
        provider.generate_structured(case.user_input, prompt=prompt, case=case, context=case.context)
        for case in cases
    ]
    report = QualityGateEvaluator().evaluate(cases, responses)
    return report.model_dump(mode="json")


def build_phase9_report() -> dict:
    pipeline = DeterministicRagPipeline()
    cases = load_rag_cases()
    answers = [pipeline.answer_case(case) for case in cases]
    report = RagReportBuilder().build(
        cases=cases,
        answers=answers,
        document_count=len(pipeline.documents),
        chunk_count=len(pipeline.chunks),
    )
    return report.model_dump(mode="json")


def build_baseline_report() -> dict:
    phase8 = build_phase8_report()
    phase9 = build_phase9_report()
    phase8["execution_status"] = "executed"
    phase9["execution_status"] = "executed"
    return {
        "phase8": phase8,
        "phase9": phase9,
        "overall_passed": bool(phase8["overall_passed"] and phase9["overall_passed"]),
    }
