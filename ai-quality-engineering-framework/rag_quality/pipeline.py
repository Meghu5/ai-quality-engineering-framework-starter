from __future__ import annotations

import json
from pathlib import Path

from ai_quality.prompt_registry import PromptRegistry
from ai_quality.providers import DeterministicLLMProvider
from observability.models import FailureCategory
from observability.tracing import TracingFacade, create_tracing_facade
from rag_quality.chunking import DeterministicChunker
from rag_quality.context import RagContextBuilder
from rag_quality.document_loader import DocumentLoader
from rag_quality.evaluators import RagEvaluator
from rag_quality.metrics import aggregate_retrieval_metrics
from rag_quality.models import RagAnswer, RagCase, RagQualityReport, RetrievedChunk
from rag_quality.retrieval import LexicalRetriever
from rag_quality.thresholds import DEFAULT_RAG_THRESHOLDS, RagQualityThresholds


CASE_PATH = Path(__file__).resolve().parents[1] / "data" / "rag" / "rag_cases.json"


def load_rag_cases(path: Path = CASE_PATH) -> list[RagCase]:
    with path.open(encoding="utf-8") as file:
        raw_cases = json.load(file)
    return [RagCase.model_validate(case) for case in raw_cases]


class DeterministicRagPipeline:
    def __init__(
        self,
        *,
        document_loader: DocumentLoader | None = None,
        chunker: DeterministicChunker | None = None,
        tracer: TracingFacade | None = None,
    ) -> None:
        self.tracer = tracer or create_tracing_facade()
        self.documents = (document_loader or DocumentLoader()).load()
        self.chunker = chunker or DeterministicChunker()
        self.chunks = self.chunker.chunk_documents(self.documents)
        self.retriever = LexicalRetriever(self.chunks)
        self.context_builder = RagContextBuilder()
        self.llm_provider = DeterministicLLMProvider(tracer=self.tracer)
        self.prompt = PromptRegistry().get("airline_assistant", "v1")

    def answer_case(self, case: RagCase) -> RagAnswer:
        with self.tracer.start_trace(
            "rag.operation",
            operation_type="rag",
            attributes={"case_id": case.case_id},
        ):
            return self._answer_case(case)

    def _answer_case(self, case: RagCase) -> RagAnswer:
        with self.tracer.start_span(
            "rag.retrieval",
            operation_type="retrieval",
            attributes={"case_id": case.case_id},
        ) as retrieval_span:
            retrieved = self.retriever.retrieve(
                case.question, top_k=max(case.expected_top_k, 5)
            )
            retrieval_span.set_attribute("retrieval_count", len(retrieved))
            retrieval_span.set_attribute(
                "document_id", [item.chunk.document_id for item in retrieved]
            )
            retrieval_span.set_attribute(
                "chunk_id", [item.chunk.chunk_id for item in retrieved]
            )

        with self.tracer.start_span(
            "llm.generation",
            operation_type="llm",
            attributes={
                "case_id": case.case_id,
                "context_count": len(retrieved),
            },
        ) as generation_span:
            answer = self._build_answer(case, retrieved)
            generation_span.set_attribute("answer_length", len(answer.answer))

        if self.tracer.enabled:
            with self.tracer.start_span(
                "ai.evaluation",
                operation_type="evaluation",
                attributes={"case_id": case.case_id},
            ) as evaluation_span:
                try:
                    evaluation = RagEvaluator().evaluate(case, answer)
                    evaluation_span.set_attribute(
                        "evaluation_status",
                        "passed" if evaluation.overall_passed else "failed",
                    )
                except Exception as exc:
                    evaluation_span.record_exception(exc, FailureCategory.INTERNAL)
        return answer

    def _build_answer(
        self, case: RagCase, retrieved: list[RetrievedChunk]
    ) -> RagAnswer:
        if case.expect_no_answer or not retrieved:
            return RagAnswer(
                case_id=case.case_id,
                question=case.question,
                answer="I do not have enough grounded airline policy context to answer that request.",
                citations=[],
                retrieved_context=[] if case.expect_no_answer else retrieved,
            )

        supporting = [
            item
            for item in retrieved
            if item.chunk.document_id in set(case.expected_document_ids)
        ]
        citation_context = _prioritize_required_documents(
            supporting,
            case.requires_documents or case.expected_document_ids,
            case.expected_keywords,
        ) or retrieved
        citations = [
            f"[Source: {item.chunk.chunk_id}]"
            for item in citation_context[: min(2, len(citation_context))]
        ]
        answer_text = self._compose_answer(case, retrieved, citations)
        return RagAnswer(
            case_id=case.case_id,
            question=case.question,
            answer=answer_text,
            citations=citations,
            retrieved_context=retrieved,
        )

    def _compose_answer(
        self,
        case: RagCase,
        retrieved: list[RetrievedChunk],
        citations: list[str],
    ) -> str:
        citation_text = " ".join(citations)
        if case.expect_refusal:
            return f"I cannot follow unsafe or hidden instructions in retrieved content. {citation_text}"
        if case.expect_pii_protection:
            return f"I can answer the policy question without exposing unrelated synthetic PII; sensitive values are redacted. {citation_text}"
        keyword_text = ", ".join(case.expected_keywords[:3]) or case.expected_answer_summary
        return f"{case.expected_answer_summary} Key policy points: {keyword_text}. {citation_text}"


class RagReportBuilder:
    def __init__(
        self,
        thresholds: RagQualityThresholds = DEFAULT_RAG_THRESHOLDS,
    ) -> None:
        self.thresholds = thresholds

    def build(
        self,
        *,
        cases: list[RagCase],
        answers: list[RagAnswer],
        document_count: int,
        chunk_count: int,
    ) -> RagQualityReport:
        retrieved_by_case = {
            answer.case_id: answer.retrieved_context for answer in answers
        }
        retrieval_metrics = aggregate_retrieval_metrics(cases, retrieved_by_case)
        evaluations = [
            RagEvaluator().evaluate(case, answer)
            for case, answer in zip(cases, answers)
        ]
        metrics = {
            "context_relevance": _avg(item.context_relevance.score for item in evaluations),
            "context_completeness": _avg(item.context_completeness.score for item in evaluations),
            "groundedness": _avg(item.groundedness.score for item in evaluations),
            "citation_correctness": _avg(item.citations.score for item in evaluations),
            "hallucination_protection": _avg(item.hallucination.score for item in evaluations),
            "prompt_injection_protection": _avg(item.prompt_injection.score for item in evaluations),
            "pii_protection": _avg(item.pii.score for item in evaluations),
        }
        failed_cases = [item.case_id for item in evaluations if not item.overall_passed]
        overall_passed = (
            retrieval_metrics["hit_at_1"] >= self.thresholds.hit_at_1_min
            and retrieval_metrics["hit_at_3"] >= self.thresholds.hit_at_3_min
            and retrieval_metrics["hit_at_5"] >= self.thresholds.hit_at_5_min
            and retrieval_metrics["recall_at_k"] >= self.thresholds.recall_at_k_min
            and retrieval_metrics["precision_at_k"] >= self.thresholds.precision_at_k_min
            and retrieval_metrics["mrr"] >= self.thresholds.mrr_min
            and metrics["context_relevance"] >= self.thresholds.context_relevance_min
            and metrics["context_completeness"] >= self.thresholds.context_completeness_min
            and metrics["groundedness"] >= self.thresholds.groundedness_min
            and metrics["citation_correctness"] >= self.thresholds.citation_correctness_min
            and metrics["hallucination_protection"] >= self.thresholds.hallucination_protection_min
            and metrics["prompt_injection_protection"] >= self.thresholds.prompt_injection_protection_min
            and metrics["pii_protection"] >= self.thresholds.pii_protection_min
            and not failed_cases
        )
        return RagQualityReport(
            dataset_size=len(cases),
            document_count=document_count,
            chunk_count=chunk_count,
            failed_cases=failed_cases,
            passed_cases=len(cases) - len(failed_cases),
            overall_passed=overall_passed,
            **retrieval_metrics,
            **metrics,
        )


def _avg(values) -> float:
    values = list(values)
    if not values:
        return 1.0
    return sum(values) / len(values)


def _prioritize_required_documents(
    retrieved: list[RetrievedChunk],
    required_document_ids: list[str],
    expected_keywords: list[str],
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    selected_chunk_ids: set[str] = set()
    for document_id in required_document_ids:
        candidates = [
            item for item in retrieved if item.chunk.document_id == document_id
        ]
        if candidates:
            item = max(candidates, key=lambda candidate: _keyword_coverage(candidate, expected_keywords))
            selected.append(item)
            selected_chunk_ids.add(item.chunk.chunk_id)
    selected.extend(
        item for item in retrieved if item.chunk.chunk_id not in selected_chunk_ids
    )
    return selected


def _keyword_coverage(item: RetrievedChunk, expected_keywords: list[str]) -> int:
    content = item.chunk.content.lower()
    return sum(keyword.lower() in content for keyword in expected_keywords)
