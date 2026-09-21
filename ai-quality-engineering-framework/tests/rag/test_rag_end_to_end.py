from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_quality.evaluators import RagEvaluator
from rag_quality.pipeline import RagReportBuilder


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_representative_rag_end_to_end_cases_pass(rag_pipeline, rag_cases):
    representative_ids = {
        "rag-001",
        "rag-006",
        "rag-015",
        "rag-018",
        "rag-022",
        "rag-023",
        "rag-024",
        "rag-026",
        "rag-028",
        "rag-030",
    }
    evaluator = RagEvaluator()
    for case in rag_cases:
        if case.case_id not in representative_ids:
            continue
        answer = rag_pipeline.answer_case(case)
        assert evaluator.evaluate(case, answer).overall_passed, case.case_id


def test_rag_quality_gate_writes_report(rag_pipeline, rag_cases, rag_answers):
    report = RagReportBuilder().build(
        cases=rag_cases,
        answers=rag_answers,
        document_count=len(rag_pipeline.documents),
        chunk_count=len(rag_pipeline.chunks),
    )
    report_path = Path("reports") / "rag" / "rag_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")

    assert report.dataset_size == 30
    assert report.document_count == 21
    assert report.overall_passed
    assert not report.failed_cases
