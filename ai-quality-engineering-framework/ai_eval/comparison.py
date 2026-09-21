from __future__ import annotations

from collections import defaultdict

from ai_eval.models import CrossFrameworkComparison, FrameworkExecutionReport


CONCEPT_BY_METRIC = {
    "faithfulness": "faithfulness / groundedness",
    "groundedness": "faithfulness / groundedness",
    "answer_relevance": "answer relevance",
    "answer_relevancy": "answer relevance",
    "context_relevance": "context relevance",
    "contextual_relevancy": "context relevance",
    "context_precision": "context precision",
    "context_recall": "context recall",
    "hallucination": "hallucination",
}


def build_comparisons(framework_reports: list[FrameworkExecutionReport]) -> list[CrossFrameworkComparison]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for report in framework_reports:
        for result in report.results:
            if result.execution_status != "executed" or result.score is None:
                continue
            concept = CONCEPT_BY_METRIC.get(result.metric)
            if concept:
                grouped[concept][report.framework].append(result.score)

    comparisons: list[CrossFrameworkComparison] = []
    for concept, scores_by_framework in sorted(grouped.items()):
        if len(scores_by_framework) < 2:
            continue
        scores = {
            framework: sum(scores) / len(scores)
            for framework, scores in scores_by_framework.items()
        }
        comparisons.append(
            CrossFrameworkComparison(
                concept=concept,
                available_frameworks=sorted(scores),
                scores=scores,
                note=(
                    "Framework scores are shown side-by-side for shared concepts, "
                    "but they are not mathematically identical metrics."
                ),
            )
        )
    return comparisons
