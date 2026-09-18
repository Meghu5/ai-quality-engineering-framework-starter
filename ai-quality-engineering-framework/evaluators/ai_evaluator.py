from dataclasses import dataclass


@dataclass
class EvaluationResult:
    relevance: int
    accuracy: int
    completeness: int
    hallucination: bool
    reason: str


def evaluate_with_model(prompt: str, answer: str, reference_answer: str) -> EvaluationResult:
    """Provider adapter seam.

    Replace this implementation with the chosen evaluator provider.
    Require strict structured JSON from the model and validate it before
    returning an EvaluationResult.
    """
    raise NotImplementedError("Connect the evaluator provider in Phase 3.")


def assert_quality(
    result: EvaluationResult,
    *,
    min_relevance: int = 8,
    min_accuracy: int = 8,
    min_completeness: int = 7,
) -> None:
    assert result.relevance >= min_relevance, (
        f"Relevance too low: {result.relevance}/10 - {result.reason}"
    )
    assert result.accuracy >= min_accuracy, (
        f"Accuracy too low: {result.accuracy}/10 - {result.reason}"
    )
    assert result.completeness >= min_completeness, (
        f"Completeness too low: {result.completeness}/10 - {result.reason}"
    )
    assert result.hallucination is False, (
        f"Hallucination detected - {result.reason}"
    )
