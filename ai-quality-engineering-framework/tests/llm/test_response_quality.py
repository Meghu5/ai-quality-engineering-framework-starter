import json
import pytest
from evaluators.ai_evaluator import evaluate_with_model, assert_quality

pytestmark = pytest.mark.skip(reason="Future phase: LLM judge is not implemented in Phase 1.")

with open("data/prompts.json", encoding="utf-8") as file:
    CASES = json.load(file)


@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_chatbot_semantic_quality(chat_client, case):
    response = chat_client.ask(case["prompt"])
    assert response.status_code == 200
    answer = response.json()["answer"]

    evaluation = evaluate_with_model(
        prompt=case["prompt"],
        answer=answer,
        reference_answer=case["reference_answer"],
    )

    assert_quality(
        evaluation,
        min_relevance=case.get("min_relevance", 8),
        min_accuracy=case.get("min_accuracy", 8),
        min_completeness=case.get("min_completeness", 7),
    )
