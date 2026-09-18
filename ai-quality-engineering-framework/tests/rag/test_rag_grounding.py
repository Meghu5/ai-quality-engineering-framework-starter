import pytest

pytestmark = pytest.mark.skip(reason="Future phase: RAG testing is not implemented yet.")


@pytest.mark.rag
def test_answer_is_grounded(rag_client, evaluator):
    result = rag_client.ask("What is our refund policy?")
    assert result.retrieved_chunks

    score = evaluator.groundedness(
        question="What is our refund policy?",
        answer=result.answer,
        context=result.retrieved_chunks,
    )
    assert score >= 8, f"Groundedness too low: {score}/10"
