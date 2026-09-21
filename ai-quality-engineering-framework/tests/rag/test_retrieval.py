from __future__ import annotations

import pytest


pytestmark = [pytest.mark.rag, pytest.mark.rag_retrieval]


@pytest.mark.parametrize(
    ("query", "expected_document_id"),
    [
        ("cabin baggage allowance", "AIR-BAG-001"),
        ("online check-in opens", "AIR-CHECKIN-001"),
        ("wheelchair special assistance", "AIR-ASSIST-001"),
        ("dangerous flammable lithium batteries", "AIR-DANGER-001"),
        ("separate tickets missed connection", "AIR-XFER-001"),
    ],
)
def test_retrieves_relevant_documents(rag_retriever, query, expected_document_id):
    retrieved = rag_retriever.retrieve(query, top_k=3)

    assert expected_document_id in {item.chunk.document_id for item in retrieved}


def test_prefers_current_policy_over_stale_policy(rag_retriever):
    retrieved = rag_retriever.retrieve("current economy checked baggage one checked bag", top_k=3)

    assert retrieved[0].chunk.document_id == "AIR-BAG-002"
    assert "AIR-BAG-OLD-001" not in [item.chunk.document_id for item in retrieved[:1]]


def test_unrelated_query_returns_no_results(rag_retriever):
    assert rag_retriever.retrieve("mars hotel room volcano", top_k=3) == []


def test_empty_query_returns_no_results(rag_retriever):
    assert rag_retriever.retrieve("", top_k=3) == []
