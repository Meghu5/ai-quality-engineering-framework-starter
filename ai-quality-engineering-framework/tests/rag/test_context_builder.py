from __future__ import annotations

import pytest

from rag_quality.context import RagContextBuilder


pytestmark = [pytest.mark.rag, pytest.mark.rag_quality]


def test_context_builder_preserves_traceable_metadata(rag_retriever):
    retrieved = rag_retriever.retrieve("checked baggage economy 23 kg", top_k=2)
    context = RagContextBuilder().build(retrieved)

    assert context
    assert {"document_id", "chunk_id", "title", "content", "metadata"} <= set(context[0])
    assert "source" in context[0]["metadata"]
    assert "version" in context[0]["metadata"]


def test_context_text_contains_citations(rag_retriever):
    retrieved = rag_retriever.retrieve("online check-in", top_k=1)
    text = RagContextBuilder().as_text(retrieved)

    assert "[Source:" in text
    assert retrieved[0].chunk.chunk_id in text
