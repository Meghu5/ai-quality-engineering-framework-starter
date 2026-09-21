from __future__ import annotations

import pytest

from rag_quality.chunking import DeterministicChunker
from rag_quality.models import RagDocument


pytestmark = [pytest.mark.rag, pytest.mark.rag_retrieval]


def test_chunk_creation_and_metadata_propagation(rag_documents, rag_chunks):
    assert len(rag_chunks) >= len(rag_documents)
    assert 50 <= len(rag_chunks) <= 80
    first = rag_chunks[0]
    assert first.document_id == rag_documents[0].document_id
    assert first.chunk_id.startswith(f"{first.document_id}#chunk-")
    assert first.title == rag_documents[0].title
    assert first.source == rag_documents[0].source


def test_realistic_corpus_contains_multi_chunk_documents(rag_chunks):
    chunks_by_document: dict[str, list] = {}
    for chunk in rag_chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    multi_chunk_documents = [
        document_id
        for document_id, chunks in chunks_by_document.items()
        if len(chunks) > 1
    ]

    assert len(multi_chunk_documents) >= 10
    assert len(chunks_by_document["AIR-BAG-001"]) > 1
    assert [chunk.chunk_id for chunk in chunks_by_document["AIR-BAG-001"]] == [
        f"AIR-BAG-001#chunk-{index:03d}"
        for index in range(1, len(chunks_by_document["AIR-BAG-001"]) + 1)
    ]


def test_chunking_is_deterministic(rag_documents):
    chunker = DeterministicChunker(chunk_size=180, overlap=40)
    first = chunker.chunk_documents(rag_documents)
    second = chunker.chunk_documents(rag_documents)

    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]


def test_overlap_and_ordering_for_long_document():
    document = RagDocument(
        document_id="TEST-LONG-001",
        title="Long Test",
        category="test",
        version=1,
        effective_date="2026-01-01",
        source="test",
        content="A" * 300 + ". " + "B" * 300 + ". " + "C" * 300,
    )
    chunks = DeterministicChunker(chunk_size=250, overlap=50).chunk_document(document)

    assert len(chunks) > 1
    assert chunks[1].start <= chunks[0].end
    assert [chunk.chunk_id for chunk in chunks] == sorted(chunk.chunk_id for chunk in chunks)


def test_short_document_creates_single_chunk():
    document = RagDocument(
        document_id="TEST-SHORT-001",
        title="Short Test",
        category="test",
        version=1,
        effective_date="2026-01-01",
        source="test",
        content="Short policy.",
    )

    chunks = DeterministicChunker().chunk_document(document)

    assert len(chunks) == 1


def test_empty_content_is_rejected():
    with pytest.raises(ValueError):
        RagDocument(
            document_id="TEST-EMPTY-001",
            title="Empty",
            category="test",
            version=1,
            effective_date="2026-01-01",
            source="test",
            content=" ",
        )
