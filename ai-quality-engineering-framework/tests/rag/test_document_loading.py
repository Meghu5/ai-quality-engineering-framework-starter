from __future__ import annotations

import pytest

from rag_quality.document_loader import DocumentLoadError, DocumentLoader


pytestmark = [pytest.mark.rag, pytest.mark.rag_retrieval]


def test_loads_valid_airline_documents(rag_documents):
    assert len(rag_documents) == 21
    assert all(document.document_id for document in rag_documents)
    assert all(document.content for document in rag_documents)


@pytest.mark.parametrize("missing_field", ["document_id", "title", "content"])
def test_rejects_missing_required_document_fields(rag_documents, missing_field):
    raw = [rag_documents[0].model_dump()]
    raw[0].pop(missing_field)

    with pytest.raises(DocumentLoadError):
        DocumentLoader().load_raw(raw)


def test_rejects_duplicate_document_ids(rag_documents):
    raw = [rag_documents[0].model_dump(), rag_documents[0].model_dump()]

    with pytest.raises(DocumentLoadError, match="duplicate document_id"):
        DocumentLoader().load_raw(raw)


def test_rejects_empty_document_list():
    with pytest.raises(DocumentLoadError):
        DocumentLoader().load_raw([])


def test_rejects_invalid_metadata(rag_documents):
    raw = [rag_documents[0].model_dump()]
    raw[0]["version"] = 0

    with pytest.raises(DocumentLoadError):
        DocumentLoader().load_raw(raw)
