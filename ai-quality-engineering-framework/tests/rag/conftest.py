from __future__ import annotations

import pytest

from rag_quality.chunking import DeterministicChunker
from rag_quality.document_loader import DocumentLoader
from rag_quality.pipeline import DeterministicRagPipeline, load_rag_cases
from rag_quality.retrieval import LexicalRetriever


@pytest.fixture(scope="session")
def rag_documents():
    return DocumentLoader().load()


@pytest.fixture(scope="session")
def rag_chunker():
    return DeterministicChunker()


@pytest.fixture(scope="session")
def rag_chunks(rag_documents, rag_chunker):
    return rag_chunker.chunk_documents(rag_documents)


@pytest.fixture(scope="session")
def rag_retriever(rag_chunks):
    return LexicalRetriever(rag_chunks)


@pytest.fixture(scope="session")
def rag_cases():
    return load_rag_cases()


@pytest.fixture(scope="session")
def rag_pipeline():
    return DeterministicRagPipeline()


@pytest.fixture(scope="session")
def rag_answers(rag_pipeline, rag_cases):
    return [rag_pipeline.answer_case(case) for case in rag_cases]
