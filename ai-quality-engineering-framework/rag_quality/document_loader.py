from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from rag_quality.models import RagDocument


DOCUMENT_PATH = Path(__file__).resolve().parents[1] / "data" / "rag" / "documents.json"


class DocumentLoadError(ValueError):
    """Raised when the deterministic RAG knowledge base is malformed."""


class DocumentLoader:
    def __init__(self, path: Path = DOCUMENT_PATH) -> None:
        self.path = path

    def load(self) -> list[RagDocument]:
        with self.path.open(encoding="utf-8") as file:
            raw_documents = json.load(file)
        return self.load_raw(raw_documents)

    def load_raw(self, raw_documents: list[dict]) -> list[RagDocument]:
        documents: list[RagDocument] = []
        seen_ids: set[str] = set()
        for raw_document in raw_documents:
            try:
                document = RagDocument.model_validate(raw_document)
            except ValidationError as exc:
                raise DocumentLoadError(str(exc)) from exc
            if document.document_id in seen_ids:
                raise DocumentLoadError(f"duplicate document_id: {document.document_id}")
            seen_ids.add(document.document_id)
            documents.append(document)
        if not documents:
            raise DocumentLoadError("knowledge base must contain at least one document")
        return documents
