from __future__ import annotations

from typing import Any

from rag_quality.models import RetrievedChunk


class RagContextBuilder:
    def build(self, retrieved: list[RetrievedChunk]) -> list[dict[str, Any]]:
        return [
            {
                "document_id": item.chunk.document_id,
                "chunk_id": item.chunk.chunk_id,
                "title": item.chunk.title,
                "content": item.chunk.content,
                "metadata": {
                    "category": item.chunk.category,
                    "version": item.chunk.version,
                    "effective_date": item.chunk.effective_date,
                    "source": item.chunk.source,
                    "language": item.chunk.language,
                    "current": item.chunk.current,
                    "rank": item.rank,
                    "score": item.score,
                },
            }
            for item in retrieved
        ]

    def as_text(self, retrieved: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[Source: {item.chunk.chunk_id}]\n{item.chunk.content}" for item in retrieved
        )
