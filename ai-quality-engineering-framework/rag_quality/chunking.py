from __future__ import annotations

from rag_quality.models import RagChunk, RagDocument


class DeterministicChunker:
    def __init__(self, *, chunk_size: int = 520, overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: RagDocument) -> list[RagChunk]:
        content = document.content.strip()
        if not content:
            raise ValueError("cannot chunk empty document content")

        chunks: list[RagChunk] = []
        start = 0
        chunk_number = 1
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            if end < len(content):
                sentence_boundary = content.rfind(". ", start, end)
                if sentence_boundary > start + int(self.chunk_size * 0.5):
                    end = sentence_boundary + 1
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk_id = f"{document.document_id}#chunk-{chunk_number:03d}"
                chunks.append(
                    RagChunk(
                        document_id=document.document_id,
                        chunk_id=chunk_id,
                        title=document.title,
                        category=document.category,
                        version=document.version,
                        effective_date=document.effective_date,
                        source=document.source,
                        language=document.language,
                        current=document.current,
                        content=chunk_content,
                        start=start,
                        end=end,
                    )
                )
                chunk_number += 1
            if end >= len(content):
                break
            start = max(0, end - self.overlap)
        return chunks

    def chunk_documents(self, documents: list[RagDocument]) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks
