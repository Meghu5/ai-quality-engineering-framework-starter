from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from rag_quality.models import RagChunk, RetrievedChunk


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "for", "from", "how",
    "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "what", "when", "with", "do", "does", "after", "before", "will",
    "take",
}
SYNONYMS = {
    "bag": "baggage",
    "bags": "baggage",
    "luggage": "baggage",
    "carryon": "cabin",
    "carry": "cabin",
    "cancel": "cancellation",
    "cancelled": "cancellation",
    "canceled": "cancellation",
    "money": "refund",
    "payback": "refund",
    "kid": "minor",
    "child": "minor",
    "connection": "connection",
    "connecting": "connection",
    "connections": "connection",
    "late": "delayed",
    "delay": "delayed",
    "dangerous": "dangerous",
    "passport": "documentation",
    "documents": "documentation",
    "paperwork": "documentation",
    "bicycle": "bicycles",
    "bike": "bicycles",
    "bikes": "bicycles",
    "opens": "open",
    "opened": "open",
    "closes": "close",
    "closed": "close",
    "closing": "close",
    "rules": "rule",
    "deadline": "deadline",
    "deadlines": "deadline",
}


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> Counter[str]:
        """Return a deterministic sparse representation."""


class DeterministicSparseEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> Counter[str]:
        tokens = []
        for token in TOKEN_PATTERN.findall(text.lower()):
            if token in STOPWORDS:
                continue
            tokens.append(SYNONYMS.get(token, token))
        return Counter(tokens)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        """Return ranked context chunks for a query."""


class LexicalRetriever(Retriever):
    def __init__(
        self,
        chunks: list[RagChunk],
        *,
        embedding_provider: EmbeddingProvider | None = None,
        min_score: float = 0.12,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider or DeterministicSparseEmbeddingProvider()
        self.min_score = min_score
        self.chunk_vectors = [self.embedding_provider.embed(self._chunk_text(chunk)) for chunk in chunks]

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed(query)
        if not query.strip() or not query_vector or not self.chunks:
            return []

        scored: list[tuple[float, int, list[str]]] = []
        for index, chunk_vector in enumerate(self.chunk_vectors):
            score = _cosine(query_vector, chunk_vector)
            matched_terms = sorted(set(query_vector).intersection(chunk_vector))
            if score == 0 or not matched_terms:
                continue
            current_boost = 0.03 if self.chunks[index].current else -0.05
            category_boost = 0.03 if self.chunks[index].category in matched_terms else 0.0
            phrase_boost = _phrase_boost(query, self._chunk_text(self.chunks[index]))
            final_score = max(0.0, score + current_boost + category_boost + phrase_boost)
            if len(matched_terms) == 1 and final_score < 0.25:
                continue
            if final_score >= self.min_score:
                scored.append((final_score, index, matched_terms))

        scored.sort(
            key=lambda item: (
                item[0],
                self.chunks[item[1]].current,
                self.chunks[item[1]].version,
                -item[1],
            ),
            reverse=True,
        )
        scored = self._diversify_by_document(scored, top_k)
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                matched_terms=matched_terms,
            )
            for rank, (score, index, matched_terms) in enumerate(scored[:top_k], start=1)
        ]

    def _chunk_text(self, chunk: RagChunk) -> str:
        return f"{chunk.title} {chunk.category} {chunk.content} {chunk.content}"

    def _diversify_by_document(
        self,
        scored: list[tuple[float, int, list[str]]],
        top_k: int,
    ) -> list[tuple[float, int, list[str]]]:
        selected: list[tuple[float, int, list[str]]] = []
        selected_documents: set[str] = set()
        remaining: list[tuple[float, int, list[str]]] = []
        for item in scored:
            document_id = self.chunks[item[1]].document_id
            selected_count = sum(
                1 for selected_item in selected
                if self.chunks[selected_item[1]].document_id == document_id
            )
            if (
                (document_id not in selected_documents or selected_count < 2)
                and len(selected) < top_k
            ):
                selected.append(item)
                selected_documents.add(document_id)
            else:
                remaining.append(item)
        return selected + remaining


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _phrase_boost(query: str, chunk_text: str) -> float:
    query_terms = [
        token
        for token in TOKEN_PATTERN.findall(query.lower())
        if token not in STOPWORDS
    ]
    chunk_text = chunk_text.lower()
    boost = 0.0
    for left, right in zip(query_terms, query_terms[1:]):
        if f"{left} {right}" in chunk_text or f"{left} {right}s" in chunk_text:
            boost += 0.08
    return min(boost, 0.16)
