"""In-memory vector store: the reference adapter, and fine for tests/demos."""

from __future__ import annotations

import math
from collections.abc import Sequence

from revops_ai.retrieval.base import Document, Embedder, MetadataValue, ScoredDocument


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._docs: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    async def upsert(self, documents: Sequence[Document]) -> None:
        vectors = await self._embedder.embed([d.text for d in documents])
        for doc, vector in zip(documents, vectors, strict=True):
            self._docs[doc.id] = doc
            self._vectors[doc.id] = vector

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[ScoredDocument]:
        (query_vector,) = await self._embedder.embed([query])
        candidates = [
            doc
            for doc in self._docs.values()
            if not where or all(doc.metadata.get(key) == value for key, value in where.items())
        ]
        scored = [
            ScoredDocument(document=doc, score=_cosine(query_vector, self._vectors[doc.id]))
            for doc in candidates
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:k]
