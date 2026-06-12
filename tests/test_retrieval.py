"""Vector store adapter contract via the in-memory implementation."""

from __future__ import annotations

from collections.abc import Sequence

from revops_ai.retrieval import Document, InMemoryVectorStore


class KeywordEmbedder:
    """Deterministic embedder: one dimension per known keyword."""

    keywords = ("pricing", "churn", "renewal")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(kw in text.lower()) for kw in self.keywords] for text in texts]


async def test_upsert_and_search_ranks_by_similarity() -> None:
    store = InMemoryVectorStore(embedder=KeywordEmbedder())
    await store.upsert(
        [
            Document(id="1", text="Pricing objections in the renewal call"),
            Document(id="2", text="Churn risk flagged by support volume"),
            Document(id="3", text="Renewal scheduled, no concerns"),
        ]
    )

    results = await store.search("why is churn happening", k=2)
    assert results[0].document.id == "2"
    assert results[0].score > results[1].score


async def test_upsert_overwrites_by_id_and_filters_by_metadata() -> None:
    store = InMemoryVectorStore(embedder=KeywordEmbedder())
    await store.upsert([Document(id="1", text="churn", metadata={"account": "acme"})])
    await store.upsert([Document(id="1", text="pricing churn", metadata={"account": "acme"})])
    await store.upsert([Document(id="2", text="churn", metadata={"account": "globex"})])

    results = await store.search("churn", k=5, where={"account": "acme"})
    assert [r.document.id for r in results] == ["1"]
    assert "pricing" in results[0].document.text
