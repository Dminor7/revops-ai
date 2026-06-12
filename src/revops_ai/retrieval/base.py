"""Vector store adapter protocol.

Any store that can upsert documents and run a similarity search satisfies the
SDK — pgvector and Qdrant implementations ship as extras; Pinecone/Weaviate
users implement the two methods below and never need to fork.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

MetadataValue = str | int | float | bool


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class ScoredDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: Document
    score: float


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStoreAdapter(Protocol):
    async def upsert(self, documents: Sequence[Document]) -> None: ...

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, MetadataValue] | None = None,
    ) -> list[ScoredDocument]: ...
