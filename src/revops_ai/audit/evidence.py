"""Evidence primitives: the structures that make findings defensible.

A finding without evidence cannot be constructed — this is enforced by the
schema, not by convention, so every verdict an agent produces can be traced to
the metric, record, retrieval, or judgment it rests on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EntityRef(BaseModel):
    """A pointer to a record in a source system (deal, account, contact...)."""

    model_config = ConfigDict(frozen=True)

    source_system: str
    entity_type: str
    entity_id: str
    url: str | None = None


class SourceRef(BaseModel):
    """Where a piece of evidence came from: a query, a record, a chunk."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["query", "record", "chunk", "model"]
    reference: str


class Evidence(BaseModel):
    """One load-bearing fact behind a finding."""

    kind: Literal["metric", "record", "retrieval", "llm_judgment"]
    summary: str
    source: SourceRef


class Finding(BaseModel):
    """A verdict about one entity, with the evidence that supports it."""

    subject: EntityRef
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(min_length=1)
