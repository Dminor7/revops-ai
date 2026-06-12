"""Typed tasks and reports.

Tasks are the structured request surface of the SDK: every unit of work an
agent performs is described by a :class:`Task` subclass, and every result is a
:class:`Report` subclass. Natural-language entry points are routers that parse
text *into* a Task — never a parallel code path.

Reports are never bare verdicts: findings carry :class:`Evidence` so a GTM
engineer can show a VP *why*, and every report records the vintage of the data
it was computed from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    """Base class for all structured tasks.

    Subclass with the parameters your agent needs::

        class CommissionTask(Task):
            deal_id: str
    """

    model_config = ConfigDict(frozen=True)


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


class Report(BaseModel):
    """Base class for all agent outputs.

    ``data_vintage`` is populated by the engine from connector sync metadata;
    agents do not set it themselves.
    """

    data_vintage: dict[str, datetime] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render an evidence-backed brief. Subclasses may override."""
        lines = [f"# {type(self).__name__}", ""]
        for role, synced_at in sorted(self.data_vintage.items()):
            lines.append(f"- data `{role}`: synced {synced_at.isoformat()}")
        if self.data_vintage:
            lines.append("")
        for name, value in self:
            if name == "data_vintage":
                continue
            if isinstance(value, list) and value and isinstance(value[0], Finding):
                for finding in value:
                    lines.append(
                        f"## {finding.subject.entity_type} {finding.subject.entity_id}: "
                        f"{finding.verdict} ({finding.confidence:.0%})"
                    )
                    lines.extend(f"- {ev.summary}" for ev in finding.evidence)
                    lines.append("")
            else:
                lines.append(f"- **{name}**: {value}")
        return "\n".join(lines).rstrip() + "\n"
