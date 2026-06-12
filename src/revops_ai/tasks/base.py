"""Typed tasks and reports.

Tasks are the structured request surface of the SDK: every unit of work an
agent performs is described by a :class:`Task` subclass, and every result is a
:class:`Report` subclass. Natural-language entry points are routers that parse
text *into* a Task — never a parallel code path.

Reports are never bare verdicts: findings carry :class:`Evidence` so a GTM
engineer can show a VP *why*, and every report records the vintage of the data
it was computed from plus any writes the agent proposed.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from revops_ai.audit.evidence import EntityRef, Evidence, Finding, SourceRef
from revops_ai.safety.write_intent import WriteIntent

__all__ = [
    "EntityRef",
    "Evidence",
    "Finding",
    "Report",
    "SourceRef",
    "Task",
]


class Task(BaseModel):
    """Base class for all structured tasks.

    Subclass with the parameters your agent needs::

        class CommissionTask(Task):
            deal_id: str
    """

    model_config = ConfigDict(frozen=True)


class Report(BaseModel):
    """Base class for all agent outputs.

    ``run_id``, ``data_vintage``, and ``proposed_writes`` are stamped by the
    engine; agents do not set them.
    """

    run_id: str | None = None
    data_vintage: dict[str, datetime] = Field(default_factory=dict)
    proposed_writes: list[WriteIntent] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render an evidence-backed brief. Subclasses may override."""
        lines = [f"# {type(self).__name__}", ""]
        for role, synced_at in sorted(self.data_vintage.items()):
            lines.append(f"- data `{role}`: synced {synced_at.isoformat()}")
        if self.data_vintage:
            lines.append("")
        for name, value in self:
            if name in ("data_vintage", "run_id", "proposed_writes"):
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
        if self.proposed_writes:
            lines.append("")
            lines.append("## Proposed writes")
            for intent in self.proposed_writes:
                fields = ", ".join(sorted(intent.changes))
                lines.append(
                    f"- [{intent.status.value}] {intent.operation} "
                    f"{intent.target.entity_type} {intent.target.entity_id} ({fields})"
                )
        return "\n".join(lines).rstrip() + "\n"
