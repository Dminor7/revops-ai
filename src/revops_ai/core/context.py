"""RunContext: everything an agent may touch during a run.

The context is assembled by the engine per run. Agents reach connectors only
through the tools resolved from their ``requires`` declaration — there is no
back door to the full connector registry, which keeps the audit surface
(every tool call) equal to the data-access surface.

Writes go through :meth:`RunContext.propose_write`: agents emit intents, never
side effects.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from revops_ai.connectors.base import Capability
from revops_ai.exceptions import CapabilityError
from revops_ai.safety.write_intent import WriteIntent


class ToolNamespace:
    """Attribute access to the tools an agent declared, keyed by role.

    ``ctx.tools.warehouse`` returns the tool bound to the ``"warehouse"``
    connector. Access to an undeclared role raises immediately with the list
    of roles the agent did declare.
    """

    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools

    def __getattr__(self, role: str) -> Any:
        try:
            return self._tools[role]
        except KeyError:
            raise AttributeError(
                f"Agent did not declare a tool for role {role!r}; "
                f"declared roles: {sorted(self._tools)}"
            ) from None

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self._tools.items())


@dataclass(frozen=True)
class RunContext:
    """Per-run execution context passed to ``BaseAgent.process_task``."""

    tools: ToolNamespace
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    data_vintage: dict[str, datetime] = field(default_factory=dict)
    proposed_writes: list[WriteIntent] = field(default_factory=list)

    def propose_write(self, intent: WriteIntent) -> WriteIntent:
        """Queue a write intent for the engine's policy pipeline.

        The role must be one the agent declared, and it must have been granted
        write capability at registration — read-only grants cannot even
        propose.
        """
        tool = getattr(self.tools, intent.connector_role)
        granted: frozenset[Capability] = getattr(tool, "granted_capabilities", frozenset())
        if Capability.WRITE not in granted:
            raise CapabilityError(
                f"Cannot propose a write to role {intent.connector_role!r}: the role "
                f"was granted {sorted(c.value for c in granted)} in this engine."
            )
        self.proposed_writes.append(intent)
        return intent
