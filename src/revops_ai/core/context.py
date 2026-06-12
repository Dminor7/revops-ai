"""RunContext: everything an agent may touch during a run.

The context is assembled by the engine per run. Agents reach connectors only
through the tools resolved from their ``requires`` declaration — there is no
back door to the full connector registry, which keeps the audit surface
(every tool call) equal to the data-access surface.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

from revops_ai.tools.base import Tool


class ToolNamespace:
    """Attribute access to the tools an agent declared, keyed by role.

    ``ctx.tools.warehouse`` returns the tool bound to the ``"warehouse"``
    connector. Access to an undeclared role raises immediately with the list
    of roles the agent did declare.
    """

    def __init__(self, tools: dict[str, Tool]) -> None:
        self._tools = tools

    def __getattr__(self, role: str) -> Tool:
        try:
            return self._tools[role]
        except KeyError:
            raise AttributeError(
                f"Agent did not declare a tool for role {role!r}; "
                f"declared roles: {sorted(self._tools)}"
            ) from None

    def __iter__(self) -> Iterator[tuple[str, Tool]]:
        return iter(self._tools.items())


@dataclass(frozen=True)
class RunContext:
    """Per-run execution context passed to ``BaseAgent.process_task``."""

    tools: ToolNamespace
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    data_vintage: dict[str, datetime] = field(default_factory=dict)
