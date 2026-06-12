"""Tool base class.

Tools are how agents touch connectors. Agents declare ``requires = {role:
ToolClass}``; the engine resolves each declaration against the connector
registry at registration time and injects bound tool instances into the
:class:`~revops_ai.core.context.RunContext`. Tools are never constructed with
credentials in an agent class body — the same agent class can run against
different connectors in different engines (multi-tenant by construction).
"""

from __future__ import annotations

from typing import ClassVar

from revops_ai.connectors.base import Capability, Connector


class Tool:
    """A capability-checked handle on a connector, bound at registration.

    Subclasses set ``required_capabilities`` and add the methods agents call::

        class SQLQueryTool(Tool):
            async def query_one(self, sql: str, **params: object) -> dict[str, object]:
                ...
    """

    required_capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.READ})

    def __init__(self, connector: Connector, role: str) -> None:
        self.connector = connector
        self.role = role
