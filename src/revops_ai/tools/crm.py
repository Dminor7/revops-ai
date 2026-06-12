"""CRM tools: reads, and a marker for agents that propose CRM writes.

There is no write *method* on any tool — writes are proposed as intents via
``ctx.propose_write`` and applied by the engine after policy and approval.
``CRMWriteTool`` exists so an agent can *declare* it needs write capability on
the role; declaring it is what lets ``propose_write`` succeed.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from revops_ai.connectors.base import Capability
from revops_ai.tools.base import Tool


class SupportsCRMRead(Protocol):
    async def get_deal(
        self, deal_id: str, properties: list[str] | None = None
    ) -> dict[str, Any]: ...

    async def search_deals(
        self,
        *,
        stage: str | None = None,
        properties: list[str] | None = None,
        max_results: int = 200,
    ) -> list[dict[str, Any]]: ...


class CRMReadTool(Tool):
    """Deal reads against the bound CRM connector."""

    required_capabilities = frozenset({Capability.READ})

    async def get_deal(self, deal_id: str, properties: list[str] | None = None) -> dict[str, Any]:
        return await cast(SupportsCRMRead, self.connector).get_deal(deal_id, properties)

    async def search_deals(
        self,
        *,
        stage: str | None = None,
        properties: list[str] | None = None,
        max_results: int = 200,
    ) -> list[dict[str, Any]]:
        return await cast(SupportsCRMRead, self.connector).search_deals(
            stage=stage, properties=properties, max_results=max_results
        )


class CRMWriteTool(CRMReadTool):
    """Read access plus the grant to *propose* writes on this role."""

    required_capabilities = frozenset({Capability.READ, Capability.WRITE})
