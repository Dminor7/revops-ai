"""Billing tools (read-only)."""

from __future__ import annotations

from typing import Any, Protocol, cast

from revops_ai.connectors.base import Capability
from revops_ai.tools.base import Tool


class SupportsBillingRead(Protocol):
    async def get_customer(self, customer_id: str) -> dict[str, Any]: ...

    async def list_subscriptions(
        self,
        *,
        customer: str | None = None,
        status: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]: ...


class BillingReadTool(Tool):
    """Customer and subscription reads against the bound billing connector."""

    required_capabilities = frozenset({Capability.READ})

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        return await cast(SupportsBillingRead, self.connector).get_customer(customer_id)

    async def list_subscriptions(
        self,
        *,
        customer: str | None = None,
        status: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        return await cast(SupportsBillingRead, self.connector).list_subscriptions(
            customer=customer, status=status, max_results=max_results
        )
