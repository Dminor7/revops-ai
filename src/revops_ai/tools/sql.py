"""SQL query tool for warehouse-role connectors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from revops_ai.connectors.base import Capability
from revops_ai.tools.base import Tool


class SupportsSQL(Protocol):
    async def query(
        self, sql: str, params: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]: ...


class SQLQueryTool(Tool):
    """Read queries against the bound warehouse connector."""

    required_capabilities = frozenset({Capability.READ})

    async def query(self, sql: str, **params: object) -> list[dict[str, Any]]:
        return await cast(SupportsSQL, self.connector).query(sql, params)

    async def query_one(self, sql: str, **params: object) -> dict[str, Any]:
        rows = await self.query(sql, **params)
        if not rows:
            raise LookupError(f"Query returned no rows: {sql!r} with {params!r}")
        return rows[0]
