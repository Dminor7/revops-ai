"""Read-only SQL warehouse connector on SQLAlchemy 2.x async.

Works with any database SQLAlchemy supports an async driver for (Postgres via
psycopg/asyncpg, Snowflake, BigQuery, SQLite for tests). Install the
``warehouse`` extra plus your driver.

The connector is deliberately read-only: warehouse write-back is not a GTM
agent concern, and keeping writes impossible here keeps the write-intent
pipeline the only write path in the SDK.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from revops_ai.connectors.base import Capability, SyncMetadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def _require_sqlalchemy() -> Any:
    try:
        import sqlalchemy
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "WarehouseConnector requires SQLAlchemy. Install it with: "
            "pip install 'revops-ai[warehouse]' plus an async driver "
            "(e.g. asyncpg, aiosqlite, snowflake-sqlalchemy)."
        ) from exc
    return sqlalchemy


class WarehouseConnector:
    """Read-only query access to a SQL warehouse.

    ``sync_resolver`` lets you surface your ELT tooling's real sync timestamp
    (Fivetran/Airbyte/dlt metadata tables) so reports carry an honest data
    vintage; without it the connector reports query time.
    """

    def __init__(
        self,
        url: str,
        *,
        source_system: str = "warehouse",
        sync_resolver: Callable[[], datetime] | None = None,
    ) -> None:
        _require_sqlalchemy()
        self._url = url
        self._source_system = source_system
        self._sync_resolver = sync_resolver
        self._engine: AsyncEngine | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.READ})

    def sync_metadata(self) -> SyncMetadata:
        synced_at = self._sync_resolver() if self._sync_resolver else datetime.now(timezone.utc)
        return SyncMetadata(source_system=self._source_system, last_synced_at=synced_at)

    def _get_engine(self) -> AsyncEngine:
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine

            self._engine = create_async_engine(self._url)
        return self._engine

    async def query(
        self, sql: str, params: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read query with bound parameters; rows come back as dicts."""
        from sqlalchemy import text

        async with self._get_engine().connect() as conn:
            result = await conn.execute(text(sql), dict(params or {}))
            return [dict(row._mapping) for row in result]

    async def health_check(self) -> bool:
        rows = await self.query("SELECT 1 AS ok")
        return bool(rows)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
