"""Shared test fixtures: a fake connector and a deterministic commission agent.

These double as the reference implementation of the extension contract — if
this file gets awkward to write, the SDK's developer experience has regressed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from revops_ai import (
    BaseAgent,
    Capability,
    Report,
    RunContext,
    SyncMetadata,
    Task,
    Tool,
)


class FakeWarehouseConnector:
    """In-memory stand-in for a SQL warehouse, satisfying the Connector protocol."""

    def __init__(
        self,
        rows: dict[str, dict[str, object]] | None = None,
        *,
        synced_at: datetime | None = None,
        capabilities: frozenset[Capability] = frozenset({Capability.READ}),
    ) -> None:
        self.rows = rows or {}
        self._synced_at = synced_at or datetime.now(timezone.utc)
        self._capabilities = capabilities
        self.queries: list[str] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def sync_metadata(self) -> SyncMetadata:
        return SyncMetadata(source_system="fake_warehouse", last_synced_at=self._synced_at)

    async def health_check(self) -> bool:
        return True


class SQLQueryTool(Tool):
    required_capabilities = frozenset({Capability.READ})

    async def query_one(self, sql: str, **params: object) -> dict[str, object]:
        connector = self.connector
        assert isinstance(connector, FakeWarehouseConnector)
        connector.queries.append(sql)
        return connector.rows[str(params["deal_id"])]


class CommissionTask(Task):
    deal_id: str


class CommissionReport(Report):
    deal_id: str
    status: str
    payout_tier: str


class CommissionAgent(BaseAgent[CommissionTask, CommissionReport]):
    """Deterministic agent mirroring the README example: margin -> payout tier."""

    name = "commission_calculator"
    description = "Calculates commissions from territory rules."
    requires = {"warehouse": SQLQueryTool}
    task_type = CommissionTask
    report_type = CommissionReport

    async def process_task(self, ctx: RunContext, task: CommissionTask) -> CommissionReport:
        row = await ctx.tools.warehouse.query_one(
            "SELECT margin FROM deals WHERE id = :deal_id", deal_id=task.deal_id
        )
        margin = float(row["margin"])  # type: ignore[arg-type]
        if margin > 40:
            return CommissionReport(
                deal_id=task.deal_id, status="approved", payout_tier="Alpha - 15%"
            )
        return CommissionReport(deal_id=task.deal_id, status="review", payout_tier="Standard - 8%")


@pytest.fixture
def warehouse() -> FakeWarehouseConnector:
    return FakeWarehouseConnector(
        rows={"D-1": {"margin": 52.0}, "D-2": {"margin": 12.0}},
    )


@pytest.fixture
def stale_warehouse() -> FakeWarehouseConnector:
    return FakeWarehouseConnector(
        rows={"D-1": {"margin": 52.0}},
        synced_at=datetime.now(timezone.utc) - timedelta(hours=48),
    )
