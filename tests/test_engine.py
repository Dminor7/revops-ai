"""Engine assembly and run-path behavior."""

from __future__ import annotations

from datetime import timedelta

import pytest

from revops_ai import (
    Capability,
    CapabilityError,
    FreshnessPolicy,
    NoAgentForTaskError,
    RegistrationError,
    RevOpsEngine,
    StaleDataError,
    Task,
    Tool,
    UnknownRoleError,
)
from tests.conftest import (
    CommissionAgent,
    CommissionReport,
    CommissionTask,
    FakeWarehouseConnector,
)


async def test_run_returns_typed_report_with_data_vintage(
    warehouse: FakeWarehouseConnector,
) -> None:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())

    report = await engine.run(CommissionTask(deal_id="D-1"))

    assert isinstance(report, CommissionReport)
    assert report.status == "approved"
    assert report.payout_tier == "Alpha - 15%"
    assert "warehouse" in report.data_vintage
    assert warehouse.queries  # the agent reached data through the bound tool


async def test_low_margin_routes_to_review(warehouse: FakeWarehouseConnector) -> None:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())

    report = await engine.run(CommissionTask(deal_id="D-2"))
    assert isinstance(report, CommissionReport)
    assert report.status == "review"


async def test_missing_connector_fails_at_registration() -> None:
    engine = RevOpsEngine()
    with pytest.raises(UnknownRoleError) as exc:
        engine.register_agent(CommissionAgent())
    assert "warehouse" in str(exc.value)


async def test_capability_narrowing_blocks_write_tools(
    warehouse: FakeWarehouseConnector,
) -> None:
    class CRMWriteTool(Tool):
        required_capabilities = frozenset({Capability.WRITE})

    class WriteTask(Task):
        pass

    class WritingAgent(CommissionAgent):
        name = "writer"
        requires = {"warehouse": CRMWriteTool}
        task_type = WriteTask

    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)  # read-only connector
    with pytest.raises(CapabilityError):
        engine.register_agent(WritingAgent())


async def test_unknown_task_type_raises(warehouse: FakeWarehouseConnector) -> None:
    class OtherTask(Task):
        pass

    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())
    with pytest.raises(NoAgentForTaskError):
        await engine.run(OtherTask())


async def test_duplicate_agent_name_rejected(warehouse: FakeWarehouseConnector) -> None:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())
    with pytest.raises(RegistrationError):
        engine.register_agent(CommissionAgent())


async def test_duplicate_connector_role_rejected(warehouse: FakeWarehouseConnector) -> None:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    with pytest.raises(RegistrationError):
        engine.register_connector("warehouse", FakeWarehouseConnector())


async def test_freshness_policy_fail_aborts_run(
    stale_warehouse: FakeWarehouseConnector,
) -> None:
    engine = RevOpsEngine(
        freshness=FreshnessPolicy(max_staleness=timedelta(hours=6), on_violation="fail")
    )
    engine.register_connector("warehouse", stale_warehouse)
    engine.register_agent(CommissionAgent())
    with pytest.raises(StaleDataError):
        await engine.run(CommissionTask(deal_id="D-1"))


async def test_freshness_policy_warn_annotates_but_runs(
    stale_warehouse: FakeWarehouseConnector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = RevOpsEngine(
        freshness=FreshnessPolicy(max_staleness=timedelta(hours=6), on_violation="warn")
    )
    engine.register_connector("warehouse", stale_warehouse)
    engine.register_agent(CommissionAgent())

    with caplog.at_level("WARNING", logger="revops_ai"):
        report = await engine.run(CommissionTask(deal_id="D-1"))

    assert isinstance(report, CommissionReport)
    assert any("freshness" in r.message for r in caplog.records)


async def test_health_check(warehouse: FakeWarehouseConnector) -> None:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    assert await engine.health_check() == {"warehouse": True}


async def test_same_agent_class_two_engines_different_connectors() -> None:
    """Multi-tenancy: one agent class, two engines, isolated data layers."""
    wh_a = FakeWarehouseConnector(rows={"D-1": {"margin": 90.0}})
    wh_b = FakeWarehouseConnector(rows={"D-1": {"margin": 5.0}})

    engine_a, engine_b = RevOpsEngine(), RevOpsEngine()
    engine_a.register_connector("warehouse", wh_a)
    engine_b.register_connector("warehouse", wh_b)
    engine_a.register_agent(CommissionAgent())
    engine_b.register_agent(CommissionAgent())

    report_a = await engine_a.run(CommissionTask(deal_id="D-1"))
    report_b = await engine_b.run(CommissionTask(deal_id="D-1"))
    assert isinstance(report_a, CommissionReport)
    assert isinstance(report_b, CommissionReport)
    assert report_a.status == "approved"
    assert report_b.status == "review"
