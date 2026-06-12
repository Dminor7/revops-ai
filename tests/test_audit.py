"""Run ledger and recorded replay."""

from __future__ import annotations

import pytest

from revops_ai import ReplayError, RevOpsEngine
from revops_ai.audit import InMemoryAuditLedger, RunStatus
from tests.conftest import CommissionAgent, CommissionReport, CommissionTask, FakeWarehouseConnector


def _engine(warehouse: FakeWarehouseConnector) -> RevOpsEngine:
    engine = RevOpsEngine(ledger=InMemoryAuditLedger())
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())
    return engine


async def test_run_is_recorded_with_tool_calls(warehouse: FakeWarehouseConnector) -> None:
    engine = _engine(warehouse)
    report = await engine.run(CommissionTask(deal_id="D-1"))
    assert report.run_id is not None

    record = engine.ledger.get_run(report.run_id)
    assert record is not None
    assert record.agent_name == "commission_calculator"
    assert record.task_type == "CommissionTask"
    assert record.status is RunStatus.SUCCEEDED
    assert record.report_json is not None

    events = engine.ledger.get_events(report.run_id)
    assert len(events) == 1
    assert events[0].role == "warehouse"
    assert events[0].method == "query_one"
    assert "D-1" in events[0].args_json
    assert events[0].error is None


async def test_failed_run_is_recorded(warehouse: FakeWarehouseConnector) -> None:
    engine = _engine(warehouse)
    with pytest.raises(KeyError):
        await engine.run(CommissionTask(deal_id="NOPE"))

    runs = engine.ledger.list_runs()
    assert len(runs) == 1
    assert runs[0].status is RunStatus.FAILED
    assert runs[0].error is not None
    # The failing tool call is in the ledger too.
    events = engine.ledger.get_events(runs[0].run_id)
    assert events[0].error is not None


async def test_recorded_replay_reproduces_report_without_touching_data(
    warehouse: FakeWarehouseConnector,
) -> None:
    engine = _engine(warehouse)
    report = await engine.run(CommissionTask(deal_id="D-1"))
    assert report.run_id is not None
    queries_after_run = list(warehouse.queries)

    # Change the underlying data: replay must still reproduce the original decision.
    warehouse.rows["D-1"]["margin"] = 1.0

    replayed = await engine.replay(report.run_id)
    assert isinstance(replayed, CommissionReport)
    assert replayed.status == "approved"  # decision from recorded data, not live
    assert warehouse.queries == queries_after_run  # no live data access


async def test_replay_unknown_run_raises(warehouse: FakeWarehouseConnector) -> None:
    engine = _engine(warehouse)
    with pytest.raises(ReplayError):
        await engine.replay("missing")


async def test_replay_detects_exhausted_recording(warehouse: FakeWarehouseConnector) -> None:
    """If the agent makes more tool calls on replay than were recorded, fail loudly."""
    engine = _engine(warehouse)
    report = await engine.run(CommissionTask(deal_id="D-1"))
    assert report.run_id is not None

    # Sabotage the recording: drop all events.
    engine.ledger._events[report.run_id].clear()  # type: ignore[attr-defined]
    with pytest.raises(ReplayError):
        await engine.replay(report.run_id)
