"""LLMAgent behavior using pydantic-ai's offline TestModel."""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from revops_ai import LLMAgent, LLMConfig, RegistrationError, Report, RevOpsEngine, Task
from tests.conftest import FakeWarehouseConnector, SQLQueryTool


class DealSummaryTask(Task):
    deal_id: str


class DealSummaryReport(Report):
    deal_id: str
    summary: str


class DealSummaryAgent(LLMAgent[DealSummaryTask, DealSummaryReport]):
    name = "deal_summarizer"
    description = "Summarizes a deal for handoff."
    instructions = "Summarize the deal described in the task payload."
    requires = {"warehouse": SQLQueryTool}
    task_type = DealSummaryTask
    report_type = DealSummaryReport


async def test_llm_agent_runs_with_test_model(warehouse: FakeWarehouseConnector) -> None:
    engine = RevOpsEngine(llm=LLMConfig(model=TestModel()))
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(DealSummaryAgent())

    report = await engine.run(DealSummaryTask(deal_id="D-1"))

    # TestModel fabricates schema-valid output; the contract under test is
    # that the engine returns the declared report type with vintage attached.
    assert isinstance(report, DealSummaryReport)
    assert "warehouse" in report.data_vintage


async def test_llm_agent_without_llm_config_fails_at_registration(
    warehouse: FakeWarehouseConnector,
) -> None:
    engine = RevOpsEngine()  # no LLMConfig
    engine.register_connector("warehouse", warehouse)
    with pytest.raises(RegistrationError) as exc:
        engine.register_agent(DealSummaryAgent())
    assert "LLMConfig" in str(exc.value)
