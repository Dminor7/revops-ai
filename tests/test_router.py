"""Natural-language routing: text in, typed task out, same audited run path."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from revops_ai import LLMConfig, RevOpsEngine, RevOpsError
from tests.conftest import CommissionAgent, CommissionReport, FakeWarehouseConnector


def _routing_model() -> FunctionModel:
    def choose(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = next(
            (t for t in info.output_tools if "CommissionTask" in t.name),
            info.output_tools[0],
        )
        return ModelResponse(parts=[ToolCallPart(tool.name, {"deal_id": "D-1"})])

    return FunctionModel(choose)


async def test_run_analysis_routes_text_to_typed_task(
    warehouse: FakeWarehouseConnector,
) -> None:
    engine = RevOpsEngine(llm=LLMConfig(model=_routing_model()))
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())

    report = await engine.run_analysis("Evaluate the commission for deal D-1")

    # NL is a router over the typed path: full report, ledgered run and all.
    assert isinstance(report, CommissionReport)
    assert report.status == "approved"
    assert report.run_id is not None
    assert engine.ledger.get_run(report.run_id) is not None


async def test_run_analysis_requires_llm_config(warehouse: FakeWarehouseConnector) -> None:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())
    with pytest.raises(RevOpsError, match="LLMConfig"):
        await engine.run_analysis("anything")


async def test_run_analysis_requires_registered_agents() -> None:
    engine = RevOpsEngine(llm=LLMConfig(model=_routing_model()))
    with pytest.raises(RevOpsError, match="agent"):
        await engine.run_analysis("anything")
