"""Shipped agents, driven offline with pydantic-ai FunctionModel."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from revops_ai import Capability, LLMConfig, RevOpsEngine, SyncMetadata
from revops_ai.agents.churn_predictor import (
    ChurnPredictorAgent,
    ChurnRiskReport,
    ChurnScanTask,
)
from revops_ai.agents.pipeline_velocity import (
    DealRiskReport,
    PipelineEvaluationTask,
    PipelineVelocityAgent,
)


class FakeCRM:
    capabilities = frozenset({Capability.READ})

    def sync_metadata(self) -> SyncMetadata:
        return SyncMetadata(source_system="fake_crm")

    async def health_check(self) -> bool:
        return True

    async def search_deals(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": "D-1", "dealstage": "commit", "days_in_stage": 45}]

    async def get_deal(self, deal_id: str, properties: Any = None) -> dict[str, Any]:
        return {"id": deal_id, "dealstage": "commit", "days_in_stage": 45}


class FakeBilling:
    capabilities = frozenset({Capability.READ})

    def sync_metadata(self) -> SyncMetadata:
        return SyncMetadata(source_system="fake_billing")

    async def health_check(self) -> bool:
        return True

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        return {"id": customer_id, "delinquent": True}

    async def list_subscriptions(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"id": "sub_1", "status": "past_due", "cancel_at_period_end": True}]


def _finding_args(entity_type: str, entity_id: str, verdict: str) -> dict[str, Any]:
    return {
        "subject": {
            "source_system": "fake",
            "entity_type": entity_type,
            "entity_id": entity_id,
        },
        "verdict": verdict,
        "confidence": 0.8,
        "evidence": [
            {
                "kind": "record",
                "summary": "45 days in stage vs 12-day median",
                "source": {"kind": "record", "reference": entity_id},
            }
        ],
    }


def _scripted_model(
    tool_name: str, tool_args: dict[str, Any], report: dict[str, Any]
) -> FunctionModel:
    """First call a tool, then produce the final structured report."""
    state = {"called": False}

    def step(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not state["called"]:
            state["called"] = True
            return ModelResponse(parts=[ToolCallPart(tool_name, tool_args)])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    return FunctionModel(step)


async def test_pipeline_velocity_agent_end_to_end() -> None:
    model = _scripted_model(
        "search_deals",
        {"stage": "commit"},
        {
            "summary": "1 of 1 commit deals is stalled.",
            "findings": [_finding_args("deal", "D-1", "at_risk")],
        },
    )
    engine = RevOpsEngine(llm=LLMConfig(model=model))
    engine.register_connector("crm", FakeCRM())
    engine.register_agent(PipelineVelocityAgent())

    report = await engine.run(PipelineEvaluationTask(stage="commit"))

    assert isinstance(report, DealRiskReport)
    assert report.findings[0].verdict == "at_risk"
    assert report.findings[0].evidence[0].summary.startswith("45 days")
    assert "crm" in report.data_vintage

    # The LLM's tool call went through the bound tool and is in the ledger.
    assert report.run_id is not None
    events = engine.ledger.get_events(report.run_id)
    assert [e.method for e in events] == ["search_deals"]
    assert "at_risk (80%)" in report.to_markdown()


async def test_churn_predictor_agent_end_to_end() -> None:
    model = _scripted_model(
        "list_subscriptions",
        {"customer_id": "cus_1"},
        {
            "summary": "cus_1 shows churn signals.",
            "findings": [_finding_args("customer", "cus_1", "churn_risk")],
        },
    )
    engine = RevOpsEngine(llm=LLMConfig(model=model))
    engine.register_connector("billing", FakeBilling())
    engine.register_agent(ChurnPredictorAgent(sensitivity="high"))

    report = await engine.run(ChurnScanTask(customer_ids=("cus_1",)))

    assert isinstance(report, ChurnRiskReport)
    assert report.findings[0].verdict == "churn_risk"
    assert report.run_id is not None
    events = engine.ledger.get_events(report.run_id)
    assert [e.method for e in events] == ["list_subscriptions"]


def test_churn_sensitivity_shapes_instructions() -> None:
    high = ChurnPredictorAgent(sensitivity="high")
    low = ChurnPredictorAgent(sensitivity="low")
    assert "false positives" in high.get_instructions()
    assert "multi-signal" in low.get_instructions()
    assert high.get_instructions() != low.get_instructions()
