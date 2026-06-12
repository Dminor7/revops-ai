"""PipelineVelocityAgent: evidence-backed risk evaluation of a pipeline stage."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai import RunContext as ModelRunContext

from revops_ai.agents.base import LLMAgent
from revops_ai.audit.evidence import Finding
from revops_ai.core.context import RunContext
from revops_ai.tasks.base import Report, Task
from revops_ai.tools.crm import CRMReadTool


class PipelineEvaluationTask(Task):
    """Evaluate open deals in one pipeline stage."""

    stage: str
    quarter: str | None = None


class DealRiskReport(Report):
    summary: str
    findings: list[Finding] = Field(default_factory=list)


class PipelineVelocityAgent(LLMAgent[PipelineEvaluationTask, DealRiskReport]):
    """Flags deals whose movement lags the stage norm, with evidence.

    Every finding must cite the CRM facts behind the verdict — the report
    schema rejects bare scores.
    """

    name = "pipeline_velocity"
    description = "Evaluates open deals in a pipeline stage for velocity risk."
    instructions = (
        "You evaluate deal velocity for a revenue team. Use the CRM tools to "
        "list the deals in the requested stage and inspect any deal that looks "
        "stalled (old close dates, no recent activity, long time in stage). "
        "For each at-risk deal produce a finding with a verdict, a confidence "
        "between 0 and 1, and evidence entries citing the specific CRM fields "
        "you relied on. Deals that look healthy do not need findings. Finish "
        "with a short summary a VP of Sales can read in ten seconds."
    )
    requires = {"crm": CRMReadTool}
    task_type = PipelineEvaluationTask
    report_type = DealRiskReport

    def configure(self, agent: PydanticAIAgent[RunContext, DealRiskReport]) -> None:
        @agent.tool
        async def search_deals(
            rc: ModelRunContext[RunContext], stage: str, max_results: int = 100
        ) -> list[dict[str, Any]]:
            """List CRM deals in a pipeline stage."""
            result: list[dict[str, Any]] = await rc.deps.tools.crm.search_deals(
                stage=stage, max_results=max_results
            )
            return result

        @agent.tool
        async def get_deal(rc: ModelRunContext[RunContext], deal_id: str) -> dict[str, Any]:
            """Fetch one deal's properties by id."""
            result: dict[str, Any] = await rc.deps.tools.crm.get_deal(deal_id)
            return result
