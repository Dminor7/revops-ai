"""ChurnPredictorAgent: billing-signal churn risk with evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai import RunContext as ModelRunContext

from revops_ai.agents.base import LLMAgent
from revops_ai.audit.evidence import Finding
from revops_ai.core.context import RunContext
from revops_ai.tasks.base import Report, Task
from revops_ai.tools.billing import BillingReadTool

Sensitivity = Literal["low", "medium", "high"]

_SENSITIVITY_GUIDANCE: dict[Sensitivity, str] = {
    "low": "Only flag customers with strong, multi-signal churn indicators.",
    "medium": "Flag customers with clear churn indicators; skip weak single signals.",
    "high": (
        "Flag any customer with a plausible churn indicator, including weak "
        "single signals; prefer false positives over misses."
    ),
}


class ChurnScanTask(Task):
    """Assess churn risk for a set of billing customers."""

    customer_ids: tuple[str, ...]


class ChurnRiskReport(Report):
    summary: str
    findings: list[Finding] = Field(default_factory=list)


class ChurnPredictorAgent(LLMAgent[ChurnScanTask, ChurnRiskReport]):
    """Scores churn risk from subscription state, with cited evidence."""

    name = "churn_predictor"
    description = "Assesses churn risk for billing customers."
    instructions = (
        "You assess churn risk for a SaaS revenue team. For each customer in "
        "the task, inspect their billing record and subscriptions (status, "
        "cancellations, downgrades, payment issues). Produce a finding per "
        "at-risk customer with a verdict, a confidence between 0 and 1, and "
        "evidence citing the specific billing facts. Finish with a short "
        "summary."
    )
    requires = {"billing": BillingReadTool}
    task_type = ChurnScanTask
    report_type = ChurnRiskReport

    def __init__(self, sensitivity: Sensitivity = "medium") -> None:
        super().__init__()
        self.sensitivity: Sensitivity = sensitivity

    def get_instructions(self) -> str:
        return f"{self.instructions} {_SENSITIVITY_GUIDANCE[self.sensitivity]}"

    def configure(self, agent: PydanticAIAgent[RunContext, ChurnRiskReport]) -> None:
        @agent.tool
        async def get_customer(rc: ModelRunContext[RunContext], customer_id: str) -> dict[str, Any]:
            """Fetch one billing customer by id."""
            result: dict[str, Any] = await rc.deps.tools.billing.get_customer(customer_id)
            return result

        @agent.tool
        async def list_subscriptions(
            rc: ModelRunContext[RunContext],
            customer_id: str,
            status: str | None = None,
        ) -> list[dict[str, Any]]:
            """List a customer's subscriptions, optionally by status."""
            result: list[dict[str, Any]] = await rc.deps.tools.billing.list_subscriptions(
                customer=customer_id, status=status
            )
            return result
