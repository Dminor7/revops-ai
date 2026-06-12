"""RevOpsEngine: the dependency-injection container and run orchestrator.

The developer assembles an engine in their environment — connectors with
their credentials, an LLM configuration, policies — then registers agents.
All wiring is validated at registration; ``run()`` executes a typed task and
returns a typed report stamped with the vintage of the data it used.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from revops_ai.agents.base import BaseAgent, LLMAgent
from revops_ai.connectors.base import Capability, Connector
from revops_ai.core.context import RunContext, ToolNamespace
from revops_ai.core.registry import ConnectorRegistry
from revops_ai.core.settings import FreshnessPolicy, LLMConfig
from revops_ai.exceptions import NoAgentForTaskError, RegistrationError, StaleDataError
from revops_ai.tasks.base import Report, Task
from revops_ai.tools.base import Tool

logger = logging.getLogger("revops_ai")


class _RegisteredAgent:
    """An agent plus the tools the engine bound for it at registration."""

    def __init__(self, agent: BaseAgent[Task, Report], tools: dict[str, Tool]) -> None:
        self.agent = agent
        self.tools = tools


class RevOpsEngine:
    """Orchestrates connectors, agents, and policies for one environment."""

    def __init__(
        self,
        *,
        llm: LLMConfig | None = None,
        freshness: FreshnessPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.freshness = freshness
        self.connectors = ConnectorRegistry()
        self._agents_by_task: dict[type[Task], _RegisteredAgent] = {}
        self._agents_by_name: dict[str, _RegisteredAgent] = {}

    # -- assembly -----------------------------------------------------------

    def register_connector(
        self,
        role: str,
        connector: Connector,
        *,
        capabilities: set[Capability] | None = None,
    ) -> None:
        """Register a connector under a role (``"crm"``, ``"warehouse"``, ...).

        Pass ``capabilities`` to narrow what this engine may do with it, e.g.
        register a writable CRM as read-only.
        """
        self.connectors.register(role, connector, capabilities=capabilities)

    def register_agent(self, agent: BaseAgent[Task, Report]) -> None:
        """Validate and wire an agent.

        Resolves the agent's ``requires`` declaration against the connector
        registry and binds the engine's LLM configuration. Any missing role,
        capability, or model configuration raises here — never mid-run.
        """
        if agent.name in self._agents_by_name:
            raise RegistrationError(f"An agent named {agent.name!r} is already registered.")
        if agent.task_type in self._agents_by_task:
            other = self._agents_by_task[agent.task_type].agent.name
            raise RegistrationError(
                f"Task type {agent.task_type.__name__!r} is already handled by "
                f"agent {other!r}; one agent per task type."
            )
        tools = {
            role: self.connectors.resolve_tool(agent.name, role, tool_cls)
            for role, tool_cls in agent.requires.items()
        }
        if isinstance(agent, LLMAgent):
            if self.llm is None:
                raise RegistrationError(
                    f"Agent {agent.name!r} is LLM-backed but the engine has no "
                    "LLMConfig; pass RevOpsEngine(llm=LLMConfig(...))."
                )
            agent.bind_llm(self.llm)
        registered = _RegisteredAgent(agent, tools)
        self._agents_by_name[agent.name] = registered
        self._agents_by_task[agent.task_type] = registered

    @property
    def agent_names(self) -> list[str]:
        return sorted(self._agents_by_name)

    # -- execution ----------------------------------------------------------

    async def run(self, task: Task) -> Report:
        """Execute a typed task with the agent registered for its type."""
        registered = self._agents_by_task.get(type(task))
        if registered is None:
            raise NoAgentForTaskError(type(task))

        vintage = self._collect_vintage(registered)
        self._enforce_freshness(registered.agent.name, vintage)

        ctx = RunContext(
            tools=ToolNamespace(registered.tools),
            run_id=uuid.uuid4().hex,
            data_vintage=vintage,
        )
        report = await registered.agent.process_task(ctx, task)
        if not isinstance(report, registered.agent.report_type):
            raise TypeError(
                f"Agent {registered.agent.name!r} returned {type(report).__name__}, "
                f"declared report_type is {registered.agent.report_type.__name__}."
            )
        return report.model_copy(update={"data_vintage": vintage})

    def _collect_vintage(self, registered: _RegisteredAgent) -> dict[str, datetime]:
        return {
            role: tool.connector.sync_metadata().last_synced_at
            for role, tool in registered.tools.items()
        }

    def _enforce_freshness(self, agent_name: str, vintage: dict[str, datetime]) -> None:
        if self.freshness is None:
            return
        max_age = self.freshness.max_staleness.total_seconds()
        for role, synced_at in vintage.items():
            age = (datetime.now(synced_at.tzinfo) - synced_at).total_seconds()
            if age <= max_age:
                continue
            message = (
                f"Data for role {role!r} is {age / 3600:.1f}h old, exceeding the "
                f"{max_age / 3600:.1f}h freshness policy (agent {agent_name!r})."
            )
            if self.freshness.on_violation == "fail":
                raise StaleDataError(message)
            logger.warning(message)

    async def health_check(self) -> dict[str, bool]:
        """Check every registered connector; returns role -> healthy."""
        results: dict[str, bool] = {}
        for role in self.connectors.roles:
            entry = self.connectors.get(role)
            assert entry is not None
            try:
                results[role] = await entry.connector.health_check()
            except Exception:
                logger.exception("Health check failed for connector role %r", role)
                results[role] = False
        return results

    def get_report_type(self, task: Task) -> type[Report]:
        """The report type ``run(task)`` will return, for callers that need it."""
        registered = self._agents_by_task.get(type(task))
        if registered is None:
            raise NoAgentForTaskError(type(task))
        return registered.agent.report_type
