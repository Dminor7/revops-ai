"""RevOpsEngine: the dependency-injection container and run orchestrator.

The developer assembles an engine in their environment — connectors with
their credentials, an LLM configuration, policies — then registers agents.
All wiring is validated at registration; ``run()`` executes a typed task and
returns a typed report stamped with the run id, the vintage of the data it
used, and any writes the agent proposed.

Every run is ledgered (task, tool calls, intents, outcome) and can be
replayed from its recorded tool outputs without touching live systems.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from revops_ai.agents.base import BaseAgent, LLMAgent
from revops_ai.audit.ledger import AuditLedger, InMemoryAuditLedger, RunRecord, RunStatus
from revops_ai.audit.recording import RecordingToolProxy, RunRecorder
from revops_ai.audit.replay import ReplayToolProxy
from revops_ai.connectors.base import Capability, Connector
from revops_ai.core.context import RunContext, ToolNamespace
from revops_ai.core.registry import ConnectorRegistry
from revops_ai.core.settings import FreshnessPolicy, LLMConfig
from revops_ai.exceptions import (
    NoAgentForTaskError,
    RegistrationError,
    ReplayError,
    StaleDataError,
    WriteError,
)
from revops_ai.safety.policies import WritePolicy
from revops_ai.safety.write_intent import WriteIntent, WriteIntentStatus
from revops_ai.tasks.base import Report, Task
from revops_ai.tools.base import Tool

logger = logging.getLogger("revops_ai")

WriteMode = Literal["dry_run", "apply"]


@runtime_checkable
class WriteCapableConnector(Protocol):
    """A connector that can apply approved write intents."""

    async def apply_write(self, intent: WriteIntent) -> object: ...


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
        write_policy: WritePolicy | None = None,
        write_mode: WriteMode = "dry_run",
        ledger: AuditLedger | None = None,
    ) -> None:
        self.llm = llm
        self.freshness = freshness
        self.write_policy = write_policy
        self.write_mode: WriteMode = write_mode
        self.ledger: AuditLedger = ledger if ledger is not None else InMemoryAuditLedger()
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

    @property
    def task_types(self) -> dict[str, type[Task]]:
        """Registered task types by class name (the serving layer's registry)."""
        return {t.__name__: t for t in self._agents_by_task}

    # -- execution ----------------------------------------------------------

    async def run(self, task: Task) -> Report:
        """Execute a typed task with the agent registered for its type."""
        registered = self._agents_by_task.get(type(task))
        if registered is None:
            raise NoAgentForTaskError(type(task))

        vintage = self._collect_vintage(registered)
        self._enforce_freshness(registered.agent.name, vintage)

        run_id = uuid.uuid4().hex
        recorder = RunRecorder(self.ledger, run_id)
        proxies = {
            role: RecordingToolProxy(tool, role, recorder)
            for role, tool in registered.tools.items()
        }
        ctx = RunContext(tools=ToolNamespace(proxies), run_id=run_id, data_vintage=vintage)
        self.ledger.record_run_started(
            RunRecord(
                run_id=run_id,
                agent_name=registered.agent.name,
                task_type=type(task).__name__,
                task_json=task.model_dump_json(),
            )
        )
        try:
            report = await registered.agent.process_task(ctx, task)
            if not isinstance(report, registered.agent.report_type):
                raise TypeError(
                    f"Agent {registered.agent.name!r} returned {type(report).__name__}, "
                    f"declared report_type is {registered.agent.report_type.__name__}."
                )
            intents = await self._process_intents(ctx.proposed_writes, run_id)
        except Exception as exc:
            self.ledger.record_run_finished(run_id, RunStatus.FAILED, error=repr(exc))
            raise
        report = report.model_copy(
            update={"run_id": run_id, "data_vintage": vintage, "proposed_writes": intents}
        )
        self.ledger.record_run_finished(
            run_id, RunStatus.SUCCEEDED, report_json=report.model_dump_json()
        )
        return report

    async def replay(self, run_id: str) -> Report:
        """Re-execute a recorded run from its ledgered tool outputs.

        Live systems are never touched and writes are never applied; proposed
        intents are returned on the report for inspection only.
        """
        record = self.ledger.get_run(run_id)
        if record is None:
            raise ReplayError(f"No run {run_id!r} in the ledger.")
        registered = self._agents_by_name.get(record.agent_name)
        if registered is None:
            raise ReplayError(
                f"Run {run_id!r} was produced by agent {record.agent_name!r}, "
                "which is not registered on this engine."
            )
        task = registered.agent.task_type.model_validate_json(record.task_json)
        events = self.ledger.get_events(run_id)
        proxies = {role: ReplayToolProxy(role, events) for role in registered.tools}
        ctx = RunContext(
            tools=ToolNamespace(proxies),
            run_id=f"replay-{run_id}",
        )
        report = await registered.agent.process_task(ctx, task)
        return report.model_copy(
            update={"run_id": f"replay-{run_id}", "proposed_writes": ctx.proposed_writes}
        )

    # -- write-intent pipeline ------------------------------------------------

    async def _process_intents(self, proposed: list[WriteIntent], run_id: str) -> list[WriteIntent]:
        accepted = 0
        for intent in proposed:
            intent.run_id = run_id
            if self.write_policy is None:
                intent.status = WriteIntentStatus.REJECTED
                intent.status_reason = (
                    "No WritePolicy configured on the engine; all writes are rejected."
                )
            elif (reason := self.write_policy.violation(intent)) is not None:
                intent.status = WriteIntentStatus.REJECTED
                intent.status_reason = reason
            elif accepted >= self.write_policy.max_writes_per_run:
                intent.status = WriteIntentStatus.REJECTED
                intent.status_reason = (
                    f"Blast-radius limit: max_writes_per_run="
                    f"{self.write_policy.max_writes_per_run} already reached in this run."
                )
            else:
                accepted += 1
                if self.write_mode == "dry_run":
                    intent.status = WriteIntentStatus.PROPOSED
                    intent.status_reason = "Engine is in dry_run mode; nothing was written."
                elif self.write_policy.auto_approve is not None and self.write_policy.auto_approve(
                    intent
                ):
                    await self._apply_intent(intent)
                else:
                    intent.status = WriteIntentStatus.PENDING_APPROVAL
            self.ledger.record_intent(intent)
        return proposed

    async def _apply_intent(self, intent: WriteIntent) -> None:
        entry = self.connectors.get(intent.connector_role)
        if entry is None or Capability.WRITE not in entry.capabilities:
            intent.status = WriteIntentStatus.FAILED
            intent.status_reason = (
                f"Role {intent.connector_role!r} is not registered with write capability."
            )
            return
        if not isinstance(entry.connector, WriteCapableConnector):
            intent.status = WriteIntentStatus.FAILED
            intent.status_reason = (
                f"Connector for role {intent.connector_role!r} does not implement apply_write()."
            )
            return
        if self.ledger.was_applied(intent.idempotency_key):
            intent.status = WriteIntentStatus.APPLIED
            intent.status_reason = "Idempotent: an identical write was already applied."
            return
        try:
            await entry.connector.apply_write(intent)
        except Exception as exc:
            intent.status = WriteIntentStatus.FAILED
            intent.status_reason = repr(exc)
            logger.exception("Applying write intent %s failed", intent.intent_id)
            return
        self.ledger.mark_applied(intent.idempotency_key)
        intent.status = WriteIntentStatus.APPLIED
        intent.status_reason = None

    def pending_intents(self) -> list[WriteIntent]:
        """Intents awaiting human approval."""
        return self.ledger.list_intents(status=WriteIntentStatus.PENDING_APPROVAL)

    async def approve_intent(self, intent_id: str) -> WriteIntent:
        """Approve and apply a pending intent."""
        intent = self.ledger.get_intent(intent_id)
        if intent is None:
            raise WriteError(f"No intent {intent_id!r} in the ledger.")
        if intent.status is not WriteIntentStatus.PENDING_APPROVAL:
            raise WriteError(
                f"Intent {intent_id!r} is {intent.status.value}, not pending approval."
            )
        await self._apply_intent(intent)
        self.ledger.update_intent(intent)
        return intent

    def reject_intent(self, intent_id: str, reason: str) -> WriteIntent:
        """Reject a pending intent with a recorded reason."""
        intent = self.ledger.get_intent(intent_id)
        if intent is None:
            raise WriteError(f"No intent {intent_id!r} in the ledger.")
        if intent.status is not WriteIntentStatus.PENDING_APPROVAL:
            raise WriteError(
                f"Intent {intent_id!r} is {intent.status.value}, not pending approval."
            )
        intent.status = WriteIntentStatus.REJECTED
        intent.status_reason = reason
        self.ledger.update_intent(intent)
        return intent

    # -- freshness and health -------------------------------------------------

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
