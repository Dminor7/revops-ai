"""BaseAgent and LLMAgent.

``BaseAgent`` is the extension point of the SDK: subclass it, declare the task
type you handle, the report type you produce, and the connector roles you
require, then implement ``process_task``. Deterministic agents (commission
math, compliance checks) need nothing else.

``LLMAgent`` layers a pydantic-ai agent loop on top for agents that reason
with a model. pydantic-ai is deliberately confined to this module — nothing in
the public API of the SDK imports it, so the runtime can be swapped without a
breaking release.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from revops_ai.tasks.base import Report, Task

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAIAgent
    from pydantic_ai.models import Model

    from revops_ai.core.context import RunContext
    from revops_ai.core.settings import LLMConfig
    from revops_ai.tools.base import Tool

TaskT = TypeVar("TaskT", bound=Task)
ReportT = TypeVar("ReportT", bound=Report)


class BaseAgent(ABC, Generic[TaskT, ReportT]):
    """Base class for all agents.

    Class attributes are *declarations*; nothing is bound until the agent is
    registered with an engine, which resolves ``requires`` against the
    connector registry and fails fast if anything is missing.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    #: Connector role -> Tool class the engine must bind for this agent.
    requires: ClassVar[dict[str, type[Tool]]] = {}
    #: The Task subclass this agent handles; the engine routes on this.
    task_type: ClassVar[type[Task]]
    #: The Report subclass this agent produces.
    report_type: ClassVar[type[Report]]

    def __init__(self) -> None:
        missing = [a for a in ("name", "task_type", "report_type") if not hasattr(type(self), a)]
        if missing:
            raise TypeError(f"{type(self).__name__} must declare {', '.join(missing)}")

    @abstractmethod
    async def process_task(self, ctx: RunContext, task: TaskT) -> ReportT:
        """Perform the task. Reach data only through ``ctx.tools``."""


class LLMAgent(BaseAgent[TaskT, ReportT]):
    """An agent whose ``process_task`` is an LLM loop with typed output.

    Subclasses declare ``instructions`` and may register pydantic-ai tools by
    overriding ``configure``. The model is supplied by the engine's
    :class:`~revops_ai.core.settings.LLMConfig`, never hardcoded in the agent.
    """

    instructions: ClassVar[str] = ""

    def __init__(self) -> None:
        super().__init__()
        self._pai_agent: PydanticAIAgent[RunContext, ReportT] | None = None
        self._llm: LLMConfig | None = None

    def bind_llm(self, config: LLMConfig) -> None:
        """Called by the engine at registration with its LLM configuration."""
        self._llm = config
        self._pai_agent = None

    def _resolve_model(self) -> Model | str:
        from revops_ai._llm import resolve_model

        if self._llm is None:
            raise RuntimeError(
                f"LLMAgent {self.name!r} has no model bound; register it with an "
                "engine configured with an LLMConfig."
            )
        return resolve_model(self._llm)

    def _build(self) -> PydanticAIAgent[RunContext, ReportT]:
        from pydantic_ai import Agent as PydanticAIAgent

        from revops_ai.core.context import RunContext

        agent = PydanticAIAgent(
            self._resolve_model(),
            deps_type=RunContext,
            output_type=cast("type[ReportT]", self.report_type),
            instructions=self.instructions or self.description,
        )
        self.configure(agent)
        return agent

    def configure(self, agent: PydanticAIAgent[RunContext, ReportT]) -> None:
        """Hook for subclasses to register pydantic-ai tools on the agent."""

    async def process_task(self, ctx: RunContext, task: TaskT) -> ReportT:
        if self._pai_agent is None:
            self._pai_agent = self._build()
        result = await self._pai_agent.run(task.model_dump_json(), deps=ctx)
        return result.output
