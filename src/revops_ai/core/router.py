"""Natural-language router: parse text into a registered, typed Task.

NL is a UX layer over the typed API — the router's only job is to choose a
Task type and fill its fields; execution then follows the exact same audited
``engine.run()`` path as a structured call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from revops_ai._llm import resolve_model
from revops_ai.core.settings import LLMConfig
from revops_ai.tasks.base import Task

if TYPE_CHECKING:
    from pydantic_ai import Agent

ROUTER_INSTRUCTIONS = (
    "You route a revenue-operations request to exactly one task. Read the "
    "request and produce the single task type that fits it best, filling every "
    "field from the request. Do not invent identifiers that are not in the "
    "request."
)


class TaskRouter:
    """Routes free text to one of the registered task types."""

    def __init__(self, llm: LLMConfig, task_types: list[type[Task]]) -> None:
        self._llm = llm
        self._task_types = task_types
        self._agent: Agent[None, Task] | None = None

    def _build(self) -> Agent[None, Task]:
        from pydantic_ai import Agent

        return Agent(
            resolve_model(self._llm),
            output_type=self._task_types,
            instructions=ROUTER_INSTRUCTIONS,
        )

    async def route(self, text: str) -> Task:
        if self._agent is None:
            self._agent = self._build()
        result = await self._agent.run(text)
        return result.output
