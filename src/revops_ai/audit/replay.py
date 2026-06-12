"""Recorded replay: re-execute a run from its ledgered tool outputs.

Replay answers "reproduce last Tuesday's decision" without touching live
systems: tools return exactly what they returned during the original run, in
the original order. Deterministic agents reproduce their report exactly;
LLM-backed agents replay their *tool* I/O but the model itself runs live (LLM
request/response recording arrives with the OTel integration — see PLAN.md).

Writes are never applied during replay.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from revops_ai.audit.ledger import ToolCallEvent
from revops_ai.exceptions import ReplayError


class ReplayToolProxy:
    """Serves recorded outputs for one role's tool, in recorded order."""

    def __init__(self, role: str, events: list[ToolCallEvent]) -> None:
        self._role = role
        self._queues: dict[str, deque[ToolCallEvent]] = {}
        for event in sorted(events, key=lambda e: e.seq):
            if event.role == role:
                self._queues.setdefault(event.method, deque()).append(event)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        async def replayed(*args: object, **kwargs: object) -> object:
            queue = self._queues.get(name)
            if not queue:
                raise ReplayError(
                    f"Replay exhausted: no recorded call left for {self._role}.{name}(). "
                    "The agent made more (or different) tool calls than the original run."
                )
            event = queue.popleft()
            if event.error is not None:
                raise ReplayError(
                    f"Recorded call {self._role}.{name}() failed in the original run: {event.error}"
                )
            assert event.result_json is not None
            return json.loads(event.result_json)

        return replayed
