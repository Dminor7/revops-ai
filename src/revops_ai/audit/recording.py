"""Per-run tool-call recording.

Agents reach data only through ``ctx.tools``; the engine wraps every bound
tool in a recording proxy per run, so the audit surface is exactly the
data-access surface — there is no unrecorded path to a connector.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from revops_ai.audit.ledger import AuditLedger, ToolCallEvent

if TYPE_CHECKING:
    from revops_ai.tools.base import Tool


def _to_json(value: object) -> str:
    return json.dumps(value, default=str)


class RunRecorder:
    """Sequences tool-call events for one run into the ledger."""

    def __init__(self, ledger: AuditLedger, run_id: str) -> None:
        self._ledger = ledger
        self._run_id = run_id
        self._seq = 0

    def record(
        self,
        role: str,
        method: str,
        args_json: str,
        *,
        result_json: str | None = None,
        error: str | None = None,
    ) -> None:
        self._seq += 1
        self._ledger.record_event(
            ToolCallEvent(
                run_id=self._run_id,
                seq=self._seq,
                role=role,
                method=method,
                args_json=args_json,
                result_json=result_json,
                error=error,
            )
        )


class RecordingToolProxy:
    """Wraps a bound tool; every public async method call is ledgered.

    Tool outputs must be JSON-serializable (rows, dicts, lists, scalars) —
    that is what makes recorded replay possible.
    """

    def __init__(self, tool: Tool, role: str, recorder: RunRecorder) -> None:
        self._tool = tool
        self._role = role
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tool, name)
        if name.startswith("_") or not inspect.iscoroutinefunction(attr):
            return attr

        async def recorded(*args: object, **kwargs: object) -> object:
            args_json = _to_json({"args": list(args), "kwargs": kwargs})
            try:
                result = await attr(*args, **kwargs)
            except Exception as exc:
                self._recorder.record(self._role, name, args_json, error=repr(exc))
                raise
            self._recorder.record(self._role, name, args_json, result_json=_to_json(result))
            return result

        return recorded
