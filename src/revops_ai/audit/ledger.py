"""The run ledger: an append-style record of every run, tool call, and intent.

The ledger answers "why did the agent decide X last Tuesday" and powers
recorded replay. ``InMemoryAuditLedger`` is the default; a SQLAlchemy-backed
ledger in the customer's own Postgres is the production target (see PLAN.md).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from revops_ai.safety.write_intent import WriteIntent, WriteIntentStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunRecord(BaseModel):
    run_id: str
    agent_name: str
    task_type: str
    task_json: str
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    report_json: str | None = None
    error: str | None = None


class ToolCallEvent(BaseModel):
    """One tool invocation: inputs, output (JSON), or the error it raised."""

    run_id: str
    seq: int
    role: str
    method: str
    args_json: str
    result_json: str | None = None
    error: str | None = None
    at: datetime = Field(default_factory=_now)


class AuditLedger(Protocol):
    """Storage interface for runs, tool-call events, intents, and idempotency."""

    def record_run_started(self, record: RunRecord) -> None: ...
    def record_run_finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        report_json: str | None = None,
        error: str | None = None,
    ) -> None: ...
    def get_run(self, run_id: str) -> RunRecord | None: ...
    def list_runs(self) -> list[RunRecord]: ...

    def record_event(self, event: ToolCallEvent) -> None: ...
    def get_events(self, run_id: str) -> list[ToolCallEvent]: ...

    def record_intent(self, intent: WriteIntent) -> None: ...
    def update_intent(self, intent: WriteIntent) -> None: ...
    def get_intent(self, intent_id: str) -> WriteIntent | None: ...
    def list_intents(self, status: WriteIntentStatus | None = None) -> list[WriteIntent]: ...

    def was_applied(self, idempotency_key: str) -> bool: ...
    def mark_applied(self, idempotency_key: str) -> None: ...


class InMemoryAuditLedger:
    """Process-local ledger; the default for development and tests."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[ToolCallEvent]] = {}
        self._intents: dict[str, WriteIntent] = {}
        self._intent_order: list[str] = []
        self._applied_keys: set[str] = set()

    def record_run_started(self, record: RunRecord) -> None:
        self._runs[record.run_id] = record

    def record_run_finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        report_json: str | None = None,
        error: str | None = None,
    ) -> None:
        record = self._runs[run_id]
        record.status = status
        record.finished_at = _now()
        record.report_json = report_json
        record.error = error

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return sorted(self._runs.values(), key=lambda r: r.started_at)

    def record_event(self, event: ToolCallEvent) -> None:
        self._events.setdefault(event.run_id, []).append(event)

    def get_events(self, run_id: str) -> list[ToolCallEvent]:
        return list(self._events.get(run_id, []))

    def record_intent(self, intent: WriteIntent) -> None:
        if intent.intent_id not in self._intents:
            self._intent_order.append(intent.intent_id)
        self._intents[intent.intent_id] = intent

    def update_intent(self, intent: WriteIntent) -> None:
        self.record_intent(intent)

    def get_intent(self, intent_id: str) -> WriteIntent | None:
        return self._intents.get(intent_id)

    def list_intents(self, status: WriteIntentStatus | None = None) -> list[WriteIntent]:
        intents = [self._intents[i] for i in self._intent_order]
        if status is not None:
            intents = [i for i in intents if i.status is status]
        return intents

    def was_applied(self, idempotency_key: str) -> bool:
        return idempotency_key in self._applied_keys

    def mark_applied(self, idempotency_key: str) -> None:
        self._applied_keys.add(idempotency_key)
