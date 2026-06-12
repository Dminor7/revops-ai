"""WriteIntent: agents propose, policies dispose.

Agents never call write APIs. They emit ``WriteIntent`` objects via
``ctx.propose_write(...)``; the engine routes every intent through the
write-policy pipeline (allowlists, blast-radius limits, approval, idempotency)
before anything reaches a connector.
"""

from __future__ import annotations

import enum
import hashlib
import json
import uuid
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator

from revops_ai.audit.evidence import EntityRef, Evidence


class FieldChange(BaseModel):
    """One field-level change: what it was, what it becomes."""

    old: JsonValue = None
    new: JsonValue


class WriteIntentStatus(str, enum.Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


class WriteIntent(BaseModel):
    """A proposed write against a connector, with the evidence justifying it."""

    intent_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str | None = None
    connector_role: str
    operation: Literal["update", "create"]
    target: EntityRef
    changes: dict[str, FieldChange] = Field(min_length=1)
    justification: list[Evidence] = Field(min_length=1)
    idempotency_key: str = ""
    status: WriteIntentStatus = WriteIntentStatus.PROPOSED
    status_reason: str | None = None

    @model_validator(mode="after")
    def _derive_idempotency_key(self) -> WriteIntent:
        if not self.idempotency_key:
            payload = json.dumps(
                {
                    "role": self.connector_role,
                    "op": self.operation,
                    "target": self.target.model_dump(),
                    "changes": {k: v.new for k, v in sorted(self.changes.items())},
                },
                sort_keys=True,
                default=str,
            )
            object.__setattr__(
                self, "idempotency_key", hashlib.sha256(payload.encode()).hexdigest()
            )
        return self
