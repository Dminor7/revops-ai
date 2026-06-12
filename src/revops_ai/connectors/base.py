"""Connector protocol and supporting types.

A connector is anything that gives agents access to a system of record: a CRM,
a billing system, a data warehouse. Connectors are registered on the engine
under a *role* (e.g. ``"crm"``, ``"billing"``, ``"warehouse"``) and agents
declare which roles they require.

The SDK ships concrete connectors as optional extras; anything that satisfies
the :class:`Connector` protocol can be registered without subclassing.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Capability(str, enum.Enum):
    """What a connector is allowed to do.

    Write capability is declared here but exercised only through the
    write-intent pipeline (M2); agents never call write APIs directly.
    """

    READ = "read"
    WRITE = "write"


class SyncMetadata(BaseModel):
    """How fresh a connector's data is.

    Live-API connectors report ``last_synced_at=now``. Warehouse connectors
    should surface the timestamp of the most recent sync from their ELT
    tooling so reports can carry an honest data vintage.
    """

    source_system: str
    last_synced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def age(self) -> float:
        """Seconds since the last sync."""
        return (datetime.now(timezone.utc) - self.last_synced_at).total_seconds()


@runtime_checkable
class Connector(Protocol):
    """Structural interface every connector must satisfy."""

    @property
    def capabilities(self) -> frozenset[Capability]: ...

    def sync_metadata(self) -> SyncMetadata: ...

    async def health_check(self) -> bool: ...
