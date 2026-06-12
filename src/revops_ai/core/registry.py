"""Connector and agent registries.

The registries are plain, inspectable containers. All validation happens at
registration time so a misassembled engine fails on startup, not mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from revops_ai.connectors.base import Capability, Connector
from revops_ai.exceptions import CapabilityError, RegistrationError, UnknownRoleError
from revops_ai.tools.base import Tool


@dataclass(frozen=True)
class ConnectorEntry:
    connector: Connector
    #: Effective capabilities: the connector's own, optionally narrowed at
    #: registration (e.g. register a writable CRM read-only for this engine).
    capabilities: frozenset[Capability]


class ConnectorRegistry:
    """Role-keyed registry of connectors."""

    def __init__(self) -> None:
        self._entries: dict[str, ConnectorEntry] = {}

    def register(
        self,
        role: str,
        connector: Connector,
        *,
        capabilities: set[Capability] | None = None,
    ) -> None:
        if role in self._entries:
            raise RegistrationError(f"Connector role {role!r} is already registered.")
        if not isinstance(connector, Connector):
            raise RegistrationError(
                f"Object registered for role {role!r} does not satisfy the Connector "
                "protocol (needs capabilities, sync_metadata(), health_check())."
            )
        effective = connector.capabilities
        if capabilities is not None:
            extra = frozenset(capabilities) - effective
            if extra:
                raise RegistrationError(
                    f"Cannot grant {sorted(c.value for c in extra)} to role {role!r}: "
                    "the connector itself does not declare them."
                )
            effective = frozenset(capabilities)
        self._entries[role] = ConnectorEntry(connector=connector, capabilities=effective)

    def get(self, role: str) -> ConnectorEntry | None:
        return self._entries.get(role)

    @property
    def roles(self) -> list[str]:
        return sorted(self._entries)

    def resolve_tool(self, agent_name: str, role: str, tool_cls: type[Tool]) -> Tool:
        """Bind a tool class to the connector registered under ``role``."""
        entry = self._entries.get(role)
        if entry is None:
            raise UnknownRoleError(agent_name, role, self.roles)
        missing = tool_cls.required_capabilities - entry.capabilities
        if missing:
            raise CapabilityError(
                f"Agent {agent_name!r} needs {sorted(c.value for c in missing)} on role "
                f"{role!r}, but the connector is registered with "
                f"{sorted(c.value for c in entry.capabilities)}."
            )
        return tool_cls(connector=entry.connector, role=role)
