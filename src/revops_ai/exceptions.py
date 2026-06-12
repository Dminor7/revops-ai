"""Exception hierarchy for revops-ai.

Every error raised by the SDK derives from :class:`RevOpsError` so callers can
catch SDK failures with a single except clause.
"""

from __future__ import annotations


class RevOpsError(Exception):
    """Base class for all revops-ai errors."""


class RegistrationError(RevOpsError):
    """An agent, connector, or task could not be registered.

    Raised at registration time — never mid-run — so misconfiguration is
    caught when the engine is assembled, not on a Monday morning.
    """


class UnknownRoleError(RegistrationError):
    """An agent requires a connector role that is not registered."""

    def __init__(self, agent_name: str, role: str, registered: list[str]) -> None:
        self.agent_name = agent_name
        self.role = role
        self.registered = registered
        super().__init__(
            f"Agent {agent_name!r} requires connector role {role!r}, but only "
            f"{registered or '[no roles]'} are registered. Call "
            f"engine.register_connector({role!r}, ...) before registering this agent."
        )


class CapabilityError(RegistrationError):
    """An agent requires a capability its connector does not declare."""


class NoAgentForTaskError(RevOpsError):
    """No registered agent handles the given task type."""

    def __init__(self, task_type: type) -> None:
        self.task_type = task_type
        super().__init__(
            f"No registered agent handles task type {task_type.__name__!r}. "
            "Register an agent whose task_type matches, or check the task you passed."
        )


class StaleDataError(RevOpsError):
    """A freshness policy with on_violation='fail' was violated."""


class ReplayError(RevOpsError):
    """A run could not be replayed from its recorded events."""


class WriteError(RevOpsError):
    """A write intent could not be processed (approval, application, lookup)."""
