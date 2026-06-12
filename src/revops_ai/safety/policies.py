"""Write policies: what agents may change, how much, and what needs a human.

The default posture is maximally conservative: an engine with no
``WritePolicy`` rejects every intent, and a policy only permits fields that
are explicitly allowlisted per connector role.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from revops_ai.safety.write_intent import WriteIntent


@dataclass
class WritePolicy:
    """Declarative limits on agent writes.

    - ``field_allowlists``: role -> fields agents may change. A role with no
      allowlist entry accepts nothing.
    - ``max_writes_per_run``: blast-radius circuit breaker; intents beyond the
      limit are rejected, not queued.
    - ``auto_approve``: predicate for intents that may skip human approval
      when the engine runs in ``write_mode="apply"``. Default: nothing
      auto-approves.
    """

    field_allowlists: dict[str, set[str]] = field(default_factory=dict)
    max_writes_per_run: int = 50
    auto_approve: Callable[[WriteIntent], bool] | None = None

    def violation(self, intent: WriteIntent) -> str | None:
        """Reason this intent is not allowed, or None if it passes."""
        allowed = self.field_allowlists.get(intent.connector_role)
        if allowed is None:
            return (
                f"No field allowlist configured for role {intent.connector_role!r}; "
                "writes to this role are rejected."
            )
        disallowed = sorted(set(intent.changes) - allowed)
        if disallowed:
            return (
                f"Fields {disallowed} are not in the allowlist for role "
                f"{intent.connector_role!r} (allowed: {sorted(allowed)})."
            )
        return None
