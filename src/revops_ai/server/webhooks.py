"""Event-driven entry: CRM/billing webhooks mapped to typed tasks.

Signature verification is mandatory — a source with no configured secret
cannot receive events, and unsigned or tampered payloads are rejected with
401, never processed "best effort". Bound events enter the exact same typed,
ledgered ``engine.run()`` path as every other invocation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from revops_ai.tasks.base import Task

if TYPE_CHECKING:
    from revops_ai.core.engine import RevOpsEngine


class WebhookEvent(BaseModel):
    """A normalized inbound event, regardless of source format."""

    source: str
    event_type: str
    object_id: str | None = None
    payload: dict[str, Any]


TaskFactory = Callable[[WebhookEvent], Task]


class WebhookVerificationError(Exception):
    """Signature missing, stale, or invalid."""


class WebhookListener:
    """Maps verified webhook events to task factories.

    ``secrets`` is the per-source signing secret (Stripe webhook secret,
    HubSpot app secret). A source absent from ``secrets`` does not exist as
    an endpoint.
    """

    def __init__(
        self,
        engine: RevOpsEngine,
        *,
        secrets: dict[str, str],
        tolerance_seconds: int = 300,
    ) -> None:
        self.engine = engine
        self._secrets = secrets
        self._tolerance = tolerance_seconds
        self._bindings: dict[tuple[str, str], TaskFactory] = {}

    def bind(self, source: str, event_type: str, task_factory: TaskFactory) -> None:
        """Route ``(source, event_type)`` to a task. One binding per pair."""
        self._bindings[(source, event_type)] = task_factory

    def knows_source(self, source: str) -> bool:
        return source in self._secrets

    def verify(
        self, source: str, *, method: str, url: str, body: bytes, headers: dict[str, str]
    ) -> None:
        """Raise WebhookVerificationError unless the request is authentic."""
        secret = self._secrets.get(source)
        if secret is None:
            raise WebhookVerificationError(f"No signing secret configured for {source!r}.")
        if source == "stripe":
            self._verify_stripe(secret, body, headers.get("stripe-signature"))
        elif source == "hubspot":
            self._verify_hubspot(
                secret,
                method,
                url,
                body,
                headers.get("x-hubspot-signature-v3"),
                headers.get("x-hubspot-request-timestamp"),
            )
        else:
            raise WebhookVerificationError(
                f"No signature scheme implemented for source {source!r}."
            )

    def _verify_stripe(self, secret: str, body: bytes, header: str | None) -> None:
        if not header:
            raise WebhookVerificationError("Missing Stripe-Signature header.")
        parts = dict(item.split("=", 1) for item in header.split(",") if "=" in item)
        timestamp, signature = parts.get("t"), parts.get("v1")
        if not timestamp or not signature:
            raise WebhookVerificationError("Malformed Stripe-Signature header.")
        if abs(time.time() - int(timestamp)) > self._tolerance:
            raise WebhookVerificationError("Stripe signature timestamp outside tolerance.")
        expected = hmac.new(
            secret.encode(), f"{timestamp}.{body.decode()}".encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise WebhookVerificationError("Stripe signature mismatch.")

    def _verify_hubspot(
        self,
        secret: str,
        method: str,
        url: str,
        body: bytes,
        signature: str | None,
        timestamp: str | None,
    ) -> None:
        if not signature or not timestamp:
            raise WebhookVerificationError("Missing HubSpot v3 signature headers.")
        if abs(time.time() * 1000 - int(timestamp)) > self._tolerance * 1000:
            raise WebhookVerificationError("HubSpot signature timestamp outside tolerance.")
        digest = hmac.new(
            secret.encode(),
            method.upper().encode() + url.encode() + body + timestamp.encode(),
            hashlib.sha256,
        ).digest()
        expected = base64.b64encode(digest).decode()
        if not hmac.compare_digest(expected, signature):
            raise WebhookVerificationError("HubSpot signature mismatch.")

    def extract_events(self, source: str, body: bytes) -> list[WebhookEvent]:
        """Normalize a verified payload into events."""
        payload = json.loads(body)
        if source == "stripe":
            obj = payload.get("data", {}).get("object", {})
            return [
                WebhookEvent(
                    source=source,
                    event_type=str(payload.get("type", "")),
                    object_id=obj.get("id"),
                    payload=payload,
                )
            ]
        if source == "hubspot":
            items = payload if isinstance(payload, list) else [payload]
            return [
                WebhookEvent(
                    source=source,
                    event_type=str(item.get("subscriptionType", "")),
                    object_id=str(item["objectId"]) if "objectId" in item else None,
                    payload=item,
                )
                for item in items
            ]
        raise WebhookVerificationError(f"Unknown source {source!r}.")

    async def dispatch(self, events: list[WebhookEvent]) -> list[str]:
        """Run the bound task for each event; returns the run ids."""
        run_ids: list[str] = []
        for event in events:
            factory = self._bindings.get((event.source, event.event_type))
            if factory is None:
                continue
            report = await self.engine.run(factory(event))
            assert report.run_id is not None
            run_ids.append(report.run_id)
        return run_ids
