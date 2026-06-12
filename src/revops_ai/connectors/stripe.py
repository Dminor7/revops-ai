"""Stripe billing connector (async, httpx). Read-only by design.

Billing write-back is out of scope for GTM agents; keeping this connector
read-only means a misrouted intent can never touch revenue records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from revops_ai.connectors.base import Capability, SyncMetadata


class StripeSettings(BaseSettings):
    """Reads STRIPE_API_KEY from the environment."""

    model_config = SettingsConfigDict(env_prefix="STRIPE_")

    api_key: SecretStr
    base_url: str = "https://api.stripe.com"


class StripeConnector:
    """Customer and subscription reads. Live API: data vintage is 'now'."""

    def __init__(
        self,
        settings: StripeSettings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings if settings is not None else StripeSettings()  # type: ignore[call-arg]
        self._client = http_client or httpx.AsyncClient(
            base_url=self._settings.base_url,
            auth=(self._settings.api_key.get_secret_value(), ""),
            timeout=30.0,
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.READ})

    def sync_metadata(self) -> SyncMetadata:
        return SyncMetadata(source_system="stripe", last_synced_at=datetime.now(timezone.utc))

    async def health_check(self) -> bool:
        response = await self._client.get("/v1/customers", params={"limit": 1})
        return response.status_code < 400

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/v1/customers/{customer_id}")
        response.raise_for_status()
        return dict(response.json())

    async def list_subscriptions(
        self,
        *,
        customer: str | None = None,
        status: str | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """List subscriptions, following pagination up to ``max_results``."""
        params: dict[str, Any] = {"limit": min(max_results, 100)}
        if customer:
            params["customer"] = customer
        if status:
            params["status"] = status
        subscriptions: list[dict[str, Any]] = []
        while len(subscriptions) < max_results:
            response = await self._client.get("/v1/subscriptions", params=params)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", [])
            subscriptions.extend(data)
            if not payload.get("has_more") or not data:
                break
            params["starting_after"] = data[-1]["id"]
        return subscriptions[:max_results]

    async def close(self) -> None:
        await self._client.aclose()
