"""HubSpot CRM connector (async, httpx).

Read methods cover deals; writes happen exclusively through
:meth:`apply_write`, which the engine calls for approved intents only —
agents cannot reach it (the method is not exposed on any tool).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from revops_ai.connectors.base import Capability, SyncMetadata
from revops_ai.exceptions import WriteError
from revops_ai.safety.write_intent import WriteIntent


class HubSpotSettings(BaseSettings):
    """Reads HUBSPOT_ACCESS_TOKEN (private-app token) from the environment."""

    model_config = SettingsConfigDict(env_prefix="HUBSPOT_")

    access_token: SecretStr
    base_url: str = "https://api.hubapi.com"


class HubSpotConnector:
    """Deal-centric HubSpot access. Live API: data vintage is 'now'."""

    def __init__(
        self,
        settings: HubSpotSettings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings if settings is not None else HubSpotSettings()  # type: ignore[call-arg]
        self._client = http_client or httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers={"Authorization": f"Bearer {self._settings.access_token.get_secret_value()}"},
            timeout=30.0,
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.READ, Capability.WRITE})

    def sync_metadata(self) -> SyncMetadata:
        return SyncMetadata(source_system="hubspot", last_synced_at=datetime.now(timezone.utc))

    async def health_check(self) -> bool:
        response = await self._client.get("/crm/v3/objects/deals", params={"limit": 1})
        return response.status_code < 400

    async def get_deal(self, deal_id: str, properties: list[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if properties:
            params["properties"] = ",".join(properties)
        response = await self._client.get(f"/crm/v3/objects/deals/{deal_id}", params=params)
        response.raise_for_status()
        payload = response.json()
        return {"id": payload["id"], **payload.get("properties", {})}

    async def search_deals(
        self,
        *,
        stage: str | None = None,
        properties: list[str] | None = None,
        max_results: int = 200,
    ) -> list[dict[str, Any]]:
        """Search deals, following pagination up to ``max_results``."""
        body: dict[str, Any] = {"limit": min(max_results, 100)}
        if properties:
            body["properties"] = properties
        if stage:
            body["filterGroups"] = [
                {"filters": [{"propertyName": "dealstage", "operator": "EQ", "value": stage}]}
            ]
        deals: list[dict[str, Any]] = []
        after: str | None = None
        while len(deals) < max_results:
            page_body = {**body, **({"after": after} if after else {})}
            response = await self._client.post("/crm/v3/objects/deals/search", json=page_body)
            response.raise_for_status()
            payload = response.json()
            deals.extend(
                {"id": item["id"], **item.get("properties", {})}
                for item in payload.get("results", [])
            )
            after = payload.get("paging", {}).get("next", {}).get("after")
            if not after:
                break
        return deals[:max_results]

    async def apply_write(self, intent: WriteIntent) -> dict[str, Any]:
        """Apply an approved intent. Called by the engine only."""
        if intent.target.entity_type != "deal":
            raise WriteError(
                f"HubSpotConnector can only write deals, got {intent.target.entity_type!r}."
            )
        properties = {field: change.new for field, change in intent.changes.items()}
        if intent.operation == "update":
            response = await self._client.patch(
                f"/crm/v3/objects/deals/{intent.target.entity_id}",
                json={"properties": properties},
            )
        else:
            response = await self._client.post(
                "/crm/v3/objects/deals", json={"properties": properties}
            )
        response.raise_for_status()
        return dict(response.json())

    async def close(self) -> None:
        await self._client.aclose()
