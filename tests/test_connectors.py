"""Connector implementations: warehouse (SQLite), HubSpot and Stripe (respx)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from revops_ai import Capability, EntityRef, Evidence, FieldChange, SourceRef, WriteIntent
from revops_ai.connectors.hubspot import HubSpotConnector, HubSpotSettings
from revops_ai.connectors.stripe import StripeConnector, StripeSettings
from revops_ai.connectors.warehouse import WarehouseConnector

# -- warehouse ---------------------------------------------------------------


@pytest.fixture
async def warehouse_url(tmp_path: Path) -> str:
    from sqlalchemy.ext.asyncio import create_async_engine

    url = f"sqlite+aiosqlite:///{tmp_path}/wh.db"
    engine = create_async_engine(url)
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE deals (id TEXT, margin REAL)"))
        await conn.execute(text("INSERT INTO deals VALUES ('D-1', 52.0), ('D-2', 12.0)"))
    await engine.dispose()
    return url


async def test_warehouse_query_returns_rows_as_dicts(warehouse_url: str) -> None:
    wh = WarehouseConnector(warehouse_url)
    rows = await wh.query("SELECT id, margin FROM deals ORDER BY id")
    assert rows == [{"id": "D-1", "margin": 52.0}, {"id": "D-2", "margin": 12.0}]

    row = await wh.query("SELECT margin FROM deals WHERE id = :deal_id", {"deal_id": "D-1"})
    assert row == [{"margin": 52.0}]
    assert await wh.health_check() is True
    await wh.close()


async def test_warehouse_is_read_only_and_reports_vintage(warehouse_url: str) -> None:
    synced = datetime(2026, 6, 12, 9, 14, tzinfo=timezone.utc)
    wh = WarehouseConnector(warehouse_url, sync_resolver=lambda: synced)
    assert wh.capabilities == frozenset({Capability.READ})
    assert wh.sync_metadata().last_synced_at == synced
    assert wh.sync_metadata().source_system == "warehouse"
    await wh.close()


# -- hubspot -------------------------------------------------------------------


def _hubspot() -> HubSpotConnector:
    return HubSpotConnector(HubSpotSettings(access_token="test-token"))


@respx.mock
async def test_hubspot_get_deal() -> None:
    route = respx.get("https://api.hubapi.com/crm/v3/objects/deals/123").mock(
        return_value=httpx.Response(
            200, json={"id": "123", "properties": {"dealname": "Acme", "dealstage": "commit"}}
        )
    )
    deal = await _hubspot().get_deal("123")
    assert deal == {"id": "123", "dealname": "Acme", "dealstage": "commit"}
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-token"


@respx.mock
async def test_hubspot_search_deals_paginates() -> None:
    pages = [
        httpx.Response(
            200,
            json={
                "results": [{"id": "1", "properties": {"dealstage": "commit"}}],
                "paging": {"next": {"after": "cursor-1"}},
            },
        ),
        httpx.Response(
            200,
            json={"results": [{"id": "2", "properties": {"dealstage": "commit"}}]},
        ),
    ]
    route = respx.post("https://api.hubapi.com/crm/v3/objects/deals/search").mock(side_effect=pages)

    deals = await _hubspot().search_deals(stage="commit")

    assert [d["id"] for d in deals] == ["1", "2"]
    first_body = route.calls[0].request.read().decode()
    assert "commit" in first_body
    second_body = route.calls[1].request.read().decode()
    assert "cursor-1" in second_body


@respx.mock
async def test_hubspot_apply_write_patches_allowed_fields() -> None:
    route = respx.patch("https://api.hubapi.com/crm/v3/objects/deals/123").mock(
        return_value=httpx.Response(200, json={"id": "123"})
    )
    intent = WriteIntent(
        connector_role="crm",
        operation="update",
        target=EntityRef(source_system="hubspot", entity_type="deal", entity_id="123"),
        changes={"next_step": FieldChange(new="Schedule exec sync")},
        justification=[
            Evidence(
                kind="metric",
                summary="stalled",
                source=SourceRef(kind="query", reference="q"),
            )
        ],
    )
    result = await _hubspot().apply_write(intent)
    assert result == {"id": "123"}
    body = json.loads(route.calls.last.request.read())
    assert body == {"properties": {"next_step": "Schedule exec sync"}}


@respx.mock
async def test_hubspot_health_check() -> None:
    respx.get("https://api.hubapi.com/crm/v3/objects/deals").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    assert await _hubspot().health_check() is True


# -- stripe --------------------------------------------------------------------


def _stripe() -> StripeConnector:
    return StripeConnector(StripeSettings(api_key="sk_test_x"))


@respx.mock
async def test_stripe_get_customer_uses_basic_auth() -> None:
    route = respx.get("https://api.stripe.com/v1/customers/cus_1").mock(
        return_value=httpx.Response(200, json={"id": "cus_1", "email": "a@b.co"})
    )
    customer = await _stripe().get_customer("cus_1")
    assert customer["email"] == "a@b.co"
    assert route.calls.last.request.headers["Authorization"].startswith("Basic ")


@respx.mock
async def test_stripe_list_subscriptions_paginates() -> None:
    pages = [
        httpx.Response(
            200,
            json={"data": [{"id": "sub_1", "status": "active"}], "has_more": True},
        ),
        httpx.Response(
            200,
            json={"data": [{"id": "sub_2", "status": "active"}], "has_more": False},
        ),
    ]
    route = respx.get("https://api.stripe.com/v1/subscriptions").mock(side_effect=pages)

    subs = await _stripe().list_subscriptions(customer="cus_1")

    assert [s["id"] for s in subs] == ["sub_1", "sub_2"]
    assert "starting_after=sub_1" in str(route.calls[1].request.url)
    assert "customer=cus_1" in str(route.calls[0].request.url)


@respx.mock
async def test_stripe_health_check() -> None:
    respx.get("https://api.stripe.com/v1/customers").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    assert await _stripe().health_check() is True
