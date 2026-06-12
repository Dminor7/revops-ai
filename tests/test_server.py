"""FastAPI serving layer: typed task endpoints, runs, intents, webhooks."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from revops_ai import RevOpsEngine
from revops_ai.server import WebhookListener, create_api
from tests.conftest import CommissionAgent, CommissionTask, FakeWarehouseConnector
from tests.test_write_safety import ALLOW_NEXT_STEP, FakeCRMConnector, NudgeAgent


@pytest.fixture
def engine(warehouse: FakeWarehouseConnector) -> RevOpsEngine:
    engine = RevOpsEngine()
    engine.register_connector("warehouse", warehouse)
    engine.register_agent(CommissionAgent())
    return engine


def test_run_task_endpoint_returns_typed_report(engine: RevOpsEngine) -> None:
    client = TestClient(create_api(engine))
    response = client.post("/tasks/CommissionTask", json={"deal_id": "D-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["status"] == "approved"
    assert body["run_id"]

    # The run is in the ledger and visible via the API.
    run = client.get(f"/runs/{body['run_id']}")
    assert run.status_code == 200
    assert run.json()["run"]["agent_name"] == "commission_calculator"
    assert run.json()["events"][0]["method"] == "query_one"


def test_unknown_task_and_validation_errors(engine: RevOpsEngine) -> None:
    client = TestClient(create_api(engine))
    assert client.post("/tasks/NopeTask", json={}).status_code == 404
    assert client.post("/tasks/CommissionTask", json={}).status_code == 422
    assert client.get("/runs/missing").status_code == 404


def test_replay_endpoint(engine: RevOpsEngine, warehouse: FakeWarehouseConnector) -> None:
    client = TestClient(create_api(engine))
    run_id = client.post("/tasks/CommissionTask", json={"deal_id": "D-1"}).json()["run_id"]
    warehouse.rows["D-1"]["margin"] = 1.0  # change live data

    response = client.post(f"/runs/{run_id}/replay")
    assert response.status_code == 200
    assert response.json()["report"]["status"] == "approved"  # recorded decision


def test_intent_approval_flow_via_api() -> None:
    crm = FakeCRMConnector()
    engine = RevOpsEngine(write_policy=ALLOW_NEXT_STEP, write_mode="apply")
    engine.register_connector("crm", crm)
    engine.register_agent(NudgeAgent())
    client = TestClient(create_api(engine))

    client.post("/tasks/NudgeTask", json={"deal_id": "D-9"})
    pending = client.get("/intents/pending").json()
    assert len(pending) == 1
    intent_id = pending[0]["intent_id"]

    approved = client.post(f"/intents/{intent_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "applied"
    assert len(crm.applied) == 1
    assert client.get("/intents/pending").json() == []

    # Approving twice is a 409, not a double write.
    assert client.post(f"/intents/{intent_id}/approve").status_code == 409


def test_reject_intent_via_api() -> None:
    crm = FakeCRMConnector()
    engine = RevOpsEngine(write_policy=ALLOW_NEXT_STEP, write_mode="apply")
    engine.register_connector("crm", crm)
    engine.register_agent(NudgeAgent())
    client = TestClient(create_api(engine))

    client.post("/tasks/NudgeTask", json={"deal_id": "D-9"})
    intent_id = client.get("/intents/pending").json()[0]["intent_id"]
    rejected = client.post(f"/intents/{intent_id}/reject", json={"reason": "wrong deal"})
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert crm.applied == []


def test_health_endpoint(engine: RevOpsEngine) -> None:
    client = TestClient(create_api(engine))
    assert client.get("/health").json() == {"warehouse": True}


# -- webhooks ------------------------------------------------------------------


def _stripe_signature(secret: str, body: bytes) -> str:
    timestamp = str(int(time.time()))
    signature = hmac.new(
        secret.encode(), f"{timestamp}.{body.decode()}".encode(), hashlib.sha256
    ).hexdigest()
    return f"t={timestamp},v1={signature}"


def _stripe_event() -> dict[str, Any]:
    return {
        "type": "customer.subscription.deleted",
        "data": {"object": {"id": "sub_1", "customer": "cus_1"}},
    }


@pytest.fixture
def webhook_app(engine: RevOpsEngine) -> TestClient:
    listener = WebhookListener(engine, secrets={"stripe": "whsec_test", "hubspot": "hs_secret"})
    listener.bind(
        "stripe",
        "customer.subscription.deleted",
        lambda event: CommissionTask(deal_id="D-1"),
    )
    return TestClient(create_api(engine, webhooks=listener))


def test_stripe_webhook_with_valid_signature_runs_bound_task(
    webhook_app: TestClient, engine: RevOpsEngine
) -> None:
    body = json.dumps(_stripe_event()).encode()
    response = webhook_app.post(
        "/webhooks/stripe",
        content=body,
        headers={"Stripe-Signature": _stripe_signature("whsec_test", body)},
    )
    assert response.status_code == 200
    run_ids = response.json()["runs"]
    assert len(run_ids) == 1
    assert engine.ledger.get_run(run_ids[0]) is not None


def test_stripe_webhook_rejects_bad_signature(webhook_app: TestClient) -> None:
    body = json.dumps(_stripe_event()).encode()
    response = webhook_app.post(
        "/webhooks/stripe",
        content=body,
        headers={"Stripe-Signature": _stripe_signature("wrong_secret", body)},
    )
    assert response.status_code == 401


def test_stripe_webhook_rejects_missing_signature(webhook_app: TestClient) -> None:
    response = webhook_app.post("/webhooks/stripe", json=_stripe_event())
    assert response.status_code == 401


def test_unbound_event_is_accepted_but_runs_nothing(webhook_app: TestClient) -> None:
    body = json.dumps({"type": "invoice.paid", "data": {"object": {"id": "in_1"}}}).encode()
    response = webhook_app.post(
        "/webhooks/stripe",
        content=body,
        headers={"Stripe-Signature": _stripe_signature("whsec_test", body)},
    )
    assert response.status_code == 202
    assert response.json()["runs"] == []


def test_hubspot_webhook_v3_signature(webhook_app: TestClient, engine: RevOpsEngine) -> None:
    listener_events = [{"subscriptionType": "deal.creation", "objectId": 42}]
    body = json.dumps(listener_events).encode()
    timestamp = str(int(time.time() * 1000))
    uri = "http://testserver/webhooks/hubspot"
    digest = hmac.new(
        b"hs_secret", b"POST" + uri.encode() + body + timestamp.encode(), hashlib.sha256
    ).digest()
    headers = {
        "X-HubSpot-Signature-v3": base64.b64encode(digest).decode(),
        "X-HubSpot-Request-Timestamp": timestamp,
    }

    # No binding for deal.creation: verified but nothing runs.
    response = webhook_app.post("/webhooks/hubspot", content=body, headers=headers)
    assert response.status_code == 202

    # Tampered body fails verification.
    response = webhook_app.post("/webhooks/hubspot", content=body + b" ", headers=headers)
    assert response.status_code == 401


def test_unknown_webhook_source_is_404(webhook_app: TestClient) -> None:
    assert webhook_app.post("/webhooks/zendesk", content=b"{}").status_code == 404
