"""FastAPI serving layer: one call turns a configured engine into a service.

``create_api(engine)`` exposes typed task execution, the run ledger, the
intent approval queue, replay, health, and (optionally) verified webhook
endpoints. Install the ``server`` extra.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request, Response
except ImportError as exc:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "The serving layer requires FastAPI. Install it with: pip install 'revops-ai[server]'"
    ) from exc

from pydantic import BaseModel, ValidationError

from revops_ai.core.engine import RevOpsEngine
from revops_ai.exceptions import ReplayError, WriteError
from revops_ai.server.webhooks import WebhookListener, WebhookVerificationError


class RejectBody(BaseModel):
    reason: str


def create_api(engine: RevOpsEngine, *, webhooks: WebhookListener | None = None) -> FastAPI:
    """Build a FastAPI app over a fully-assembled engine."""
    from revops_ai import __version__

    app = FastAPI(title="revops-ai", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return await engine.health_check()

    @app.post("/tasks/{task_name}")
    async def run_task(task_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        task_cls = engine.task_types.get(task_name)
        if task_cls is None:
            raise HTTPException(
                404,
                detail=f"Unknown task {task_name!r}; registered: {sorted(engine.task_types)}.",
            )
        try:
            task = task_cls.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, detail=json.loads(exc.json())) from exc
        report = await engine.run(task)
        return {"run_id": report.run_id, "report": report.model_dump(mode="json")}

    @app.get("/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return [record.model_dump(mode="json") for record in engine.ledger.list_runs()]

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        record = engine.ledger.get_run(run_id)
        if record is None:
            raise HTTPException(404, detail=f"No run {run_id!r}.")
        return {
            "run": record.model_dump(mode="json"),
            "events": [e.model_dump(mode="json") for e in engine.ledger.get_events(run_id)],
        }

    @app.post("/runs/{run_id}/replay")
    async def replay_run(run_id: str) -> dict[str, Any]:
        try:
            report = await engine.replay(run_id)
        except ReplayError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        return {"run_id": report.run_id, "report": report.model_dump(mode="json")}

    @app.get("/intents/pending")
    async def pending_intents() -> list[dict[str, Any]]:
        return [intent.model_dump(mode="json") for intent in engine.pending_intents()]

    @app.post("/intents/{intent_id}/approve")
    async def approve_intent(intent_id: str) -> dict[str, Any]:
        try:
            intent = await engine.approve_intent(intent_id)
        except WriteError as exc:
            raise HTTPException(404 if "No intent" in str(exc) else 409, detail=str(exc)) from exc
        return intent.model_dump(mode="json")

    @app.post("/intents/{intent_id}/reject")
    async def reject_intent(intent_id: str, body: RejectBody) -> dict[str, Any]:
        try:
            intent = engine.reject_intent(intent_id, body.reason)
        except WriteError as exc:
            raise HTTPException(404 if "No intent" in str(exc) else 409, detail=str(exc)) from exc
        return intent.model_dump(mode="json")

    if webhooks is not None:

        @app.post("/webhooks/{source}")
        async def receive_webhook(
            source: str, request: Request, response: Response
        ) -> dict[str, Any]:
            if not webhooks.knows_source(source):
                raise HTTPException(404, detail=f"No webhook source {source!r} configured.")
            body = await request.body()
            try:
                webhooks.verify(
                    source,
                    method=request.method,
                    url=str(request.url),
                    body=body,
                    headers={k.lower(): v for k, v in request.headers.items()},
                )
            except WebhookVerificationError as exc:
                raise HTTPException(401, detail=str(exc)) from exc
            events = webhooks.extract_events(source, body)
            run_ids = await webhooks.dispatch(events)
            response.status_code = 200 if run_ids else 202
            return {"runs": run_ids}

    return app
