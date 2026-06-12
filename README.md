# revops-ai

An extensible SDK for building **auditable, write-safe RevOps AI agents** — built for GTM
AI Engineers who need agent decisions they can defend in front of a VP, not just demo.

> Status: **alpha**. Core engine, trust layer (audit ledger, write intents, replay),
> connectors (HubSpot, Stripe, SQL warehouse), serving layer, and two shipped agents are
> functional and fully tested. See [PLAN.md](PLAN.md) for the architecture and roadmap.

## Design principles

- **Inversion of control.** You inject connectors (with your credentials), an LLM
  configuration, and policies into a `RevOpsEngine`. The SDK ships the orchestration;
  your environment owns the secrets and the data layer.
- **Typed everything.** Tasks in, reports out — all pydantic models. Natural language
  (`engine.run_analysis("...")`) is a router that parses text into a registered task,
  never a parallel code path.
- **Evidence, not verdicts.** Findings structurally require evidence; every report
  carries the vintage of the data it was computed from.
- **Agents propose, policies dispose.** No agent can call a write API. Writes are
  `WriteIntent`s flowing through field allowlists, blast-radius limits, a dry-run
  default, approval queues, and idempotency keys.
- **Every run is reconstructable.** The ledger records the task, every tool call, every
  intent transition; `engine.replay(run_id)` re-executes a run from recorded tool
  outputs without touching live systems.
- **Fail at assembly, not mid-run.** Missing connectors, capabilities, or model config
  raise when you register an agent.
- **Rent the plumbing.** The LLM loop is [pydantic-ai](https://ai.pydantic.dev);
  validation is pydantic; SQL is SQLAlchemy; serving is FastAPI. The SDK's own code is
  the GTM domain layer: guardrails, audit, freshness, evidence.

## Install

```bash
pip install revops-ai                       # core
pip install 'revops-ai[warehouse,server]'   # extras: hubspot, salesforce, stripe,
                                            # warehouse, server, openai, anthropic
```

## Quick start: shipped agents

```python
from revops_ai import RevOpsEngine, LLMConfig, FreshnessPolicy, WritePolicy
from revops_ai.agents import PipelineVelocityAgent, ChurnPredictorAgent, PipelineEvaluationTask
from revops_ai.connectors.hubspot import HubSpotConnector   # reads HUBSPOT_ACCESS_TOKEN
from revops_ai.connectors.stripe import StripeConnector     # reads STRIPE_API_KEY
from revops_ai.connectors.warehouse import WarehouseConnector

engine = RevOpsEngine(
    llm=LLMConfig(model="openai:gpt-4o", fallback="anthropic:claude-sonnet-4-6"),
)
engine.register_connector("crm", HubSpotConnector())
engine.register_connector("billing", StripeConnector())
engine.register_connector("warehouse", WarehouseConnector("postgresql+asyncpg://..."))

engine.register_agent(PipelineVelocityAgent())
engine.register_agent(ChurnPredictorAgent(sensitivity="high"))

report = await engine.run(PipelineEvaluationTask(stage="Commit", quarter="2026-Q3"))
print(report.to_markdown())        # evidence-backed findings + data vintage
print(report.run_id)               # fully reconstructable from the ledger

# Or route natural language onto the same typed, audited path:
report = await engine.run_analysis("Evaluate all open deals in the Commit stage for Q3.")
```

## Custom agents

Tools are *declared*, not constructed — the engine binds them to your connectors at
registration, so one agent class serves many environments (multi-tenant safe):

```python
from revops_ai import BaseAgent, Report, RunContext, Task
from revops_ai.tools import SQLQueryTool

class CommissionTask(Task):
    deal_id: str

class CommissionReport(Report):
    deal_id: str
    payout_tier: str

class CommissionAgent(BaseAgent[CommissionTask, CommissionReport]):
    name = "commission_calculator"
    description = "Calculates commissions from territory rules."
    requires = {"warehouse": SQLQueryTool}
    task_type = CommissionTask
    report_type = CommissionReport

    async def process_task(self, ctx: RunContext, task: CommissionTask) -> CommissionReport:
        row = await ctx.tools.warehouse.query_one(
            "SELECT margin FROM deals WHERE id = :deal_id", deal_id=task.deal_id
        )
        tier = "Alpha - 15%" if float(row["margin"]) > 40 else "Standard - 8%"
        return CommissionReport(deal_id=task.deal_id, payout_tier=tier)
```

LLM-backed agents subclass `LLMAgent` and get a typed pydantic-ai loop with the engine's
configured model; deterministic agents need no model at all.

## Write-back safety

```python
from revops_ai import EntityRef, FieldChange, WriteIntent, WritePolicy

engine = RevOpsEngine(
    write_policy=WritePolicy(
        field_allowlists={"crm": {"next_step", "risk_note"}},  # nothing else, ever
        max_writes_per_run=25,                                  # blast-radius breaker
    ),
    write_mode="apply",  # default is "dry_run": intents are produced, nothing written
)

# Inside an agent: propose, never write.
ctx.propose_write(WriteIntent(
    connector_role="crm", operation="update",
    target=EntityRef(source_system="hubspot", entity_type="deal", entity_id="123"),
    changes={"next_step": FieldChange(new="Schedule exec sync")},
    justification=[...],   # Evidence required by schema
))

# Outside: human-in-the-loop.
engine.pending_intents()
await engine.approve_intent(intent_id)   # applied with idempotency
engine.reject_intent(intent_id, "wrong deal")
```

## Audit and replay

```python
record = engine.ledger.get_run(report.run_id)     # task, agent, status, report
events = engine.ledger.get_events(report.run_id)  # every tool call, inputs and outputs

replayed = await engine.replay(report.run_id)     # recorded data, no live access,
                                                  # writes never applied
```

## Serve it

```python
from revops_ai.server import WebhookListener, create_api

listener = WebhookListener(engine, secrets={"hubspot": "...", "stripe": "..."})
listener.bind("hubspot", "deal.propertyChange",
              lambda event: CommissionTask(deal_id=event.object_id))

app = create_api(engine, webhooks=listener)
# uvicorn main:app — typed task endpoints, run ledger, approval queue,
# replay, health, and signature-verified webhooks (unsigned => 401).
```

## Freshness policies

```python
from datetime import timedelta
engine = RevOpsEngine(freshness=FreshnessPolicy(max_staleness=timedelta(hours=6),
                                                on_violation="fail"))
```

Every connector reports `sync_metadata()` (warehouse connectors accept a
`sync_resolver` to surface your ELT tool's real sync time); runs against stale data warn
or abort, and every report's `data_vintage` records how fresh each source was.

## Development

```bash
uv venv && uv pip install -e . --group dev
pytest                       # 58 tests, all offline (respx, SQLite, FunctionModel)
ruff check src tests && mypy # mypy --strict
```

## Roadmap

See [PLAN.md](PLAN.md). Done: M0 core, M1 data layer (HubSpot/Stripe/warehouse,
retrieval adapter, NL router), M2 trust layer (ledger, intents, replay), M3 serving
(API + verified webhooks), M4 shipped agents. Next: SQLAlchemy-backed ledger +
Alembic, OTel/Langfuse exporter, pgvector/Qdrant adapters, Salesforce connector, CLI,
Temporal runner, docs site, PyPI release.
