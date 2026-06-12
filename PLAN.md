# revops-ai SDK — Architecture & Implementation Plan

Target user: GTM AI Engineers at a ~500-person SaaS company. The bar is not "works in a
demo" — it's "survives a Monday morning incident review": every agent decision is
auditable, every CRM write is gated, and stale data is surfaced, not silently consumed.

Design principle: **own the thin domain layer (GTM semantics, guardrails, audit), rent
everything else** from proven open-source packages. We do not write our own LLM client,
retry loop, HTTP stack, or tracing system.

---

## 1. Open-source foundation (the "rent" list)

| Layer | Package | Why this one |
|---|---|---|
| Schemas / validation | `pydantic` v2 | De-facto standard; powers every typed boundary in the SDK |
| Configuration | `pydantic-settings` | Env-var injection with validation; no hand-rolled `os.environ` |
| Agent runtime | `pydantic-ai` | Model-agnostic (OpenAI/Anthropic/Gemini/local), typed tool calling, structured outputs, and first-class dependency injection (`deps_type`) — maps 1:1 to our IoC story |
| Multi-provider LLM fallback | via `pydantic-ai` model abstraction (or `litellm` if we outgrow it) | Avoid coupling the public API to one provider string like `"gpt-4o"` |
| HTTP | `httpx` | Async-first, used by the SDKs below anyway |
| Retries / backoff | `tenacity` | Battle-tested; declarative policies per connector |
| CRM connectors | `hubspot-api-client`, `simple-salesforce` | Official / canonical SDKs; we wrap, never reimplement auth or pagination |
| Billing connector | `stripe` | Official SDK; includes webhook signature verification |
| Warehouse / SQL | `sqlalchemy` 2.x (+ driver extras: `snowflake-sqlalchemy`, `sqlalchemy-bigquery`, `psycopg`) | One query interface across Postgres/Snowflake/BigQuery |
| Vector stores | `pgvector` (SQLAlchemy), `qdrant-client` | Behind our adapter interface (§3.3); pgvector is the zero-extra-infra default |
| Audit ledger persistence | `sqlalchemy` + `alembic` | The run ledger (§5.3) is just tables; migrations ship with the SDK |
| Serving | `fastapi` + `uvicorn` | The `create_api(engine)` story; OpenAPI docs for free |
| Background / scheduled runs | in-process by default; `temporalio` as an optional extra | Don't force Temporal on day one, but durable execution is the only honest answer for "replay last Tuesday's commission run" |
| Observability | `opentelemetry-sdk` (GenAI semantic conventions) + optional `langfuse` exporter (self-hostable, MIT) | Enterprises already have OTel collectors; Langfuse gives LLM-native trace UI without SaaS lock-in |
| Data ingestion (optional extra) | `dlt` | For teams without a warehouse sync; we don't build ELT ourselves |
| Dev/test | `pytest`, `respx` (httpx mocking), `vcrpy`, `ruff`, `mypy`, `pre-commit` | Standard quality gate |

Packaging: `pyproject.toml` (hatchling), `src/` layout, Python ≥3.10. Connectors and
heavy deps ship as extras so `pip install revops-ai` stays lean:

```
pip install revops-ai[hubspot,stripe,snowflake,server,temporal,langfuse]
```

---

## 2. Package layout

```
src/revops_ai/
├── core/
│   ├── engine.py          # RevOpsEngine: DI container + run orchestration
│   ├── registry.py        # ConnectorRegistry, AgentRegistry, ToolRegistry
│   ├── context.py         # RunContext: what agents see at runtime
│   └── settings.py        # pydantic-settings models
├── agents/
│   ├── base.py            # BaseAgent (wraps pydantic-ai Agent)
│   ├── pipeline_velocity.py
│   └── churn_predictor.py
├── connectors/
│   ├── base.py            # Connector protocol + capability flags (read/write)
│   ├── hubspot.py, salesforce.py, stripe.py, warehouse.py
├── retrieval/
│   ├── base.py            # VectorStoreAdapter protocol
│   ├── pgvector.py, qdrant.py
├── tasks/
│   ├── base.py            # Task / Report pydantic models
│   └── library.py         # PipelineEvaluationTask, ChurnScanTask, ...
├── safety/
│   ├── write_intent.py    # WriteIntent, DryRunResult
│   ├── approval.py        # ApprovalQueue + policies
│   └── policies.py        # rate limits, field allowlists, idempotency
├── audit/
│   ├── ledger.py          # run/event ledger (SQLAlchemy models)
│   ├── evidence.py        # Evidence objects attached to findings
│   └── replay.py          # deterministic replay from ledger
├── freshness/
│   └── policy.py          # FreshnessPolicy, SyncMetadata, staleness errors
├── server/
│   ├── api.py             # create_api(engine) -> FastAPI app
│   └── webhooks.py        # WebhookListener + per-source signature verification
└── telemetry/
    └── otel.py            # OTel GenAI spans; optional Langfuse exporter
```

---

## 3. Core abstractions (fixing the four critique items)

### 3.1 Connector registry, not named kwargs

`crm=` / `billing=` as constructor kwargs is replaced by a typed registry keyed by
**role**, with capability declarations:

```python
engine = RevOpsEngine(llm=LLMConfig(model="openai:gpt-4o", fallback="anthropic:claude-sonnet-4-6"))
engine.register_connector("crm", HubSpotConnector(settings=HubSpotSettings()))      # reads env
engine.register_connector("crm_legacy", SalesforceConnector(...), capabilities={"read"})
engine.register_connector("billing", StripeConnector(...))
engine.register_connector("warehouse", WarehouseConnector(url=env("WAREHOUSE_URL")))
```

- Roles are free-form strings; well-known roles (`crm`, `billing`, `warehouse`) get
  typed accessors on `RunContext` (`ctx.crm.get_deal(...)`).
- A `Connector` is a `Protocol`: `health_check()`, `capabilities`, `sync_metadata()`.
  Anyone can implement a custom one without subclassing our classes.
- Multi-instance is native (two CRMs during a migration is the *normal* enterprise case).

### 3.2 Tools injected at registration, not class attributes

Class-level `tools = [SQLQueryTool(db_connection="...")]` bakes credentials into class
definitions and breaks multi-tenancy. Instead, agents *declare* requirements; the engine
*resolves* them at registration:

```python
class CustomCommissionAgent(BaseAgent):
    name = "commission_calculator"
    description = "Calculates commissions from territory rules."
    requires = {"warehouse": SQLQueryTool, "crm": CRMReadTool}   # declaration only

    async def process_task(self, ctx: RunContext, task: CommissionTask) -> CommissionReport:
        margin = await ctx.tools.warehouse.query_one(MARGIN_SQL, deal_id=task.deal_id)
        ...
```

`engine.register_agent(CustomCommissionAgent())` wires `requires` against the connector
registry and fails **at registration time** (not mid-run) if a requirement is missing.
Two engine instances (two tenants) can host the same agent class with different
connectors. Under the hood, `BaseAgent` builds a `pydantic-ai` `Agent` with these tools
and `deps_type=RunContext` — we get typed tool calling, streaming, retries, and usage
tracking without writing an agent loop.

### 3.3 VectorStoreAdapter, not a connection string

```python
class VectorStoreAdapter(Protocol):
    async def upsert(self, docs: Sequence[Document]) -> None: ...
    async def search(self, query: str, *, k: int, filter: MetadataFilter | None) -> list[ScoredDocument]: ...
```

Ship `PgVectorStore` (default — most SaaS companies already run Postgres) and
`QdrantStore`. `vector_store="postgres://..."` remains as sugar that constructs
`PgVectorStore`, so the simple path stays one line. Pinecone/Weaviate users implement
the 2-method protocol; we never fork-block them.

### 3.4 Structured tasks and typed reports, with NL as a front-end

```python
report: DealRiskReport = await engine.run(
    PipelineEvaluationTask(stage="Commit", quarter="2026-Q3"),
)
```

- Every task is a pydantic model; every report is a pydantic model with `.to_dict()`,
  `.to_markdown()`, and stable field names for dashboards.
- `engine.run_analysis("Evaluate all open deals in Commit for Q3")` still exists, but it
  is explicitly a **router**: an LLM step that parses NL into a registered Task model,
  then calls `engine.run()`. NL is a UX layer over the typed API, never a parallel path.
- Structured outputs come from `pydantic-ai`'s `output_type` — no JSON-parsing regexes.

---

## 4. The GTM-specific pillars (the enterprise adoption gates)

### 4.1 Data freshness

The SDK takes a position: **agents must know how old their data is.**

- Every `Connector` exposes `sync_metadata() -> SyncMetadata` (`last_synced_at`, source
  system, lag estimate). Live-API connectors report `now`; warehouse connectors read it
  from sync tooling (Fivetran/Airbyte/dlt metadata tables — pluggable resolver).
- `FreshnessPolicy(max_staleness=timedelta(hours=6), on_violation="warn"|"fail")` is set
  per engine or per task. Violations either annotate the report or abort the run.
- Every `Report` carries `data_vintage: dict[role, datetime]` so a VP can see "this
  churn score used billing data synced 09:14 UTC."
- Recommended posture (documented, not enforced): warehouse-first reads via dbt models;
  `dlt` extra for teams that need us to do ingestion.

### 4.2 Explainability — Evidence, not verdicts

Findings are never bare scores:

```python
class Finding(BaseModel):
    subject: EntityRef                # deal/account/contact id + source system
    verdict: str                      # "at_risk"
    confidence: float
    evidence: list[Evidence]          # the load-bearing part
    data_vintage: dict[str, datetime]

class Evidence(BaseModel):
    kind: Literal["metric", "record", "retrieval", "llm_judgment"]
    summary: str                      # "No activity in 21 days vs 4-day stage median"
    source: SourceRef                 # query text / record URL / retrieved chunk id
```

`BaseAgent` enforces this: the structured output schema for every shipped agent requires
evidence, and the docs make it the norm for custom agents. `report.to_markdown()`
renders an evidence-backed brief a GTM engineer can paste in front of a VP.

### 4.3 Auditability and replay — the run ledger

Every `engine.run()` writes an append-only ledger (SQLAlchemy models + Alembic
migrations, stored in the customer's own Postgres):

- `runs`: task payload, agent + SDK versions, model id, config hash, prompt versions.
- `run_events`: every tool call with inputs/outputs (with field-level redaction hooks),
  every LLM call (request/response ids, token usage), every retrieval (chunk ids).
- `write_intents`: see §4.4 — proposed vs. applied vs. rejected.

This answers "why did the commission agent pay X last Tuesday" with
`revops-ai runs show <run_id>` (CLI) or the ledger API. **Replay** re-executes a run
from recorded tool outputs (`replay(run_id, mode="recorded")`) to reproduce the decision
deterministically, or against live data (`mode="live"`) to diff then-vs-now. The ledger
doubles as the OTel span source — one instrumentation, two consumers (traces +audit).

### 4.4 CRM write-back safety — agents propose, policies dispose

Agents **never** call write APIs directly. Write tools emit `WriteIntent` objects:

```python
class WriteIntent(BaseModel):
    connector_role: str               # "crm"
    operation: Literal["update", "create"]
    target: EntityRef
    changes: dict[str, FieldChange]   # field -> (old, new)
    justification: list[Evidence]
    idempotency_key: str
```

The engine routes intents through a `WritePolicy` pipeline:

1. **Dry-run is the default.** `engine.run(task)` produces intents; nothing is written
   until `engine = RevOpsEngine(write_mode="apply")` or per-run override.
2. **Field allowlists** per connector role (e.g., agents may update `next_step__c`, never
   `amount` or `stage`).
3. **Approval flows**: policy rules route intents to an `ApprovalQueue` (exposed via the
   FastAPI app + webhook to Slack); auto-approve below thresholds, human-gate above.
4. **Blast-radius limits**: max writes per run / per hour; a misbehaving agent trips a
   circuit breaker, not a 10,000-record HubSpot update.
5. **Idempotency**: applied intents record their key; replays and webhook retries can't
   double-write.

All four pillars live in plain modules (`safety/`, `audit/`, `freshness/`) — they are the
moat; the LLM plumbing underneath is rented.

---

## 5. Serving and event-driven deployment

- `create_api(engine)` returns a FastAPI app: `POST /tasks/{task_name}` (typed bodies
  from the Task registry), `GET /runs/{id}`, `GET /runs/{id}/events`, approval endpoints
  (`GET /intents/pending`, `POST /intents/{id}/approve`). OpenAPI schema for free.
- `WebhookListener` mounts per-source routes with **mandatory signature verification**
  (Stripe via the official SDK, HubSpot v3 signatures, Salesforce CDC) — unsigned events
  are rejected, not "best effort". Bindings map `(source, event_type) -> Task factory`,
  so webhooks enter the same typed/audited path as everything else:

```python
listener.bind("hubspot", "deal.closed_won",
              lambda evt: CommissionTask(deal_id=evt.object_id))
```

- Execution backends: `InProcessRunner` (default, fine for cron + low-volume webhooks)
  and `TemporalRunner` (extra) for durable, resumable runs — same `engine.run()` API.

---

## 6. Quality gate

- **Connector tests**: `respx`/`vcrpy` cassettes against recorded HubSpot/Stripe/SFDC
  responses; no live credentials in CI.
- **Agent tests**: `pydantic-ai`'s `TestModel`/`FunctionModel` for deterministic
  agent-loop tests; golden-report snapshot tests for shipped agents.
- **Safety tests**: property-style tests asserting no code path reaches a connector's
  write method without an approved `WriteIntent` (enforced by making write methods
  private to the intent applier).
- CI: `ruff` + `mypy --strict` on `src/`, pytest matrix on 3.10–3.13, `alembic` migration
  check, package-build + import-smoke-test of every extra combination.

---

## 7. Phased roadmap

**M0 — Skeleton (week 1–2):** pyproject + src layout, `RevOpsEngine`, registries,
`Connector` protocol, `BaseAgent` on pydantic-ai, `Task`/`Report` models, settings,
CI. *Exit: a custom agent runs a typed task against a fake connector with full typing.*

**M1 — Data layer (week 3–5):** HubSpot, Stripe, Warehouse connectors; pgvector adapter;
freshness metadata + policy; NL router over the task registry. *Exit: `PipelineVelocityAgent`
produces an evidence-backed `DealRiskReport` from real HubSpot + warehouse data.*

**M2 — Trust layer (week 6–8):** run ledger + Alembic, OTel spans (+ Langfuse exporter
extra), `WriteIntent` pipeline with dry-run default, field allowlists, approval queue,
recorded replay. *Exit: a commission run is fully reconstructable and a write-back demo
shows dry-run → approval → apply with idempotency.*

**M3 — Serving (week 9–10):** `create_api`, `WebhookListener` with signature
verification, CLI (`revops-ai runs show/replay`), Dockerfile + compose example,
`TemporalRunner` extra. *Exit: docker-compose up gives a webhook-driven, approvable,
audited commission pipeline.*

**M4 — Polish & publish (week 11–12):** second shipped agent (`ChurnPredictorAgent`),
Qdrant adapter, Salesforce connector, docs site (mkdocs-material) with a "Monday-morning
incident" runbook, PyPI release with trusted publishing + `pip-audit` in CI.

---

## 8. Risks and stances

- **Framework coupling** (the LangChain lesson): `pydantic-ai` is isolated inside
  `agents/base.py`; nothing in the public API imports it. If we must swap runtimes, the
  blast radius is one module.
- **Scope creep into ELT**: we read sync metadata; we do not become a sync engine. `dlt`
  extra is the escape hatch, clearly labeled optional.
- **Provider drift**: model ids are config strings with fallbacks, never hardcoded in
  agents; shipped agents are tested against ≥2 providers.
- **Ledger PII**: redaction hooks are in the event-write path from M2 day one — retrofitting
  redaction into an audit log is how compliance reviews fail.
