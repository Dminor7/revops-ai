# revops-ai

An extensible SDK for building **auditable, write-safe RevOps AI agents** — built for GTM
AI Engineers who need agent decisions they can defend in front of a VP, not just demo.

> Status: **alpha (M0)**. Core engine, typed tasks/reports, agent extension contract, and
> freshness policies are functional. Connectors, the audit ledger, write-intents, and the
> serving layer land in M1–M3 — see [PLAN.md](PLAN.md).

## Design principles

- **Inversion of control.** You inject connectors (with your credentials), an LLM
  configuration, and policies into a `RevOpsEngine`. The SDK ships the orchestration;
  your environment owns the secrets and the data layer.
- **Typed everything.** Tasks in, reports out — all pydantic models. Natural language is
  a router on top of the typed API, never a parallel path.
- **Evidence, not verdicts.** Findings structurally require evidence, and every report
  carries the vintage of the data it was computed from.
- **Fail at assembly, not mid-run.** Missing connectors, capabilities, or model config
  raise when you register an agent — not on a Monday morning.
- **Rent the plumbing.** The LLM loop is [pydantic-ai](https://ai.pydantic.dev) (isolated
  in one module); validation is pydantic; serving will be FastAPI. The SDK's own code is
  the GTM domain layer: guardrails, audit, freshness, evidence.

## Quick start

```python
from revops_ai import (
    BaseAgent, Capability, LLMConfig, Report, RevOpsEngine, RunContext, Task, Tool,
)

# 1. Define (or import) a connector — anything satisfying the Connector protocol.
class WarehouseConnector:
    capabilities = frozenset({Capability.READ})
    def sync_metadata(self): ...
    async def health_check(self): ...

# 2. Declare a typed task and report.
class CommissionTask(Task):
    deal_id: str

class CommissionReport(Report):
    deal_id: str
    status: str
    payout_tier: str

# 3. Write your agent. Tools are *declared*, not constructed — the engine
#    binds them to your connectors at registration (multi-tenant safe).
class SQLQueryTool(Tool):
    async def query_one(self, sql: str, **params): ...

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
        return CommissionReport(deal_id=task.deal_id, status="approved", payout_tier=tier)

# 4. Assemble the engine in *your* environment and run.
engine = RevOpsEngine(llm=LLMConfig(model="openai:gpt-4o"))
engine.register_connector("warehouse", WarehouseConnector())
engine.register_agent(CommissionAgent())

report = await engine.run(CommissionTask(deal_id="D-1"))
print(report.payout_tier, report.data_vintage)
```

LLM-backed agents subclass `LLMAgent` instead and get a typed pydantic-ai loop with the
engine's configured model (with fallback support); deterministic agents need no model at
all.

## Freshness policies

```python
from datetime import timedelta
from revops_ai import FreshnessPolicy

engine = RevOpsEngine(
    freshness=FreshnessPolicy(max_staleness=timedelta(hours=6), on_violation="fail"),
)
```

Every connector reports `sync_metadata()`; runs against stale data either warn or abort,
and every report's `data_vintage` records exactly how fresh each source was.

## Development

```bash
uv venv && uv pip install -e . --group dev
pytest
ruff check src tests && mypy
```

## Roadmap

See [PLAN.md](PLAN.md) for the full architecture and the M0–M4 roadmap (connectors,
audit ledger with replay, write-intent pipeline with approvals, FastAPI serving, webhook
listeners).
