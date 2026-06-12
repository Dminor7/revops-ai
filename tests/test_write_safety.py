"""Write-intent pipeline: dry-run default, allowlists, approvals, idempotency."""

from __future__ import annotations

import pytest

from revops_ai import (
    BaseAgent,
    Capability,
    CapabilityError,
    EntityRef,
    Evidence,
    FieldChange,
    Report,
    RevOpsEngine,
    RunContext,
    SourceRef,
    SyncMetadata,
    Task,
    Tool,
    WriteIntent,
    WriteIntentStatus,
    WritePolicy,
)


class FakeCRMConnector:
    """Writable CRM double that records applied writes."""

    capabilities = frozenset({Capability.READ, Capability.WRITE})

    def __init__(self) -> None:
        self.applied: list[WriteIntent] = []

    def sync_metadata(self) -> SyncMetadata:
        return SyncMetadata(source_system="fake_crm")

    async def health_check(self) -> bool:
        return True

    async def apply_write(self, intent: WriteIntent) -> dict[str, str]:
        self.applied.append(intent)
        return {"id": intent.target.entity_id}


class CRMWriteMarker(Tool):
    required_capabilities = frozenset({Capability.WRITE})


class NudgeTask(Task):
    deal_id: str
    next_step: str = "Schedule exec sync"


class NudgeReport(Report):
    deal_id: str


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            kind="metric",
            summary="No activity in 21 days",
            source=SourceRef(kind="query", reference="SELECT ..."),
        )
    ]


class NudgeAgent(BaseAgent[NudgeTask, NudgeReport]):
    """Proposes a CRM field update; never writes directly."""

    name = "nudge_agent"
    description = "Suggests a next step on stalled deals."
    requires = {"crm": CRMWriteMarker}
    task_type = NudgeTask
    report_type = NudgeReport

    async def process_task(self, ctx: RunContext, task: NudgeTask) -> NudgeReport:
        ctx.propose_write(
            WriteIntent(
                connector_role="crm",
                operation="update",
                target=EntityRef(
                    source_system="fake_crm", entity_type="deal", entity_id=task.deal_id
                ),
                changes={"next_step": FieldChange(old=None, new=task.next_step)},
                justification=_evidence(),
            )
        )
        return NudgeReport(deal_id=task.deal_id)


def _engine(crm: FakeCRMConnector, **kwargs: object) -> RevOpsEngine:
    engine = RevOpsEngine(**kwargs)  # type: ignore[arg-type]
    engine.register_connector("crm", crm)
    engine.register_agent(NudgeAgent())
    return engine


ALLOW_NEXT_STEP = WritePolicy(field_allowlists={"crm": {"next_step"}})


async def test_dry_run_is_the_default_nothing_is_written() -> None:
    crm = FakeCRMConnector()
    engine = _engine(crm, write_policy=ALLOW_NEXT_STEP)

    report = await engine.run(NudgeTask(deal_id="D-1"))

    assert crm.applied == []
    assert len(report.proposed_writes) == 1
    assert report.proposed_writes[0].status is WriteIntentStatus.PROPOSED


async def test_no_policy_means_intents_are_rejected() -> None:
    crm = FakeCRMConnector()
    engine = _engine(crm)  # no WritePolicy at all

    report = await engine.run(NudgeTask(deal_id="D-1"))

    assert report.proposed_writes[0].status is WriteIntentStatus.REJECTED
    assert "WritePolicy" in (report.proposed_writes[0].status_reason or "")


async def test_field_allowlist_rejects_undeclared_fields() -> None:
    crm = FakeCRMConnector()
    engine = _engine(
        crm,
        write_policy=WritePolicy(field_allowlists={"crm": {"notes"}}),
        write_mode="apply",
    )

    report = await engine.run(NudgeTask(deal_id="D-1"))

    assert crm.applied == []
    intent = report.proposed_writes[0]
    assert intent.status is WriteIntentStatus.REJECTED
    assert "next_step" in (intent.status_reason or "")


async def test_apply_mode_routes_to_approval_queue_then_applies() -> None:
    crm = FakeCRMConnector()
    engine = _engine(crm, write_policy=ALLOW_NEXT_STEP, write_mode="apply")

    report = await engine.run(NudgeTask(deal_id="D-1"))
    intent = report.proposed_writes[0]
    assert intent.status is WriteIntentStatus.PENDING_APPROVAL
    assert crm.applied == []

    pending = engine.pending_intents()
    assert [i.intent_id for i in pending] == [intent.intent_id]

    applied = await engine.approve_intent(intent.intent_id)
    assert applied.status is WriteIntentStatus.APPLIED
    assert len(crm.applied) == 1
    assert engine.pending_intents() == []


async def test_auto_approve_applies_immediately() -> None:
    crm = FakeCRMConnector()
    policy = WritePolicy(
        field_allowlists={"crm": {"next_step"}},
        auto_approve=lambda intent: True,
    )
    engine = _engine(crm, write_policy=policy, write_mode="apply")

    report = await engine.run(NudgeTask(deal_id="D-1"))

    assert report.proposed_writes[0].status is WriteIntentStatus.APPLIED
    assert len(crm.applied) == 1


async def test_idempotency_same_key_is_not_applied_twice() -> None:
    crm = FakeCRMConnector()
    policy = WritePolicy(field_allowlists={"crm": {"next_step"}}, auto_approve=lambda i: True)
    engine = _engine(crm, write_policy=policy, write_mode="apply")

    await engine.run(NudgeTask(deal_id="D-1"))
    await engine.run(NudgeTask(deal_id="D-1"))  # identical change -> same idempotency key

    assert len(crm.applied) == 1


async def test_blast_radius_limit_rejects_excess_intents() -> None:
    class FanoutTask(Task):
        deal_ids: tuple[str, ...]

    class FanoutReport(Report):
        count: int

    class FanoutAgent(BaseAgent[FanoutTask, FanoutReport]):
        name = "fanout"
        requires = {"crm": CRMWriteMarker}
        task_type = FanoutTask
        report_type = FanoutReport

        async def process_task(self, ctx: RunContext, task: FanoutTask) -> FanoutReport:
            for deal_id in task.deal_ids:
                ctx.propose_write(
                    WriteIntent(
                        connector_role="crm",
                        operation="update",
                        target=EntityRef(
                            source_system="fake_crm", entity_type="deal", entity_id=deal_id
                        ),
                        changes={"next_step": FieldChange(new="ping")},
                        justification=_evidence(),
                    )
                )
            return FanoutReport(count=len(task.deal_ids))

    crm = FakeCRMConnector()
    engine = RevOpsEngine(
        write_policy=WritePolicy(
            field_allowlists={"crm": {"next_step"}},
            max_writes_per_run=2,
            auto_approve=lambda i: True,
        ),
        write_mode="apply",
    )
    engine.register_connector("crm", crm)
    engine.register_agent(FanoutAgent())

    report = await engine.run(FanoutTask(deal_ids=("D-1", "D-2", "D-3")))

    statuses = [i.status for i in report.proposed_writes]
    assert statuses.count(WriteIntentStatus.APPLIED) == 2
    assert statuses.count(WriteIntentStatus.REJECTED) == 1
    assert len(crm.applied) == 2


async def test_propose_write_requires_write_capability() -> None:
    """An agent whose role was granted read-only cannot even propose."""

    class ReadMarker(Tool):
        required_capabilities = frozenset({Capability.READ})

    class SneakyAgent(NudgeAgent):
        name = "sneaky"
        requires = {"crm": ReadMarker}

    crm = FakeCRMConnector()
    engine = RevOpsEngine(write_policy=ALLOW_NEXT_STEP, write_mode="apply")
    # Narrow the writable connector to read-only for this engine.
    engine.register_connector("crm", crm, capabilities={Capability.READ})
    engine.register_agent(SneakyAgent())

    with pytest.raises(CapabilityError):
        await engine.run(NudgeTask(deal_id="D-1"))
