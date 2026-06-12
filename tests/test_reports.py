"""Evidence and report rendering."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from revops_ai import EntityRef, Evidence, Finding, Report, SourceRef


def _finding() -> Finding:
    return Finding(
        subject=EntityRef(source_system="hubspot", entity_type="deal", entity_id="D-1"),
        verdict="at_risk",
        confidence=0.82,
        evidence=[
            Evidence(
                kind="metric",
                summary="No activity in 21 days vs 4-day stage median",
                source=SourceRef(kind="query", reference="SELECT ... FROM activities"),
            )
        ],
    )


def test_finding_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(
            subject=EntityRef(source_system="hubspot", entity_type="deal", entity_id="D-1"),
            verdict="at_risk",
            confidence=0.8,
            evidence=[],
        )


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate({**_finding().model_dump(), "confidence": 1.5})


def test_report_markdown_includes_vintage_and_evidence() -> None:
    class RiskReport(Report):
        findings: list[Finding]

    report = RiskReport(
        findings=[_finding()],
        data_vintage={"crm": datetime(2026, 6, 12, 9, 14, tzinfo=timezone.utc)},
    )
    md = report.to_markdown()
    assert "synced 2026-06-12T09:14:00+00:00" in md
    assert "at_risk (82%)" in md
    assert "No activity in 21 days" in md
