import pytest
from pydantic import ValidationError

from buc_factory.app.models import RunListResponse, RunRequest, RunResponse, RunSummary


def test_run_request_minimal():
    req = RunRequest(
        industry="insurance",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Power BI Desktop",
        duration_minutes=60,
    )
    assert req.deliverable_format == "PBIP"
    assert req.dimensions is None
    assert req.entities is None


def test_run_request_missing_required_field():
    with pytest.raises(ValidationError):
        RunRequest(
            company_context="ctx",
            location="Paris",
            language="French",
            role="Analyst",
            seniority="Mid",
            tool="Power BI Desktop",
            duration_minutes=60,
        )


def test_run_request_with_optional_fields():
    req = RunRequest(
        industry="banking",
        company_context="ctx",
        location="Paris",
        language="French",
        role="Analyst",
        seniority="Mid",
        tool="Tableau",
        duration_minutes=75,
        entities=["customers", "transactions"],
        dimensions={"region": ["EMEA", "APAC"]},
        deliverable_format="TWBX",
    )
    assert req.entities == ["customers", "transactions"]
    assert req.deliverable_format == "TWBX"


def test_run_response():
    resp = RunResponse(run_id="run_001", status="queued")
    assert resp.run_id == "run_001"
    assert resp.status == "queued"


def test_run_summary():
    s = RunSummary(run_id="run_002", status="done")
    assert s.run_id == "run_002"


def test_run_list_response():
    resp = RunListResponse(
        runs=[
            RunSummary(run_id="run_001", status="done"),
            RunSummary(run_id="run_002", status="running"),
        ]
    )
    assert len(resp.runs) == 2
    assert resp.runs[0].run_id == "run_001"
