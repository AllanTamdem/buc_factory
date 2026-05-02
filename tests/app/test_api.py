import io
import json
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import buc_factory.app.api as api_module
from buc_factory.app.api import app, _resolve_run
from fastapi import HTTPException


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_module, "_run_status", {})
    with patch("buc_factory.app.api.setup_mlflow"):
        with TestClient(app) as c:
            yield c, tmp_path


_MINIMAL_PAYLOAD = {
    "industry": "insurance",
    "company_context": "A test company",
    "location": "Paris",
    "language": "French",
    "role": "Analyst",
    "seniority": "Mid",
    "tool": "Power BI Desktop",
    "duration_minutes": 60,
}


# ── GET /runs ─────────────────────────────────────────────────────


def test_list_runs_empty_data_dir(client):
    c, _ = client
    resp = c.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_list_runs_shows_disk_runs(client):
    c, data_dir = client
    run = data_dir / "run_001"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"task_index": 8}))
    resp = c.get("/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_001"
    assert runs[0]["status"] == "complete"


def test_list_runs_partial_status(client):
    c, data_dir = client
    run = data_dir / "run_001"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"task_index": 3}))
    resp = c.get("/runs")
    runs = resp.json()["runs"]
    assert "partial" in runs[0]["status"]


# ── POST /runs ────────────────────────────────────────────────────


def test_create_run_returns_202(client):
    c, _ = client
    with patch("buc_factory.app.api._execute_run"):
        resp = c.post("/runs", json=_MINIMAL_PAYLOAD)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["run_id"].startswith("run_")


def test_create_run_creates_directory(client):
    c, data_dir = client
    with patch("buc_factory.app.api._execute_run"):
        resp = c.post("/runs", json=_MINIMAL_PAYLOAD)
    run_id = resp.json()["run_id"]
    assert (data_dir / run_id).is_dir()


def test_create_run_increments_id(client):
    c, data_dir = client
    with patch("buc_factory.app.api._execute_run"):
        r1 = c.post("/runs", json=_MINIMAL_PAYLOAD).json()["run_id"]
        r2 = c.post("/runs", json=_MINIMAL_PAYLOAD).json()["run_id"]
    assert r1 != r2


def test_create_run_invalid_payload(client):
    c, _ = client
    resp = c.post("/runs", json={"industry": "only-one-field"})
    assert resp.status_code == 422


# ── GET /runs/{run_id} ────────────────────────────────────────────


def test_get_run_existing(client):
    c, data_dir = client
    (data_dir / "run_001").mkdir()
    resp = c.get("/runs/run_001")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run_001"


def test_get_run_not_found(client):
    c, _ = client
    resp = c.get("/runs/run_999")
    assert resp.status_code == 404


def test_get_run_in_memory_status_takes_priority(client):
    c, data_dir = client
    (data_dir / "run_001").mkdir()
    api_module._run_status["run_001"] = "running"
    resp = c.get("/runs/run_001")
    assert resp.json()["status"] == "running"


# ── GET /runs/{run_id}/recruiter.zip ─────────────────────────────


def test_download_recruiter_zip(client):
    c, data_dir = client
    run = data_dir / "run_001"
    (run / "brief").mkdir(parents=True)
    (run / "brief" / "brief.md").write_text("brief content")
    (run / "solution").mkdir()
    (run / "solution" / "solution.md").write_text("solution content")
    resp = c.get("/runs/run_001/recruiter.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert "brief/brief.md" in names
    assert "solution/solution.md" in names


def test_download_recruiter_zip_not_found(client):
    c, _ = client
    resp = c.get("/runs/run_999/recruiter.zip")
    assert resp.status_code == 404


# ── GET /runs/{run_id}/candidate.zip ─────────────────────────────


def test_download_candidate_zip_excludes_generate_data(client):
    c, data_dir = client
    run = data_dir / "run_001"
    (run / "brief").mkdir(parents=True)
    (run / "brief" / "brief.md").write_text("brief")
    (run / "starter").mkdir()
    (run / "starter" / "project.pbip").write_text("project")
    (run / "starter" / "generate_data.py").write_text("# script")
    resp = c.get("/runs/run_001/candidate.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert "starter/generate_data.py" not in names
    assert "starter/project.pbip" in names
    assert "brief/brief.md" in names


# ── path traversal protection ─────────────────────────────────────


def test_resolve_run_path_traversal_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        _resolve_run("../sensitive")
    assert exc.value.status_code == 400


def test_resolve_run_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        _resolve_run("run_999")
    assert exc.value.status_code == 404
