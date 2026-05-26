import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import buc_factory.app.api as api_module
from buc_factory.app.api import app

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


def _make_mlflow_run(run_id="abc123", status="FINISHED", api_run_id="run_001"):
    run = MagicMock()
    run.info.run_id = run_id
    run.info.status = status
    run.data.tags = {"api_run_id": api_run_id}
    run.data.params = {
        "industry": "insurance",
        "role": "Analyst",
        "seniority": "Mid",
        "tool": "Power BI Desktop",
        "language": "French",
        "location": "Paris",
        "duration_minutes": "60",
        "deliverable_format": "PBIP",
    }
    return run


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(api_module, "_run_status", {})
    monkeypatch.setattr(api_module, "_mlflow_run_ids", {})
    monkeypatch.setattr(api_module, "_run_counter", 0)


@pytest.fixture
def client():
    with (
        patch("buc_factory.app.api.setup_mlflow"),
        patch("buc_factory.app.api._init_run_counter"),
        patch("buc_factory.app.api._init_sim_counter"),
        TestClient(app) as c,
    ):
        yield c


# ── GET /runs ─────────────────────────────────────────────────────


def test_list_runs_empty(client):
    with patch("buc_factory.app.api.mlflow") as mock_mlflow:
        mock_mlflow.MlflowClient.return_value.get_experiment_by_name.return_value = None
        resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_list_runs_returns_mlflow_runs(client):
    mlflow_run = _make_mlflow_run()
    experiment = MagicMock()
    experiment.experiment_id = "1"

    with (
        patch("buc_factory.app.api.mlflow") as mock_mlflow,
        patch("buc_factory.app.api._fetch_scenario", return_value=None),
    ):
        mock_mlflow.MlflowClient.return_value.get_experiment_by_name.return_value = experiment
        mock_mlflow.MlflowClient.return_value.search_runs.return_value = [mlflow_run]
        resp = client.get("/runs")

    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_001"
    assert runs[0]["status"] == "done"
    assert runs[0]["mlflow_run_id"] == "abc123"
    assert runs[0]["parameters"]["industry"] == "insurance"


def test_list_runs_includes_queued_not_yet_in_mlflow(client, monkeypatch):
    monkeypatch.setattr(api_module, "_run_status", {"run_001": "queued"})
    with patch("buc_factory.app.api.mlflow") as mock_mlflow:
        mock_mlflow.MlflowClient.return_value.get_experiment_by_name.return_value = None
        resp = client.get("/runs")
    runs = resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "queued"
    assert runs[0]["mlflow_run_id"] is None


# ── POST /runs ────────────────────────────────────────────────────


def test_create_run_returns_202(client):
    with patch("buc_factory.app.api._execute_run"):
        resp = client.post("/runs", json=_MINIMAL_PAYLOAD)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["run_id"].startswith("run_")


def test_create_run_increments_id(client):
    with patch("buc_factory.app.api._execute_run"):
        r1 = client.post("/runs", json=_MINIMAL_PAYLOAD).json()["run_id"]
        r2 = client.post("/runs", json=_MINIMAL_PAYLOAD).json()["run_id"]
    assert r1 != r2


def test_create_run_invalid_payload(client):
    resp = client.post("/runs", json={"industry": "only-one-field"})
    assert resp.status_code == 422


# ── agent failure path ────────────────────────────────────────────


def test_failed_run_sets_mlflow_status_to_failed():
    """Agent exhausts retries → MLflow run must be terminated as FAILED, not FINISHED."""
    failed_state = {"failed": True, "current_task": "generate_data_script"}

    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "mlflow-uuid-123"

    with (
        patch("buc_factory.app.api.mlflow") as mock_mlflow,
        patch("buc_factory.app.api.build_graph") as mock_build_graph,
        patch("buc_factory.app.api.log_config"),
        patch("buc_factory.app.api.log_tasks_summary"),
        patch("buc_factory.app.api.log_output_artifacts"),
        patch("buc_factory.app.api.reset_run"),
    ):
        mock_mlflow.start_run.return_value.__enter__ = lambda _: mock_active_run
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.set_experiment = MagicMock()
        mock_build_graph.return_value.invoke.return_value = failed_state

        from buc_factory.agent.entity import DomainConfig
        from buc_factory.app.api import _execute_run

        cfg = DomainConfig(
            industry="test",
            company_context="ctx",
            location="Paris",
            language="French",
            role="Analyst",
            seniority="Mid",
            tool="Power BI Desktop",
            duration_minutes=60,
            deliverable_format="PBIP",
        )
        _execute_run("run_001", cfg)

    # MLflow run must be terminated as FAILED
    mock_mlflow.MlflowClient.return_value.set_terminated.assert_called_once_with(
        "mlflow-uuid-123", status="FAILED"
    )
    # In-memory status must also be "failed"
    assert api_module._run_status["run_001"] == "failed"


def test_failed_run_status_survives_restart(monkeypatch):
    """After a server restart, a run with failure_task tag must show as 'failed', not 'done'."""
    monkeypatch.setattr(api_module, "_run_status", {})  # simulate cleared in-memory state

    mlflow_run = _make_mlflow_run(status="FAILED")
    mlflow_run.data.tags["failure_task"] = "generate_data_script"

    with (
        patch("buc_factory.app.api._find_mlflow_run", return_value=mlflow_run),
        patch("buc_factory.app.api._fetch_scenario", return_value=None),
        patch("buc_factory.app.api.setup_mlflow"),
        patch("buc_factory.app.api._init_run_counter"),
        patch("buc_factory.app.api._init_sim_counter"),
        TestClient(app) as c,
    ):
        resp = c.get("/runs/run_001")

    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"


# ── GET /runs/{run_id} ────────────────────────────────────────────


def test_get_run_not_found(client):
    with patch("buc_factory.app.api._find_mlflow_run", return_value=None):
        resp = client.get("/runs/run_999")
    assert resp.status_code == 404


def test_get_run_from_mlflow(client):
    mlflow_run = _make_mlflow_run()
    with (
        patch("buc_factory.app.api._find_mlflow_run", return_value=mlflow_run),
        patch("buc_factory.app.api._fetch_scenario", return_value=None),
    ):
        resp = client.get("/runs/run_001")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run_001"
    assert resp.json()["status"] == "done"
    assert resp.json()["mlflow_run_id"] == "abc123"


def test_get_run_in_memory_status_takes_priority(client, monkeypatch):
    monkeypatch.setattr(api_module, "_run_status", {"run_001": "running"})
    mlflow_run = _make_mlflow_run(status="RUNNING")
    with (
        patch("buc_factory.app.api._find_mlflow_run", return_value=mlflow_run),
        patch("buc_factory.app.api._fetch_scenario", return_value=None),
    ):
        resp = client.get("/runs/run_001")
    assert resp.json()["status"] == "running"


# ── GET /runs/{run_id}/recruiter.zip ─────────────────────────────


def test_download_recruiter_zip(client, tmp_path):
    (tmp_path / "brief").mkdir()
    (tmp_path / "brief" / "brief.md").write_text("brief content")
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solution.md").write_text("solution content")

    with (
        patch("buc_factory.app.api._resolve_mlflow_run", return_value="abc123"),
        patch("mlflow.artifacts.download_artifacts", return_value=str(tmp_path)),
    ):
        resp = client.get("/runs/run_001/recruiter.zip")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
    assert "brief/brief.md" in names
    assert "solution/solution.md" in names


def test_download_recruiter_zip_not_found(client):
    with patch("buc_factory.app.api._resolve_mlflow_run") as mock_resolve:
        mock_resolve.side_effect = HTTPException(status_code=404, detail="not found")
        resp = client.get("/runs/run_999/recruiter.zip")
    assert resp.status_code == 404


# ── GET /runs/{run_id}/candidate.zip ─────────────────────────────


def test_download_candidate_zip_excludes_generate_data(client, tmp_path):
    (tmp_path / "brief").mkdir()
    (tmp_path / "brief" / "brief.md").write_text("brief")
    (tmp_path / "starter").mkdir()
    (tmp_path / "starter" / "project.pbip").write_text("project")
    (tmp_path / "starter" / "generate_data.py").write_text("# script")

    with (
        patch("buc_factory.app.api._resolve_mlflow_run", return_value="abc123"),
        patch("mlflow.artifacts.download_artifacts", return_value=str(tmp_path)),
    ):
        resp = client.get("/runs/run_001/candidate.zip")

    assert resp.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
    assert "starter/generate_data.py" not in names
    assert "starter/project.pbip" in names
    assert "brief/brief.md" in names


# ── GET /runs/search ──────────────────────────────────────────────


def test_search_runs_returns_results(client):
    mlflow_run = _make_mlflow_run()
    with (
        patch("buc_factory.app.api._search_runs", return_value=[("run_001", 0.92)]),
        patch("buc_factory.app.api._find_mlflow_run", return_value=mlflow_run),
        patch("buc_factory.app.api._fetch_scenario", return_value=None),
    ):
        resp = client.get("/runs/search?q=data+analyst+insurance")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["run_id"] == "run_001"
    assert results[0]["score"] == pytest.approx(0.92)
    assert results[0]["status"] == "done"


def test_search_runs_empty_when_no_hits(client):
    with patch("buc_factory.app.api._search_runs", return_value=[]):
        resp = client.get("/runs/search?q=nothing")
    assert resp.status_code == 200
    assert resp.json() == []


def test_search_runs_missing_query_param(client):
    resp = client.get("/runs/search")
    assert resp.status_code == 422


def test_search_runs_limit_param_passed_through(client):
    with patch("buc_factory.app.api._search_runs", return_value=[]) as mock_search:
        client.get("/runs/search?q=test&limit=3")
    mock_search.assert_called_once_with("test", k=3)


def test_search_runs_unknown_mlflow_run(client):
    with (
        patch("buc_factory.app.api._search_runs", return_value=[("run_999", 0.75)]),
        patch("buc_factory.app.api._find_mlflow_run", return_value=None),
    ):
        resp = client.get("/runs/search?q=test")
    assert resp.status_code == 200
    results = resp.json()
    assert results[0]["run_id"] == "run_999"
    assert results[0]["mlflow_run_id"] is None
    assert results[0]["status"] == "unknown"
