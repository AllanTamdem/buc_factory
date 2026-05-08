"""
FastAPI server for BUC Factory: run the agent and download assessment packages.

Endpoints:
    GET  /runs                         – list all runs from MLflow (params + scenario + status)
    POST /runs                         – submit a new agent run (async, 202)
    GET  /runs/{run_id}                – run details from MLflow (params, scenario, status)
    GET  /runs/{run_id}/recruiter.zip  – brief/ + solution/ (from MLflow artifacts)
    GET  /runs/{run_id}/candidate.zip  – brief/ + starter/ (from MLflow artifacts,
                                         minus generate_data.py)

Usage:
    uvicorn buc_factory.app.api:app --reload
    python -m buc_factory.app.api
"""

import importlib.resources
import io
import json
import logging
import logging.handlers
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent.entity import PLAN, BucState, DomainConfig
from ..agent.graph import (
    _CLAUDE_TO_OPENAI,
    _TASK_MODEL,
    _TASK_OPENAI_OVERRIDE,
    build_graph,
)
from ..tracking import (
    evaluate_outputs,
    get_run_id,
    log_config,
    log_output_artifacts,
    log_tasks_summary,
    reset_run,
    set_run_id,
    setup_mlflow,
)
from .models import RunListResponse, RunParameters, RunRequest, RunResponse, RunSummary

load_dotenv()

_EXPERIMENT = "buc-factory"
_MODEL_TAG = (
    " / ".join(sorted(set(_TASK_MODEL.values())))
    + " — fallback: "
    + " / ".join(sorted(set(_CLAUDE_TO_OPENAI.values()) | set(_TASK_OPENAI_OVERRIDE.values())))
)


class _RunIdFilter(logging.Filter):
    """Injects the current run_id into every log record for this thread."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = get_run_id()
        return True


Path("log").mkdir(exist_ok=True)
_fmt = logging.Formatter("%(asctime)s [%(run_id)s] %(levelname)s - %(message)s")
_run_filter = _RunIdFilter()
_handlers: list[logging.Handler] = [
    logging.handlers.RotatingFileHandler(
        "log/agent.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,  # keep .1 … .5  →  ~50 MB total
        encoding="utf-8",
    ),
    logging.StreamHandler(),
]
for _h in _handlers:
    _h.setFormatter(_fmt)
    _h.addFilter(_run_filter)
logging.basicConfig(level=logging.INFO, handlers=_handlers)

LOGGER = logging.getLogger(__name__)

_run_status: dict[str, str] = {}  # run_id -> "queued"|"running"|"done"|"failed"
_mlflow_run_ids: dict[str, str] = {}  # run_id -> MLflow run UUID (populated once run starts)
_run_counter = 0
_id_lock = Lock()

_MLFLOW_STATUS_MAP = {
    "RUNNING": "running",
    "FINISHED": "done",
    "FAILED": "failed",
    "KILLED": "failed",
}


# ── app lifecycle ──────────────────────────────────────────────────


def _init_run_counter() -> None:
    """Seed _run_counter from the highest api_run_id already registered in MLflow."""
    global _run_counter
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_EXPERIMENT)
    if experiment is None:
        return
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.api_run_id != ''",
        max_results=1000,
    )
    nums = [
        int(r.data.tags["api_run_id"][4:])
        for r in runs
        if r.data.tags.get("api_run_id", "").startswith("run_")
        and r.data.tags["api_run_id"][4:].isdigit()
    ]
    if nums:
        _run_counter = max(nums)


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    setup_mlflow()
    _init_run_counter()
    yield


_STATIC_DIR = importlib.resources.files("buc_factory").joinpath("static")

app = FastAPI(
    title="BUC Factory",
    description="Generate and download BI recruitment assessment packages.",
    lifespan=_lifespan,
    docs_url=None,  # replaced by custom endpoint below
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="BUC Factory",
        swagger_favicon_url="/static/logo_mark.svg",
    )


# ── internal helpers ───────────────────────────────────────────────


def _next_run_id() -> str:
    global _run_counter
    with _id_lock:
        _run_counter += 1
        return f"run_{_run_counter:03d}"


def _effective_status(run_id: str, mlflow_status: str | None) -> str:
    """In-memory status is authoritative for the current session; MLflow is the fallback."""
    if run_id in _run_status:
        return _run_status[run_id]
    return _MLFLOW_STATUS_MAP.get(mlflow_status or "", "unknown")


def _extract_parameters(run: Any) -> RunParameters:
    p = run.data.params
    return RunParameters(
        industry=p.get("industry"),
        role=p.get("role"),
        seniority=p.get("seniority"),
        tool=p.get("tool"),
        language=p.get("language"),
        location=p.get("location"),
        duration_minutes=int(p["duration_minutes"]) if "duration_minutes" in p else None,
        deliverable_format=p.get("deliverable_format"),
    )


def _fetch_scenario(mlflow_run_id: str) -> dict | None:
    """Download and parse outputs/scenario.json from MLflow artifacts."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path="outputs/scenario.json",
                dst_path=tmpdir,
            )
            return json.loads(Path(local).read_text())
    except Exception:
        return None


def _find_mlflow_run(run_id: str) -> Any | None:
    """Return the MLflow Run for an api_run_id, or None."""
    # Fast path: current session cached the UUID
    mlflow_uuid = _mlflow_run_ids.get(run_id)
    if mlflow_uuid:
        try:
            return mlflow.MlflowClient().get_run(mlflow_uuid)
        except Exception:
            pass

    # Fallback: tag search (works across server restarts)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_EXPERIMENT)
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.api_run_id = '{run_id}'",
        max_results=1,
    )
    return runs[0] if runs else None


def _resolve_mlflow_run(run_id: str) -> str:
    """Return the MLflow run UUID for artifact download, or raise an appropriate HTTP error."""
    if _run_status.get(run_id) in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Run '{run_id}' artifacts not yet available (status: {_run_status[run_id]}).",
        )
    mlflow_run = _find_mlflow_run(run_id)
    if mlflow_run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return mlflow_run.info.run_id


def _build_zip_from_mlflow(
    mlflow_run_id: str,
    folders: list[tuple[str, str]],
    exclude: set[str] | None = None,
) -> io.BytesIO:
    """Download the outputs artifact tree from MLflow and pack selected folders into a zip."""
    excluded = exclude or set()
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=mlflow_run_id,
            artifact_path="outputs",
            dst_path=tmpdir,
        )
        run_path = Path(local_path)
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for subdir, prefix in folders:
                src = run_path / subdir
                if not src.is_dir():
                    continue
                for file_path in sorted(src.rglob("*")):
                    if not file_path.is_file():
                        continue
                    arc_name = prefix + "/" + file_path.relative_to(src).as_posix()
                    if arc_name in excluded:
                        continue
                    zf.write(file_path, arc_name)
    buf.seek(0)
    return buf


def _load_initial_state(output_dir: Path) -> BucState:
    return BucState(
        messages=[],
        output_dir=str(output_dir),
        task_index=0,
        current_task=PLAN[0],
        tasks_remaining=PLAN,
        retry_count=0,
        bootstrapped_dimensions=None,
        bootstrapped_entities=None,
        scenario=None,
        rolled=None,
        validation_error=None,
        failed=False,
    )


def _execute_run(run_id: str, cfg: DomainConfig) -> None:
    """Full agent pipeline — runs synchronously in a background thread."""
    set_run_id(run_id)
    try:
        _execute_run_inner(run_id, cfg)
    finally:
        set_run_id("-")


def _execute_run_inner(run_id: str, cfg: DomainConfig) -> None:
    mlflow.set_experiment(_EXPERIMENT)  # thread-local; must be set in each background thread
    _run_status[run_id] = "running"
    reset_run()
    run_name = f"{cfg.industry}__{cfg.role}".replace(" ", "_").lower()
    LOGGER.info(f"agent: industry={cfg.industry!r}, role={cfg.role!r}, tool={cfg.tool!r}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            LOGGER.info(f"  temp output: {output_dir}")
            with mlflow.start_run(run_name=run_name) as active_run:
                _mlflow_run_ids[run_id] = active_run.info.run_id
                log_config(cfg)
                mlflow.set_tags(
                    {
                        "api_run_id": run_id,
                        "model": _MODEL_TAG,
                        "task_count": len(PLAN),
                    }
                )
                initial_state = _load_initial_state(output_dir)
                graph = build_graph(output_dir, cfg)

                t0 = time.perf_counter()
                final_state: BucState = graph.invoke(initial_state)
                total = time.perf_counter() - t0

                m, s = divmod(total, 60)
                fmt_total = f"{int(m)}m {s:.1f}s"

                failed = bool(final_state.get("failed"))

                log_tasks_summary(wall_clock_s=total)
                log_output_artifacts(output_dir)

                if not failed:
                    evaluate_outputs(output_dir)
                    LOGGER.info(f"\n✓ [{run_id}] all sub-tasks complete — total {fmt_total}")
                else:
                    mlflow.set_tag("failure_task", final_state.get("current_task", "unknown"))
                    LOGGER.error(f"\n✗ [{run_id}] agent failed after {fmt_total}")

        if failed:
            mlflow.MlflowClient().set_terminated(_mlflow_run_ids[run_id], status="FAILED")
        _run_status[run_id] = "failed" if failed else "done"

    except ValueError as exc:
        LOGGER.warning(f"run {run_id}: {exc}")
        _run_status[run_id] = "done"
    except Exception as exc:
        LOGGER.error(f"run {run_id} failed: {exc}", exc_info=True)
        _run_status[run_id] = "failed"


# ── endpoints ──────────────────────────────────────────────────────


@app.get("/runs", response_model=RunListResponse)
def list_runs() -> RunListResponse:
    """List all runs from MLflow with parameters and scenario, merged with in-memory status."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_EXPERIMENT)

    mlflow_by_run_id: dict[str, Any] = {}
    if experiment is not None:
        for mlflow_run in client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time ASC"],
            max_results=1000,
        ):
            api_run_id = mlflow_run.data.tags.get("api_run_id")
            if api_run_id:
                mlflow_by_run_id[api_run_id] = mlflow_run

    # Include queued runs not yet registered in MLflow
    all_run_ids = sorted(set(mlflow_by_run_id.keys()) | set(_run_status.keys()))

    def _fetch_for(rid: str) -> tuple[str, dict | None]:
        mlflow_run = mlflow_by_run_id.get(rid)
        return rid, _fetch_scenario(mlflow_run.info.run_id) if mlflow_run else None

    scenarios: dict[str, dict | None] = {}
    if all_run_ids:
        with ThreadPoolExecutor(max_workers=min(8, len(all_run_ids))) as pool:
            scenarios = dict(pool.map(_fetch_for, all_run_ids))

    runs = []
    for rid in all_run_ids:
        mlflow_run = mlflow_by_run_id.get(rid)
        runs.append(
            RunSummary(
                run_id=rid,
                status=_effective_status(rid, mlflow_run.info.status if mlflow_run else None),
                mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
                parameters=_extract_parameters(mlflow_run) if mlflow_run else None,
                scenario=scenarios.get(rid),
            )
        )

    return RunListResponse(runs=runs)


@app.post("/runs", status_code=202, response_model=RunResponse)
def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    """Submit a new agent run. Returns immediately; agent executes in the background."""
    run_id = _next_run_id()
    cfg = DomainConfig(**request.model_dump())
    _run_status[run_id] = "queued"
    background_tasks.add_task(_execute_run, run_id, cfg)
    return RunResponse(run_id=run_id, status="queued")


@app.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str) -> RunSummary:
    """Return the current status and full details of a single run from MLflow."""
    mlflow_run = _find_mlflow_run(run_id)
    if mlflow_run is None and run_id not in _run_status:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    scenario = _fetch_scenario(mlflow_run.info.run_id) if mlflow_run else None

    return RunSummary(
        run_id=run_id,
        status=_effective_status(run_id, mlflow_run.info.status if mlflow_run else None),
        mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
        parameters=_extract_parameters(mlflow_run) if mlflow_run else None,
        scenario=scenario,
    )


@app.get("/runs/{run_id}/recruiter.zip")
def download_recruiter(run_id: str) -> StreamingResponse:
    """Download recruiter package: brief/ + solution/ (from MLflow artifacts)."""
    mlflow_run_id = _resolve_mlflow_run(run_id)
    buf = _build_zip_from_mlflow(mlflow_run_id, [("brief", "brief"), ("solution", "solution")])
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=recruiter_{run_id}.zip"},
    )


@app.get("/runs/{run_id}/candidate.zip")
def download_candidate(run_id: str) -> StreamingResponse:
    """Download candidate package: brief/ + starter/ (without generate_data.py)."""
    mlflow_run_id = _resolve_mlflow_run(run_id)
    buf = _build_zip_from_mlflow(
        mlflow_run_id,
        [("brief", "brief"), ("starter", "starter")],
        exclude={"starter/generate_data.py"},
    )
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=candidate_{run_id}.zip"},
    )


def run() -> None:
    import uvicorn

    uvicorn.run("buc_factory.app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
