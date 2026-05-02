"""
FastAPI server for BUC Factory: run the agent and download assessment packages.

Endpoints:
    GET  /runs                         – list runs (on-disk + in-progress)
    POST /runs                         – submit a new agent run (async, 202)
    GET  /runs/{run_id}                – status of a single run
    GET  /runs/{run_id}/recruiter.zip  – brief/ + solution/
    GET  /runs/{run_id}/candidate.zip  – brief/ + starter/ (minus generate_data.py)

Usage:
    uvicorn buc_factory.app.api:app --reload
    python -m buc_factory.app.api
"""

import io
import json
import logging
import os
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from ..agent.entity import PLAN, BucState, DomainConfig
from ..agent.graph import build_graph
from ..tracking import (
    evaluate_outputs,
    log_config,
    log_output_artifacts,
    log_run_summary,
    log_tasks_summary,
    setup_mlflow,
)
from .models import RunListResponse, RunRequest, RunResponse, RunSummary

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

Path("log").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("log/agent.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

_run_status: dict[str, str] = {}  # in-memory: run_id -> "queued"|"running"|"done"|"failed"
_id_lock = Lock()


# ── app lifecycle ──────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    setup_mlflow()
    yield


app = FastAPI(
    title="BUC Factory",
    description="Generate and download BI recruitment assessment packages.",
    lifespan=_lifespan,
)


# ── internal helpers ───────────────────────────────────────────────

def _next_run_id() -> str:
    """Atomically allocate the next run_NNN directory and return its name."""
    with _id_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        existing = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        nums = [int(n[4:]) for n in existing if n.startswith("run_") and n[4:].isdigit()]
        run_id = f"run_{max(nums, default=0) + 1:03d}"
        (DATA_DIR / run_id).mkdir()
        return run_id


def _resolve_run(run_id: str) -> Path:
    path = (DATA_DIR / run_id).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return path


def _disk_status(run_dir: Path) -> str:
    state_file = run_dir / "state.json"
    if not state_file.exists():
        return "unknown"
    try:
        state = json.loads(state_file.read_text())
        idx = state.get("task_index", 0)
        return "complete" if idx >= len(PLAN) else f"partial ({idx}/{len(PLAN)} tasks)"
    except (json.JSONDecodeError, KeyError):
        return "error"


def _build_zip(
    run: Path,
    folders: list[tuple[str, str]],
    exclude: set[str] | None = None,
) -> io.BytesIO:
    excluded = exclude or set()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for subdir, prefix in folders:
            src = run / subdir
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
    state_file = output_dir / "state.json"
    saved: dict[str, Any] = {}
    if state_file.exists():
        saved = json.loads(state_file.read_text())
        task_index = saved.get("task_index", 0)
        if task_index >= len(PLAN):
            raise ValueError("run already complete")
        LOGGER.info(f"resuming from task [{PLAN[task_index]}] (index {task_index})")
    else:
        task_index = 0

    return BucState(
        messages=[],
        output_dir=str(output_dir),
        task_index=task_index,
        current_task=PLAN[task_index],
        tasks_remaining=saved.get("tasks_remaining", PLAN[task_index:]),
        retry_count=0,
        bootstrapped_dimensions=saved.get("bootstrapped_dimensions"),
        bootstrapped_entities=saved.get("bootstrapped_entities"),
        scenario=saved.get("scenario"),
        rolled=None,
        validation_error=None,
        failed=False,
    )


def _execute_run(run_id: str, cfg: DomainConfig, output_dir: Path) -> None:
    """Full agent pipeline — runs synchronously in a background thread."""
    _run_status[run_id] = "running"
    run_name = f"{cfg.industry}__{cfg.role}".replace(" ", "_").lower()
    try:
        with mlflow.start_run(run_name=run_name):
            log_config(cfg)
            mlflow.set_tags(
                {
                    "output_dir": str(output_dir),
                    "model": "claude-opus-4-7",
                    "task_count": len(PLAN),
                }
            )
            initial_state = _load_initial_state(output_dir)
            graph = build_graph(output_dir, cfg)

            t0 = time.perf_counter()
            final_state: BucState = graph.invoke(initial_state)
            total = time.perf_counter() - t0

            failed = bool(final_state.get("failed"))
            tasks_completed = final_state.get("task_index", 0)

            log_run_summary(total_s=total, failed=failed, tasks_completed=tasks_completed)
            log_tasks_summary()
            log_output_artifacts(output_dir)

            if not failed:
                evaluate_outputs(output_dir, cfg)

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
    """List all runs with their current status."""
    if not DATA_DIR.is_dir():
        return RunListResponse(runs=[])

    runs = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        status = _run_status.get(d.name) or _disk_status(d)
        runs.append(RunSummary(run_id=d.name, status=status))

    return RunListResponse(runs=runs)


@app.post("/runs", status_code=202, response_model=RunResponse)
def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    """Submit a new agent run. Returns immediately; agent executes in the background."""
    run_id = _next_run_id()
    output_dir = DATA_DIR / run_id
    cfg = DomainConfig(**request.model_dump())
    _run_status[run_id] = "queued"
    background_tasks.add_task(_execute_run, run_id, cfg, output_dir)
    return RunResponse(run_id=run_id, status="queued")


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    """Return the current status of a single run."""
    run = _resolve_run(run_id)
    status = _run_status.get(run_id) or _disk_status(run)
    return RunResponse(run_id=run_id, status=status)


@app.get("/runs/{run_id}/recruiter.zip")
def download_recruiter(run_id: str) -> StreamingResponse:
    """Download recruiter package: brief/ + solution/."""
    run = _resolve_run(run_id)
    buf = _build_zip(run, [("brief", "brief"), ("solution", "solution")])
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=recruiter_{run_id}.zip"},
    )


@app.get("/runs/{run_id}/candidate.zip")
def download_candidate(run_id: str) -> StreamingResponse:
    """Download candidate package: brief/ + starter/ (without generate_data.py)."""
    run = _resolve_run(run_id)
    buf = _build_zip(
        run,
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
