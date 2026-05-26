"""
FastAPI server for BUC Factory: run the agent and download assessment packages.

Endpoints:
    GET  /runs                              – list all runs from MLflow (params + scenario + status)
    GET  /runs/search?q=…&limit=…          – semantic search over completed runs
    POST /runs                              – submit a new agent run (async, 202)
    GET  /runs/{run_id}                     – run details from MLflow (params, scenario, status)
    GET  /runs/{run_id}/brief               – candidate brief as plain-text markdown
    GET  /runs/{run_id}/solution            – recruiter solution as plain-text markdown
    GET  /runs/{run_id}/recruiter.zip       – brief/ + solution/ (from MLflow artifacts)
    GET  /runs/{run_id}/candidate.zip       – brief/ + starter/ (minus generate_data.py)

    POST /simulations                       – simulate a candidate completing an assessment (202)
    GET  /simulations/{sim_id}              – simulation status and metadata
    GET  /simulations/{sim_id}/solution.zip – completed starter project as a downloadable ZIP

Usage:
    uvicorn buc_factory.app.api:app --reload
    python -m buc_factory.app.api
"""

import importlib.resources
import io
import json
import logging
import logging.handlers
import random
import shutil
import tempfile
import time
import zipfile
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, cast

import mlflow
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
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
from .models import (
    RunListResponse,
    RunParameters,
    RunRequest,
    RunResponse,
    RunSummary,
    SearchResult,
    SimulationDetails,
    SimulationRequest,
    SimulationResponse,
)
from .search import build_index_text, store_embedding
from .search import search as _search_runs

load_dotenv()

_EXPERIMENT = "buc-factory"
_SIM_EXPERIMENT = "buc-factory-simulations"
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

# ── simulation session state ───────────────────────────────────────
_sim_status: dict[str, str] = {}  # sim_id -> "queued"|"running"|"done"|"failed"
_sim_mlflow_ids: dict[str, str] = {}  # sim_id -> MLflow run UUID
_sim_source_run: dict[str, str] = {}  # sim_id -> source run_id
_sim_meta: dict[str, dict[str, Any]] = {}  # sim_id -> {mode, proficiency, seed}
_sim_counter = 0
_sim_lock = Lock()

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


def _init_sim_counter() -> None:
    """Seed _sim_counter from the highest api_sim_id already registered in MLflow."""
    global _sim_counter
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_SIM_EXPERIMENT)
    if experiment is None:
        return
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.api_sim_id != ''",
        max_results=1000,
    )
    nums = [
        int(r.data.tags["api_sim_id"][4:])
        for r in runs
        if r.data.tags.get("api_sim_id", "").startswith("sim_")
        and r.data.tags["api_sim_id"][4:].isdigit()
    ]
    if nums:
        _sim_counter = max(nums)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    setup_mlflow()
    _init_run_counter()
    _init_sim_counter()
    yield


_STATIC_DIR = importlib.resources.files("buc_factory").joinpath("static")

app = FastAPI(
    title="BUC Factory",
    description="Generate and download BI recruitment assessment packages.",
    lifespan=_lifespan,
    docs_url=None,  # replaced by custom endpoint below
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon() -> Response:
    return Response(status_code=204)


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


def _fetch_scenario(mlflow_run_id: str) -> dict[str, Any] | None:
    """Download and parse outputs/scenario.json from MLflow artifacts."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path="outputs/scenario.json",
                dst_path=tmpdir,
            )
            return cast(dict[str, Any], json.loads(Path(local).read_text()))
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
    return str(mlflow_run.info.run_id)


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


def _index_run(run_id: str, mlflow_run_id: str, cfg: DomainConfig, state: BucState) -> None:
    text = build_index_text(asdict(cfg), state.get("scenario"))
    store_embedding(run_id, mlflow_run_id, text)


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


def _next_sim_id() -> str:
    global _sim_counter
    with _sim_lock:
        _sim_counter += 1
        return f"sim_{_sim_counter:03d}"


def _effective_sim_status(sim_id: str, mlflow_status: str | None) -> str:
    if sim_id in _sim_status:
        return _sim_status[sim_id]
    return _MLFLOW_STATUS_MAP.get(mlflow_status or "", "unknown")


def _find_sim_mlflow_run(sim_id: str) -> Any | None:
    """Return the MLflow Run for a simulation sim_id, or None."""
    mlflow_uuid = _sim_mlflow_ids.get(sim_id)
    if mlflow_uuid:
        try:
            return mlflow.MlflowClient().get_run(mlflow_uuid)
        except Exception:
            pass
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_SIM_EXPERIMENT)
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.api_sim_id = '{sim_id}'",
        max_results=1,
    )
    return runs[0] if runs else None


def _resolve_sim_run(sim_id: str) -> str:
    """Return the MLflow run UUID for a simulation, or raise an appropriate HTTP error."""
    if _sim_status.get(sim_id) in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Simulation '{sim_id}' artifacts not yet available"
            f" (status: {_sim_status[sim_id]}).",
        )
    mlflow_run = _find_sim_mlflow_run(sim_id)
    if mlflow_run is None:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found.")
    return str(mlflow_run.info.run_id)


def _execute_simulation(
    sim_id: str, source_run_id: str, mode: str, proficiency: float, seed: int | None
) -> None:
    """Full simulation pipeline — runs synchronously in a background thread."""
    set_run_id(sim_id)
    try:
        _execute_simulation_inner(sim_id, source_run_id, mode, proficiency, seed)
    finally:
        set_run_id("-")


def _execute_simulation_inner(
    sim_id: str, source_run_id: str, mode: str, proficiency: float, seed: int | None
) -> None:
    from ..agent.candidate_graph import CandidateState, build_candidate_graph

    _sim_status[sim_id] = "running"
    LOGGER.info(
        "simulation %s: source=%s mode=%s proficiency=%.2f seed=%s",
        sim_id,
        source_run_id,
        mode,
        proficiency,
        seed,
    )

    try:
        # Locate source run artifacts
        source_mlflow_run = _find_mlflow_run(source_run_id)
        if source_mlflow_run is None:
            raise ValueError(f"Source run '{source_run_id}' not found in MLflow")
        source_mlflow_id = source_mlflow_run.info.run_id
        deliverable_format = source_mlflow_run.data.params.get("deliverable_format", "PBIP")

        # Resolve proficiency for random mode
        if mode == "random" and proficiency == 0.0:
            rng = random.Random(seed)
            proficiency = round(rng.uniform(0.3, 0.95), 2)
            LOGGER.info("  random proficiency sampled: %.2f", proficiency)

        # Get or create the simulations experiment (thread-safe via MlflowClient)
        client = mlflow.MlflowClient()
        sim_exp = client.get_experiment_by_name(_SIM_EXPERIMENT)
        sim_exp_id = (
            sim_exp.experiment_id
            if sim_exp is not None
            else client.create_experiment(_SIM_EXPERIMENT)
        )

        run_name = f"{sim_id}__{source_run_id}__{mode}"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Copy brief/ and starter/ from the source run into the simulation workspace
            LOGGER.info("  downloading source artifacts from mlflow run %s", source_mlflow_id)
            with tempfile.TemporaryDirectory() as artdir:
                local_artifacts = mlflow.artifacts.download_artifacts(
                    run_id=source_mlflow_id,
                    artifact_path="outputs",
                    dst_path=artdir,
                )
                src_root = Path(local_artifacts)
                for folder in ("brief", "starter"):
                    src_folder = src_root / folder
                    if src_folder.exists():
                        shutil.copytree(src_folder, output_dir / folder)
                        n = sum(1 for p in (output_dir / folder).rglob("*") if p.is_file())
                        LOGGER.info("  copied %s/ (%d files)", folder, n)

            with mlflow.start_run(run_name=run_name, experiment_id=sim_exp_id) as active_run:
                _sim_mlflow_ids[sim_id] = active_run.info.run_id
                mlflow.log_params(
                    {
                        "mode": mode,
                        "proficiency": proficiency,
                        "seed": seed if seed is not None else "random",
                        "source_run_id": source_run_id,
                        "deliverable_format": deliverable_format,
                    }
                )
                mlflow.set_tags(
                    {
                        "api_sim_id": sim_id,
                        "source_run_id": source_run_id,
                        "source_mlflow_run_id": source_mlflow_id,
                    }
                )

                graph = build_candidate_graph(output_dir, mode, proficiency, deliverable_format)
                initial_state = CandidateState(
                    messages=[],
                    output_dir=str(output_dir),
                    mode=mode,
                    proficiency=proficiency,
                    deliverable_format=deliverable_format,
                    retry_count=0,
                    validation_error=None,
                    failed=False,
                )

                t0 = time.perf_counter()
                final_state: CandidateState = graph.invoke(initial_state)
                elapsed = time.perf_counter() - t0

                failed = bool(final_state.get("failed"))
                LOGGER.info(
                    "  simulation %s in %.1fs",
                    "failed" if failed else "complete",
                    elapsed,
                )

                # Upload the completed starter project as MLflow artifacts
                starter_out = output_dir / "starter"
                if starter_out.exists():
                    mlflow.log_artifacts(str(starter_out), artifact_path="solution")
                mlflow.log_metric("elapsed_s", elapsed)

                # Score the simulation against the recruiter solution
                if not failed:
                    try:
                        from ..agent.scorer import score_submission

                        with tempfile.TemporaryDirectory() as score_tmp:
                            sol_local = mlflow.artifacts.download_artifacts(
                                run_id=source_mlflow_id,
                                artifact_path="outputs/solution/recruiter_solution.md",
                                dst_path=score_tmp,
                            )
                            recruiter_solution = Path(sol_local).read_text(encoding="utf-8")

                        scoring_md = score_submission(
                            output_dir=output_dir,
                            recruiter_solution=recruiter_solution,
                            deliverable_format=deliverable_format,
                        )
                        if scoring_md:
                            score_path = output_dir / "scoring.md"
                            score_path.write_text(scoring_md, encoding="utf-8")
                            mlflow.log_artifact(str(score_path), artifact_path=None)
                            LOGGER.info("  scoring complete (%d chars)", len(scoring_md))
                    except Exception as exc:
                        LOGGER.warning("scoring step failed (non-fatal): %s", exc)

                if failed:
                    mlflow.MlflowClient().set_terminated(active_run.info.run_id, status="FAILED")

        _sim_status[sim_id] = "failed" if failed else "done"

    except Exception as exc:
        LOGGER.error("simulation %s failed: %s", sim_id, exc, exc_info=True)
        _sim_status[sim_id] = "failed"


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
                    _index_run(run_id, active_run.info.run_id, cfg, final_state)
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
        for _mlf_run in client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time ASC"],
            max_results=1000,
        ):
            api_run_id = _mlf_run.data.tags.get("api_run_id")
            if api_run_id:
                mlflow_by_run_id[api_run_id] = _mlf_run

    # Include queued runs not yet registered in MLflow
    all_run_ids = sorted(set(mlflow_by_run_id.keys()) | set(_run_status.keys()))

    def _fetch_for(rid: str) -> tuple[str, dict[str, Any] | None]:
        mlflow_run = mlflow_by_run_id.get(rid)
        return rid, _fetch_scenario(str(mlflow_run.info.run_id)) if mlflow_run else None

    scenarios: dict[str, dict[str, Any] | None] = {}
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


@app.get("/runs/search", response_model=list[SearchResult])
def search_runs(
    q: str = Query(..., description="Free-text query matched by semantic similarity"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
) -> list[SearchResult]:
    """Semantic search over completed runs using embedding similarity."""
    hits = _search_runs(q, k=limit)
    if not hits:
        return []
    results = []
    for api_run_id, score in hits:
        mlflow_run = _find_mlflow_run(api_run_id)
        scenario = _fetch_scenario(mlflow_run.info.run_id) if mlflow_run else None
        results.append(
            SearchResult(
                score=score,
                run_id=api_run_id,
                status=_effective_status(
                    api_run_id, mlflow_run.info.status if mlflow_run else None
                ),
                mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
                parameters=_extract_parameters(mlflow_run) if mlflow_run else None,
                scenario=scenario,
            )
        )
    return results


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


@app.get("/runs/{run_id}/brief", response_class=PlainTextResponse)
def get_brief(run_id: str) -> str:
    """Return the candidate brief as plain-text markdown."""
    mlflow_run_id = _resolve_mlflow_run(run_id)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path="outputs/brief/candidate_brief.md",
                dst_path=tmpdir,
            )
            return Path(local).read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Brief not found: {exc}") from exc


@app.get("/runs/{run_id}/solution", response_class=PlainTextResponse)
def get_solution(run_id: str) -> str:
    """Return the recruiter solution as plain-text markdown."""
    mlflow_run_id = _resolve_mlflow_run(run_id)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path="outputs/solution/recruiter_solution.md",
                dst_path=tmpdir,
            )
            return Path(local).read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Solution not found: {exc}") from exc


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


@app.post("/runs/{run_id}/score", response_class=PlainTextResponse)
async def score_run(run_id: str, solution: Annotated[UploadFile, File(...)]) -> str:
    """Score a candidate-submitted solution ZIP against the recruiter answer key.

    The ZIP must contain the completed project at its root (PBIP or IPYNB layout),
    matching the deliverable_format of the run.
    """
    from ..agent.scorer import score_submission

    mlflow_run_id = _resolve_mlflow_run(run_id)
    mlflow_run = _find_mlflow_run(run_id)
    if mlflow_run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    deliverable_format = mlflow_run.data.params.get("deliverable_format", "PBIP")

    zip_bytes = await solution.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    import zipfile as _zipfile

    if not _zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP.")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            starter_dir = output_dir / "starter"
            starter_dir.mkdir()

            with _zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(starter_dir)

            # If the ZIP wrapped everything in a single top-level folder, strip it
            # so scorer always sees files directly under starter/.
            top_level = [p for p in starter_dir.iterdir() if not p.name.startswith("__")]
            if len(top_level) == 1 and top_level[0].is_dir():
                for child in list(top_level[0].iterdir()):
                    child.rename(starter_dir / child.name)
                top_level[0].rmdir()

            # Brief
            with tempfile.TemporaryDirectory() as artdir:
                local_brief = mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run_id,
                    artifact_path="outputs/brief/candidate_brief.md",
                    dst_path=artdir,
                )
                brief_dest = output_dir / "brief"
                brief_dest.mkdir()
                shutil.copy(local_brief, brief_dest / "candidate_brief.md")

            # Recruiter solution
            with tempfile.TemporaryDirectory() as artdir:
                local_sol = mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run_id,
                    artifact_path="outputs/solution/recruiter_solution.md",
                    dst_path=artdir,
                )
                recruiter_solution = Path(local_sol).read_text(encoding="utf-8")

            scoring_md = score_submission(
                output_dir=output_dir,
                recruiter_solution=recruiter_solution,
                deliverable_format=deliverable_format,
            )

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("score_run %s failed: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    if scoring_md is None:
        raise HTTPException(status_code=422, detail="Scorer returned no result. Check server logs.")

    return scoring_md


# ── simulation endpoints ───────────────────────────────────────────


@app.post("/simulations", status_code=202, response_model=SimulationResponse)
def create_simulation(
    request: SimulationRequest, background_tasks: BackgroundTasks
) -> SimulationResponse:
    """Simulate a candidate completing an assessment.

    Spawns a background agent that reads the source run's brief and starter project,
    then fills in the solution according to the requested proficiency level.

    - **mode=perfect**: agent behaves as an expert; all tasks completed flawlessly.
    - **mode=random**: agent simulates a candidate whose skill matches *proficiency* (0.0–1.0).
      When *proficiency* is omitted it is sampled uniformly in [0.3, 0.95] (use *seed* to fix it).
    """
    # Validate source run exists before queuing
    if _find_mlflow_run(request.run_id) is None and request.run_id not in _run_status:
        raise HTTPException(status_code=404, detail=f"Source run '{request.run_id}' not found.")

    mode = request.mode
    if mode not in ("perfect", "random"):
        raise HTTPException(status_code=422, detail="mode must be 'perfect' or 'random'")

    # For perfect mode, proficiency is always 1.0
    if mode == "perfect":
        proficiency = 1.0
    else:
        # 0.0 is sentinel meaning "sample randomly inside the background task"
        proficiency = request.proficiency if request.proficiency is not None else 0.0

    sim_id = _next_sim_id()
    _sim_status[sim_id] = "queued"
    _sim_source_run[sim_id] = request.run_id
    _sim_meta[sim_id] = {"mode": mode, "proficiency": proficiency, "seed": request.seed}

    background_tasks.add_task(
        _execute_simulation, sim_id, request.run_id, mode, proficiency, request.seed
    )
    return SimulationResponse(simulation_id=sim_id, status="queued")


@app.get("/simulations/{sim_id}", response_model=SimulationDetails)
def get_simulation(sim_id: str) -> SimulationDetails:
    """Return the status and metadata of a candidate simulation."""
    mlflow_run = _find_sim_mlflow_run(sim_id)
    if mlflow_run is None and sim_id not in _sim_status:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found.")

    meta = _sim_meta.get(sim_id, {})
    if mlflow_run is not None:
        p = mlflow_run.data.params
        meta = {
            "mode": p.get("mode", meta.get("mode", "unknown")),
            "proficiency": float(p["proficiency"])
            if "proficiency" in p
            else meta.get("proficiency"),
            "seed": int(p["seed"]) if p.get("seed", "random") != "random" else None,
        }

    return SimulationDetails(
        simulation_id=sim_id,
        source_run_id=_sim_source_run.get(sim_id)
        or (mlflow_run.data.tags.get("source_run_id") if mlflow_run else "unknown"),
        status=_effective_sim_status(sim_id, mlflow_run.info.status if mlflow_run else None),
        mode=meta.get("mode", "unknown"),
        proficiency=meta.get("proficiency"),
        seed=meta.get("seed"),
        mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
    )


@app.get("/simulations/{sim_id}/scoring", response_class=PlainTextResponse)
def get_simulation_scoring(sim_id: str) -> str:
    """Return the scoring markdown for a completed simulation."""
    mlflow_run_id = _resolve_sim_run(sim_id)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path="scoring.md",
                dst_path=tmpdir,
            )
            return Path(local).read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Scoring not found: {exc}") from exc


@app.get("/simulations/{sim_id}/solution.zip")
def download_simulation_solution(sim_id: str) -> StreamingResponse:
    """Download the completed assessment project as a ZIP file.

    The archive contains the fully filled-in starter project (notebook or PBIP)
    including all data files, ready for the recruiter to open and evaluate.
    """
    mlflow_run_id = _resolve_sim_run(sim_id)
    buf = io.BytesIO()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path="solution",
                dst_path=tmpdir,
            )
            solution_path = Path(local_path)
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(solution_path.rglob("*")):
                    if not file_path.is_file():
                        continue
                    arc_name = file_path.relative_to(solution_path).as_posix()
                    if arc_name == "generate_data.py":
                        continue
                    zf.write(file_path, arc_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Solution artifacts not found: {exc}") from exc
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=solution_{sim_id}.zip"},
    )


def run() -> None:
    import uvicorn

    uvicorn.run("buc_factory.app.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
