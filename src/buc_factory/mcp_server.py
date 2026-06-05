"""Dev MCP server — gives Claude Code direct access to runs, artifacts, and configs."""

import json
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("buc-factory")

_ROOT = Path(__file__).resolve().parent.parent.parent
_MLFLOW_DB = _ROOT / "data" / "mlflow" / "mlflow.db"
_ARTIFACTS = _ROOT / "data" / "mlflow" / "artifacts"
_CONF_DIR = _ROOT / "conf" / "industry_spec"


# ── helpers ───────────────────────────────────────────────────────────────────


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_MLFLOW_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _artifact_dir(run_id: str) -> Path:
    with _connect() as conn:
        row = conn.execute(
            "SELECT experiment_id FROM runs WHERE run_uuid = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"run {run_id!r} not found")
    return _ARTIFACTS / str(row["experiment_id"]) / run_id / "artifacts"


def _safe_path(base: Path, relative: str) -> Path:
    target = (base / relative).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise ValueError("path traversal not allowed")
    return target


# ── tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_runs() -> str:
    """List all buc-factory runs with their status, config params, and relevancy score."""
    with _connect() as conn:
        runs = conn.execute(
            "SELECT run_uuid, status, start_time, end_time "
            "FROM runs WHERE lifecycle_stage = 'active' ORDER BY start_time DESC"
        ).fetchall()

        result = []
        for run in runs:
            rid = run["run_uuid"]
            params = {
                r["key"]: r["value"]
                for r in conn.execute(
                    "SELECT key, value FROM params WHERE run_uuid = ?", (rid,)
                ).fetchall()
            }
            metrics = {
                r["key"]: r["value"]
                for r in conn.execute(
                    "SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (rid,)
                ).fetchall()
            }
            result.append(
                {
                    "run_id": rid,
                    "status": run["status"],
                    "start_time": run["start_time"],
                    "end_time": run["end_time"],
                    "params": params,
                    "metrics": metrics,
                }
            )

    return json.dumps(result, indent=2)


@mcp.tool()
def get_run(run_id: str) -> str:
    """Get full details for a run: params, metrics, tags, and artifact path."""
    with _connect() as conn:
        run = conn.execute(
            "SELECT run_uuid, status, start_time, end_time, experiment_id "
            "FROM runs WHERE run_uuid = ? AND lifecycle_stage = 'active'",
            (run_id,),
        ).fetchone()
        if run is None:
            return json.dumps({"error": f"run {run_id!r} not found"})

        params = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key, value FROM params WHERE run_uuid = ?", (run_id,)
            ).fetchall()
        }
        metrics = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (run_id,)
            ).fetchall()
        }
        tags = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,)
            ).fetchall()
        }

    artifact_dir = _ARTIFACTS / str(run["experiment_id"]) / run_id / "artifacts"

    return json.dumps(
        {
            "run_id": run_id,
            "status": run["status"],
            "start_time": run["start_time"],
            "end_time": run["end_time"],
            "params": params,
            "metrics": metrics,
            "tags": tags,
            "artifact_dir": str(artifact_dir),
        },
        indent=2,
    )


@mcp.tool()
def list_artifacts(run_id: str) -> str:
    """List all artifact files for a run (outputs, prompts, task_summary)."""
    try:
        base = _artifact_dir(run_id)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not base.exists():
        return json.dumps({"error": f"artifact directory not found for run {run_id!r}"})

    files = sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())
    return json.dumps({"run_id": run_id, "files": files}, indent=2)


@mcp.tool()
def read_artifact(run_id: str, path: str) -> str:
    """Read a specific artifact file from a run. Use list_artifacts to discover paths."""
    try:
        base = _artifact_dir(run_id)
        target = _safe_path(base, path)
    except ValueError as e:
        return f"ERROR: {e}"

    if not target.exists():
        return f"ERROR: {path!r} does not exist in run {run_id!r}"

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path!r} is a binary file — cannot read as text"


@mcp.tool()
def list_configs() -> str:
    """List available YAML configs in conf/industry_spec/."""
    if not _CONF_DIR.exists():
        return json.dumps({"error": "conf/industry_spec/ not found"})
    configs = sorted(p.name for p in _CONF_DIR.glob("*.yml"))
    return json.dumps({"configs": configs}, indent=2)


@mcp.tool()
def read_config(name: str) -> str:
    """Read a config YAML from conf/industry_spec/. Pass the filename (e.g. saas_france.yml)."""
    try:
        target = _safe_path(_CONF_DIR, name)
    except ValueError as e:
        return f"ERROR: {e}"

    if not target.exists():
        return f"ERROR: config {name!r} not found"
    if target.suffix not in {".yml", ".yaml"}:
        return f"ERROR: {name!r} is not a YAML file"

    return target.read_text(encoding="utf-8")


# ── entrypoint ────────────────────────────────────────────────────────────────


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
