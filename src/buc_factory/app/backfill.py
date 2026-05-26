"""Backfill embeddings for existing MLflow runs that predate the search index.

Usage:
    buc-factory-backfill
    python -m buc_factory.app.backfill
"""

import json
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, cast

import mlflow
from dotenv import load_dotenv

from .search import _DB_PATH, build_index_text, store_embedding

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

_EXPERIMENT = "buc-factory"


def _already_indexed() -> set[str]:
    if not _DB_PATH.exists():
        return set()
    with sqlite3.connect(_DB_PATH) as conn:
        rows = conn.execute("SELECT api_run_id FROM run_embeddings").fetchall()
    return {r[0] for r in rows}


def _fetch_scenario(run: mlflow.entities.Run) -> dict[str, Any] | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path="outputs/scenario.json",
                dst_path=tmpdir,
            )
            return cast(dict[str, Any], json.loads(Path(local).read_text()))
    except Exception:
        return None


def backfill() -> None:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(_EXPERIMENT)
    if experiment is None:
        LOGGER.error("Experiment '%s' not found in MLflow.", _EXPERIMENT)
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.api_run_id != '' AND attributes.status = 'FINISHED'",
        max_results=10_000,
    )
    if not runs:
        LOGGER.info("No runs with api_run_id tag found.")
        return

    already = _already_indexed()
    pending = [r for r in runs if r.data.tags.get("api_run_id") not in already]

    LOGGER.info(
        "%d total runs, %d already indexed, %d to backfill.",
        len(runs),
        len(already),
        len(pending),
    )

    ok = skipped = 0
    for run in pending:
        api_run_id = run.data.tags["api_run_id"]
        text = build_index_text(run.data.params, _fetch_scenario(run))
        if not text.strip():
            LOGGER.warning("[%s] no text to index, skipping.", api_run_id)
            skipped += 1
            continue
        store_embedding(api_run_id, run.info.run_id, text)
        ok += 1

    LOGGER.info("Done — %d indexed, %d skipped.", ok, skipped)


if __name__ == "__main__":
    backfill()
