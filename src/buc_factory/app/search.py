"""Embedding-based semantic search over completed runs."""

import json
import logging
import os
import sqlite3
from pathlib import Path

import numpy as np
from openai import OpenAI

LOGGER = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_DB_PATH = Path(os.environ.get("EMBEDDINGS_DB_PATH", "mlflow_data/embeddings.db"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_embeddings (
            api_run_id    TEXT PRIMARY KEY,
            mlflow_run_id TEXT,
            embed_text    TEXT,
            embedding     TEXT
        )
    """)
    conn.commit()
    return conn


def build_index_text(params: dict, scenario: dict | None) -> str:
    """Build the text to embed for a run, shared by live indexing and backfill."""
    g = params.get
    header = (
        f"Role: {g('role', '')}. "
        f"Seniority: {g('seniority', '')}. "
        f"Industry: {g('industry', '')}. "
        f"Tool: {g('tool', '')}. "
        f"Location: {g('location', '')}. "
        f"Language: {g('language', '')}."
    )
    if not scenario:
        return header

    scenario_parts = []
    for v in scenario.values():
        if isinstance(v, list):
            scenario_parts.extend(str(i) for i in v)
        else:
            scenario_parts.append(str(v))

    return header + " Scenario: " + ", ".join(p for p in scenario_parts if p)


def _embed(text: str) -> list[float]:
    return _get_client().embeddings.create(input=text, model=_EMBED_MODEL).data[0].embedding


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def store_embedding(api_run_id: str, mlflow_run_id: str, text: str) -> None:
    try:
        vector = _embed(text)
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_embeddings VALUES (?, ?, ?, ?)",
                (api_run_id, mlflow_run_id, text, json.dumps(vector)),
            )
        LOGGER.info("[%s] embedding stored (%d dims)", api_run_id, len(vector))
    except Exception as exc:
        LOGGER.warning("embedding store failed for %s: %s", api_run_id, exc)


def search(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return up to k (api_run_id, cosine_score) pairs, sorted descending."""
    if not _DB_PATH.exists():
        return []
    try:
        query_vec = _embed(query)
        with _get_conn() as conn:
            rows = conn.execute("SELECT api_run_id, embedding FROM run_embeddings").fetchall()
        if not rows:
            return []
        scores = [(rid, _cosine(query_vec, json.loads(vec))) for rid, vec in rows]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    except Exception as exc:
        LOGGER.warning("search failed: %s", exc)
        return []
