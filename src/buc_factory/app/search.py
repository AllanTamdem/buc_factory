"""Embedding-based semantic search over completed runs."""

import json
import logging
import os
import sqlite3
from pathlib import Path

import numpy as np
from openai import OpenAI

from buc_factory.llm.gptai import OpenAILLM

LOGGER = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_DB_PATH = Path(os.environ.get("EMBEDDINGS_DB_PATH", "data/embeddings.db"))

_embed_client: OpenAI | None = None
_llm: OpenAILLM | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI()
    return _embed_client


def _get_llm() -> OpenAILLM:
    global _llm
    if _llm is None:
        _llm = OpenAILLM(model="gpt-4o-mini", max_tokens=80)
    return _llm


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
    title = " ".join(p for p in [g("seniority", ""), g("role", "")] if p)
    industry = g("industry", "")
    tool = g("tool", "")
    location = g("location", "")
    language = g("language", "")
    header = f"A {title}"
    if industry:
        header += f" working in the {industry} industry"
    if tool:
        header += f", using {tool}"
    if location:
        header += f", based in {location}"
    if language:
        header += f", speaking {language}"
    header += "."

    if not scenario:
        return header

    scenario_parts = []
    for k, v in scenario.items():
        if isinstance(v, list):
            scenario_parts.extend(f"{k}: {i}" for i in v if str(i))
        elif str(v):
            scenario_parts.append(f"{k}: {v}")

    return header + " Scenario: " + ", ".join(scenario_parts)


def _embed(text: str) -> list[float]:
    return _get_embed_client().embeddings.create(input=text, model=_EMBED_MODEL).data[0].embedding


def _hypothetical_doc(query: str) -> str:
    """Generate a synthetic run description from a natural-language query (HyDE)."""
    prompt = (
        f"Write one sentence describing a BUC coaching run for: {query}. "
        "Mention role, seniority, industry, tool, location, and language where relevant."
    )
    return _get_llm().complete(prompt, model="gpt-4o-mini", max_tokens=80)


def _batch_cosine(query_vec: list[float], matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between query_vec and every row in matrix."""
    q = np.array(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(len(matrix))
    q = q / q_norm
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = np.divide(matrix, row_norms, where=row_norms > 0)
    return normed @ q


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


def search(
    query: str, k: int = 10, min_score: float = 0.0, use_hyde: bool = True
) -> list[tuple[str, float]]:
    """Return up to k (api_run_id, cosine_score) pairs, sorted descending.

    use_hyde: expand the query into a hypothetical run description before embedding,
    for better alignment with indexed prose (costs one extra LLM call).
    """
    if not _DB_PATH.exists():
        return []
    try:
        embed_input = _hypothetical_doc(query) if use_hyde else query
        query_vec = _embed(embed_input)
        with _get_conn() as conn:
            rows = conn.execute("SELECT api_run_id, embedding FROM run_embeddings").fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        matrix = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
        scores = _batch_cosine(query_vec, matrix)
        order = np.argsort(scores)[::-1]
        return [(ids[i], float(scores[i])) for i in order if scores[i] >= min_score][:k]
    except Exception as exc:
        LOGGER.warning("search failed: %s", exc)
        return []
