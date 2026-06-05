"""Power BI scoring subpackage.

Owns the dimension definitions and prompt templates for evaluating
PBIP deliverables.  Dimensions focus on modelling reasoning, DAX strategy,
and report maturity — not on whether numbers match the answer key.
"""

from __future__ import annotations

import importlib.resources
from typing import Any, cast

import yaml

_CACHE: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = cast(
            dict[str, Any],
            yaml.safe_load(
                importlib.resources.files("buc_factory.scorer.powerbi")
                .joinpath("prompt_template.yml")
                .read_text(encoding="utf-8")
            ),
        )
    return _CACHE


def dimensions() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _load()["dimensions"])


def load_prompt() -> tuple[str, str]:
    """Return (system_prompt, user_template) with dimension rubrics baked in."""
    tmpl = _load()
    dim_lines = "\n".join(
        f"  {i + 1}. [{d['category'].upper()}] {d['name']} (max {d['max_score']}/10)\n"
        f"     {d['rubric']}"
        for i, d in enumerate(tmpl["dimensions"])
    )
    schema_dims = "\n".join(
        f'      {{"name": "{d["name"]}", "display_name": "<localized name in brief language>", '
        f'"category": "{d["category"]}", '
        f'"score": <0–{d["max_score"]}>, "max_score": {d["max_score"]}, '
        f'"comment": "<1-2 sentences in brief language>"}}'
        for d in tmpl["dimensions"]
    )
    max_total = sum(d["max_score"] for d in tmpl["dimensions"])
    system = (
        tmpl["system"]
        .replace("{dimensions}", dim_lines)
        .replace("{schema_dimensions}", schema_dims)
        .replace("{max_total_score}", str(max_total))
        .replace("{tool_type}", "powerbi")
    )
    return system, tmpl["user"]
