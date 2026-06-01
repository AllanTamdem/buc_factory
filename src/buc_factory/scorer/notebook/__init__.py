"""Notebook (Python) scoring subpackage.

Owns dimension definitions and prompt templates for evaluating IPYNB deliverables.
Separate dimension sets for Data Analyst and Data Scientist roles.
"""

from __future__ import annotations

import importlib.resources
from typing import Any, Literal, cast

import yaml

_CACHE: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = cast(
            dict[str, Any],
            yaml.safe_load(
                importlib.resources.files("buc_factory.scorer.notebook")
                .joinpath("prompt_template.yml")
                .read_text(encoding="utf-8")
            ),
        )
    return _CACHE


def dimensions(tool_type: Literal["python_da", "python_ds"]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _load()["dimensions"][tool_type])


def load_prompt(tool_type: Literal["python_da", "python_ds"]) -> tuple[str, str]:
    """Return (system_prompt, user_template) with dimension rubrics baked in."""
    tmpl = _load()
    dims = tmpl["dimensions"][tool_type]
    dim_lines = "\n".join(
        f"  {i + 1}. [{d['category'].upper()}] {d['name']} (max {d['max_score']}/10)\n"
        f"     {d['rubric']}"
        for i, d in enumerate(dims)
    )
    schema_dims = "\n".join(
        f'      {{"name": "{d["name"]}", "category": "{d["category"]}", '
        f'"score": <0–{d["max_score"]}>, "max_score": {d["max_score"]}, '
        f'"comment": "<one sentence>"}}'
        for d in dims
    )
    max_total = sum(d["max_score"] for d in dims)
    role_label = "Data Scientist" if tool_type == "python_ds" else "Data Analyst"
    system = (
        tmpl["system"]
        .replace("{dimensions}", dim_lines)
        .replace("{schema_dimensions}", schema_dims)
        .replace("{max_total_score}", str(max_total))
        .replace("{tool_type}", tool_type)
        .replace("{role_label}", role_label)
    )
    return system, tmpl["user"]
