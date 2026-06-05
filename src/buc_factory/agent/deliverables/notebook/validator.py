"""Structural validation for Jupyter Notebook (IPYNB) generate_starter output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....agent.entity import DomainConfig


def validate_generate_starter(starter: Path, _cfg: DomainConfig) -> tuple[bool, str]:
    """Validate a generated Jupyter Notebook starter project.

    Checks that notebook.ipynb and requirements.txt exist, the notebook is
    valid nbformat 4, and it has enough cells to be considered non-empty.
    Returns (ok, message).
    """
    nb_path = starter / "notebook.ipynb"
    req_path = starter / "requirements.txt"
    if not nb_path.exists():
        return False, "missing starter/notebook.ipynb"
    if not req_path.exists():
        return False, "missing starter/requirements.txt"
    try:
        nb = json.loads(nb_path.read_text())
    except json.JSONDecodeError as e:
        return False, f"notebook.ipynb invalid JSON: {e}"
    if nb.get("nbformat") != 4:
        return False, "notebook.ipynb must be nbformat 4"
    cells = nb.get("cells", [])
    if len(cells) < 5:
        return False, f"notebook too sparse ({len(cells)} cells)"
    return True, f"Python IPYNB starter OK ({len(cells)} cells)"


def validate_generate_starter_generic(starter: Path, _cfg: DomainConfig) -> tuple[bool, str]:
    """Fallback validator for non-PBIP, non-notebook tool deliverables."""
    files = list(starter.rglob("*")) if starter.exists() else []
    if len(files) < 2:
        return False, "starter looks empty"
    return True, f"{len(files)} files (generic check)"
