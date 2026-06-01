"""Simulation output validation for Jupyter Notebook (IPYNB) starter projects."""

from __future__ import annotations

import json
from pathlib import Path


def validate_simulation(starter_dir: Path) -> tuple[bool, str]:
    """Validate a simulated Jupyter Notebook submission.

    Intentionally lenient: checks structure and basic validity only.
    A junior candidate is allowed to have fewer than the full cell count.
    """
    notebook = starter_dir / "notebook.ipynb"
    if not notebook.exists():
        return False, "starter/notebook.ipynb not found"
    try:
        nb = json.loads(notebook.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, f"notebook.ipynb is not valid JSON: {e}"
    if nb.get("nbformat") != 4:
        return False, "notebook.ipynb must be nbformat 4"
    if len(nb.get("cells", [])) < 3:
        return False, "notebook.ipynb has fewer than 3 cells — does not appear modified"
    return True, "simulation output validated (IPYNB)"
