"""Tool-deliverable registry.

Each sub-package (powerbi/, notebook/) owns the full stack for one assessment format:
  - sanitizer.py / validator.py  — output validation and auto-repair
  - templates.yml                — generate_starter prompt template
  - templates_openai_patch.yml   — provider-specific overrides (optional)

To add a new tool assessment, create a new sub-package that exposes
``validate_generate_starter(starter, cfg) -> tuple[bool, str]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..entity import DomainConfig


def get_deliverable(cfg: DomainConfig) -> str:
    """Return 'powerbi', 'notebook', or 'generic' based on config."""
    tool = cfg.tool.lower()
    fmt = cfg.deliverable_format.upper()
    if "power bi" in tool or fmt == "PBIP":
        return "powerbi"
    if "python" in tool or "jupyter" in tool or "notebook" in tool or fmt == "IPYNB":
        return "notebook"
    return "generic"
