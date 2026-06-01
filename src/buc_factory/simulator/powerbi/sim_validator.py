"""Simulation output validation for Power BI PBIP starter projects.

Intentionally lenient about content completeness — a junior candidate may leave
measures or visuals incomplete by design.  Structural integrity is all we enforce.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from buc_factory.agent.deliverables.powerbi import (
    check_graph_ambiguity,
    parse_tmdl_relationships,
    sanitize_pbip_starter,
)

LOGGER = logging.getLogger(__name__)


def validate_simulation(starter_dir: Path) -> tuple[bool, str]:
    """Sanitise and structurally validate a simulated PBIP submission.

    Applies the 6-phase sanitiser first (same as the generator), then checks:
    - At least one .pbip file exists
    - TMDL files or report pages exist (project was actually modified)
    - Critical JSON files parse correctly
    - visual.json config strings are valid nested JSON
    - model.tmdl relationship graph has no ambiguous paths
    """
    if not list(starter_dir.glob("*.pbip")):
        return False, "No .pbip file found in starter/"

    fixes = sanitize_pbip_starter(starter_dir)
    if fixes:
        LOGGER.info("Simulator PBIP sanitizer applied %d fix(es): %s", len(fixes), fixes)

    has_tmdl = bool(list(starter_dir.rglob("*.tmdl")))
    has_report = (starter_dir / "Assessment.Report/definition/pages/pages.json").exists() or (
        starter_dir / "Assessment.Report/pages/pages.json"
    ).exists()
    if not has_tmdl and not has_report:
        return False, "No TMDL or report files in starter/ — project was not modified"

    for rel in (
        "Assessment.SemanticModel/definition.pbism",
        "Assessment.Report/definition.pbir",
        "Assessment.Report/definition/pages/pages.json",
    ):
        p = starter_dir / rel
        if p.exists():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                return False, f"{rel} contains invalid JSON after your edits: {e}"

    # PBIR 1.0.0 visual.json validation:
    # - file must be valid JSON
    # - must not have invalid root-level fields (additionalProperties: false)
    # - data visuals should have visual.query (warn → fail if placeholder is the only visual)
    _VALID_VISUAL_KEYS = {
        "$schema",
        "name",
        "position",
        "visual",
        "visualGroup",
        "parentGroupName",
        "filterConfig",
        "isHidden",
        "annotations",
        "howCreated",
    }
    _PLACEHOLDER_NAMES = {"starter_placeholder"}
    report_dir = starter_dir / "Assessment.Report"
    if report_dir.exists():
        visuals_without_query: list[str] = []
        for vis in report_dir.rglob("visual.json"):
            rel = str(vis.relative_to(starter_dir))
            try:
                vis_data = json.loads(vis.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                return False, f"{rel} contains invalid JSON: {e}"

            # Reject invalid root-level fields left by old-format generators.
            invalid = [k for k in vis_data if k not in _VALID_VISUAL_KEYS]
            if invalid:
                return (
                    False,
                    f"{rel} contains invalid root field(s) {invalid}. "
                    f"The PBIR 1.0.0 schema uses 'visual.query.queryState' for field bindings — "
                    f"'config', 'filters', 'tabOrder' (root), and 'displayState' are not allowed.",
                )

            # Warn (don't fail) when a non-placeholder visual has no query bindings.
            vis_name = vis_data.get("name", "")
            vis_obj = vis_data.get("visual", {})
            if vis_name not in _PLACEHOLDER_NAMES and vis_obj and not vis_obj.get("query"):
                visuals_without_query.append(vis_name or rel)

        if visuals_without_query:
            LOGGER.warning(
                "Visuals with no field/measure bindings (visual.query missing): %s",
                visuals_without_query,
            )

    model_tmdl_path = starter_dir / "Assessment.SemanticModel" / "definition" / "model.tmdl"
    if model_tmdl_path.exists():
        rel_graph = parse_tmdl_relationships(model_tmdl_path.read_text(encoding="utf-8"))
        if rel_graph:
            ok, rel_err = check_graph_ambiguity(rel_graph)
            if not ok:
                return False, rel_err

    # Check table TMDL files for the two most common LLM TMDL syntax mistakes.
    _OBJECT_SYNTAX_RE = re.compile(r"""^\s*measure\s+["'][^"'\n]+["']\s*\{""", re.MULTILINE)
    _UNINDENTED_PROP_RE = re.compile(r"^\t?(formatString|displayFolder):", re.MULTILINE)

    tables_dir = starter_dir / "Assessment.SemanticModel/definition/tables"
    if tables_dir.exists():
        for tmdl in sorted(tables_dir.glob("*.tmdl")):
            text = tmdl.read_text(encoding="utf-8")

            # Bug 1 — object syntax: measure "X" { expression: ... }
            # Power BI 2.154 declarative TMDL never uses curly braces.
            if _OBJECT_SYNTAX_RE.search(text):
                return (
                    False,
                    f"{tmdl.name}: measure uses object syntax (curly braces) — "
                    f"Power BI 2.154 uses declarative TMDL without braces. "
                    f"Replace with:\n"
                    f"\tmeasure 'Name' = <DAX>\n"
                    f"\t\tformatString: 0.00%\n"
                    f"\t\tdisplayFolder: KPIs",
                )

            # Bug 2 — formatString/displayFolder at 0 or 1 tab.
            # Measure properties must be at 2 tabs (measure is at 1 tab inside the table block).
            m = _UNINDENTED_PROP_RE.search(text)
            if m:
                prop = m.group(1)
                n_tabs = len(m.group(0)) - len(m.group(1)) - 1
                return (
                    False,
                    f"{tmdl.name}: '{prop}:' is at {n_tabs} tab(s) — "
                    f"it must be at 2 tabs as a sub-property of the measure. "
                    f"Correct pattern:\n"
                    f"\tmeasure 'Name' =\n"
                    f"\t\t\t<DAX formula>\n"
                    f"\t\t{prop}: <value>",
                )

    # Check every page listed in pages.json has at least one visual.
    pages_root = starter_dir / "Assessment.Report/definition/pages"
    if pages_root.exists():
        pages_json = pages_root / "pages.json"
        if pages_json.exists():
            try:
                idx = json.loads(pages_json.read_text(encoding="utf-8"))
                page_ids = idx.get("pageOrder") or [
                    p["name"] for p in idx.get("pages", []) if "name" in p
                ]
                for pid in page_ids:
                    visuals_dir = pages_root / pid / "visuals"
                    has_visuals = visuals_dir.exists() and any(
                        p.is_dir() for p in visuals_dir.iterdir()
                    )
                    if not has_visuals:
                        return (
                            False,
                            f"Page '{pid}' has no visuals. "
                            f"Each page must contain at least one visual.json under "
                            f"definition/pages/{pid}/visuals/<visual_id>/visual.json.",
                        )
            except (json.JSONDecodeError, OSError, KeyError):
                pass

    return True, "simulation output validated (PBIP)"
