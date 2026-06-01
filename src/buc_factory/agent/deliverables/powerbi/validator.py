"""Structural validation for Power BI PBIP generate_starter output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .sanitizer import sanitize_pbip_starter

if TYPE_CHECKING:
    from ....agent.entity import DomainConfig

LOGGER = logging.getLogger(__name__)


def validate_generate_starter(starter: Path, _cfg: DomainConfig) -> tuple[bool, str]:
    """Validate and auto-repair a generated PBIP starter project.

    Runs the 6-phase sanitizer first, then checks that all required files
    exist and contain valid JSON.  Returns (ok, message).
    """
    sanitize_fixes = sanitize_pbip_starter(starter)
    if sanitize_fixes:
        LOGGER.info("PBIP sanitizer applied %d fix(es): %s", len(sanitize_fixes), sanitize_fixes)

    required_json = [
        "Assessment.pbip",
        "Assessment.SemanticModel/definition.pbism",
        "Assessment.SemanticModel/definition/model.tmdl",
        "Assessment.SemanticModel/definition/expressions/DataPath.tmdl",
        "Assessment.Report/definition.pbir",
        "Assessment.Report/definition/version.json",
        "Assessment.Report/definition/report.json",
    ]
    for r in required_json:
        if not (starter / r).exists():
            return False, f"missing {r}"
        if r.endswith((".json", ".pbir", ".pbip", ".pbism")):
            try:
                json.loads((starter / r).read_text())
            except json.JSONDecodeError as e:
                return False, f"{r} invalid JSON: {e}"

    pbism_path = starter / "Assessment.SemanticModel/definition.pbism"
    if pbism_path.exists():
        pbism = json.loads(pbism_path.read_text())
        if pbism.get("version") != "4.2":
            return (
                False,
                f"definition.pbism version must be '4.2', got '{pbism.get('version')}'",
            )

    pbir_path = starter / "Assessment.Report/definition.pbir"
    if pbir_path.exists():
        pbir = json.loads(pbir_path.read_text())
        if pbir.get("version") not in ("4.0", "1.0"):
            return False, f"definition.pbir version unexpected: '{pbir.get('version')}'"

    pages_json_path = starter / "Assessment.Report/definition/pages/pages.json"
    def_report_path = starter / "Assessment.Report/definition/report.json"
    if not pages_json_path.exists():
        return False, "missing Assessment.Report/definition/pages/pages.json"
    try:
        json.loads(pages_json_path.read_text())
    except json.JSONDecodeError as e:
        return False, f"definition/pages/pages.json invalid JSON: {e}"
    try:
        json.loads(def_report_path.read_text())
    except json.JSONDecodeError as e:
        return False, f"definition/report.json invalid JSON: {e}"

    return True, "PBIP starter OK (PBIR 2.154+)"
