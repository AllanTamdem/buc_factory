"""Per-task output validation dispatcher.

Generic task validators live here (bootstrap, roll_scenario, write_brief, …).
Tool-specific generate_starter validation is delegated to the appropriate
deliverable sub-package under agent/deliverables/.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .deliverables import get_deliverable
from .deliverables.notebook import (
    validate_generate_starter as _validate_notebook_starter,
)
from .deliverables.notebook import (
    validate_generate_starter_generic,
)
from .deliverables.powerbi import (
    check_relationship_coherence,
)
from .deliverables.powerbi import (
    validate_generate_starter as _validate_pbip_starter,
)
from .entity import DomainConfig
from .prompting import _is_python, calc_language

LOGGER = logging.getLogger(__name__)

MAX_RETRIES_PER_TASK = 3
TASKS_WITH_RUN_PYTHON = {"generate_data_script"}


def validate_and_extract(
    name: str, cfg: DomainConfig, output_dir: Path, state: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Validate a completed subtask; return (ok, message, state_updates)."""
    try:
        if name == "bootstrap_domain":
            data = json.loads((output_dir / "bootstrap.json").read_text())
            if not data.get("dimensions") or not data.get("entities"):
                return (
                    False,
                    "bootstrap.json must contain non-empty 'dimensions' and 'entities'",
                    {},
                )
            if len(data["entities"]) < 3:
                return False, f"need at least 3 entities, got {len(data['entities'])}", {}
            updates = {
                "bootstrapped_dimensions": data["dimensions"],
                "bootstrapped_entities": data["entities"],
            }
            msg = f"{len(data['dimensions'])} dimensions, {len(data['entities'])} entities"
            return True, msg, updates

        if name == "roll_scenario":
            data = json.loads((output_dir / "scenario.json").read_text())
            expected = set((state.get("bootstrapped_dimensions") or cfg.dimensions or {}).keys())
            missing_dims = expected - data.keys()
            if missing_dims:
                return False, f"scenario.json missing dimensions: {missing_dims}", {}
            return True, "scenario rolled", {"scenario": data}

        if name == "write_brief":
            f = output_dir / "brief/candidate_brief.md"
            if not f.exists():
                return False, "brief not created", {}
            content = f.read_text()
            if len(content) < 1500:
                return False, f"brief too short ({len(content)} chars)", {}
            return True, "brief OK", {}

        if name == "design_data_schema":
            schema = json.loads((output_dir / "brief/data_schema.json").read_text())
            files_in_schema = {f["filename"].replace(".csv", "") for f in schema.get("files", [])}
            schema_entities = set(state.get("bootstrapped_entities") or cfg.entities or [])
            overlap = len(files_in_schema & schema_entities)
            if overlap < max(3, int(0.8 * len(schema_entities))):
                return False, f"schema covers only {overlap}/{len(schema_entities)} entities", {}
            if not schema.get("traps"):
                return False, "schema must declare at least one trap", {}
            ok, rel_err = check_relationship_coherence(schema)
            if not ok:
                return False, rel_err, {}
            return True, f"{overlap} entities, {len(schema['traps'])} traps", {}

        if name == "generate_data_script":
            csv_dir = output_dir / "starter/data"
            csv_entities: list[str] = list(state.get("bootstrapped_entities") or cfg.entities or [])
            for entity in csv_entities:
                if not (csv_dir / f"{entity}.csv").exists():
                    return False, f"{entity}.csv not generated", {}
            if len(csv_entities) >= 2:
                import pandas as pd

                fact = pd.read_csv(str(csv_dir / f"{csv_entities[0]}.csv"))
                dim = pd.read_csv(str(csv_dir / f"{csv_entities[1]}.csv"))
                fk = next(
                    (c for c in fact.columns if csv_entities[1].rstrip("s").lower() in c.lower()),
                    None,
                )
                pk = next((c for c in dim.columns if "id" in c.lower()), None)
                if fk and pk:
                    orphans = set(fact[fk].dropna()) - set(dim[pk])
                    if orphans:
                        return (
                            False,
                            f"FK violated: {len(orphans)} orphans "
                            f"{csv_entities[0]}.{fk}→{csv_entities[1]}.{pk}",
                            {},
                        )
            return True, "CSVs generated, FK heuristic passes", {}

        if name == "generate_starter":
            starter = output_dir / "starter"
            deliverable = get_deliverable(cfg)
            if deliverable == "powerbi":
                ok, msg = _validate_pbip_starter(starter, cfg)
            elif deliverable == "notebook" or _is_python(cfg.tool):
                ok, msg = _validate_notebook_starter(starter, cfg)
            else:
                ok, msg = validate_generate_starter_generic(starter, cfg)
            return ok, msg, {}

        if name == "write_recruiter_solution":
            f = output_dir / "solution/recruiter_solution.md"
            if not f.exists():
                return False, "solution not created", {}
            content = f.read_text()
            if len(content) < 3000:
                return False, f"solution too short ({len(content)} chars)", {}
            calc_kw = calc_language(cfg.tool).split()[0]
            if calc_kw.lower() not in content.lower():
                return False, f"solution missing {calc_kw} content", {}
            return True, "solution OK", {}

        if name == "final_assembly":
            required_files = [
                "scenario.json",
                "bootstrap.json",
                "brief/candidate_brief.md",
                "brief/data_schema.json",
                "solution/recruiter_solution.md",
                "starter/generate_data.py",
            ]
            absent_files = [p for p in required_files if not (output_dir / p).exists()]
            if absent_files:
                return False, f"missing: {absent_files}", {}
            return True, "all artifacts present", {}

        return False, f"unknown task {name}", {}
    except Exception as e:
        return False, f"validator crashed: {type(e).__name__}: {e}", {}
