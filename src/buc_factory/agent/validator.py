import json
from pathlib import Path

from .entity import DomainConfig
from .prompting import dax_or_calc

MAX_RETRIES_PER_TASK = 3
TASKS_WITH_RUN_PYTHON = {"generate_data_script"}

# ──────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────


def validate_and_extract(
    name: str, cfg: DomainConfig, output_dir: Path, state: dict
) -> tuple[bool, str, dict]:
    """Validate a completed subtask; return (ok, message, state_updates)."""
    try:
        if name == "bootstrap_domain":
            data = json.loads((output_dir / "bootstrap.json").read_text())
            if not data.get("dimensions") or not data.get("entities"):
                msg = "bootstrap.json must contain non-empty 'dimensions' and 'entities'"
                return False, msg, {}
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
            missing = expected - data.keys()
            if missing:
                return False, f"scenario.json missing dimensions: {missing}", {}
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
            entities = set(state.get("bootstrapped_entities") or cfg.entities or [])
            overlap = len(files_in_schema & entities)
            if overlap < max(3, int(0.8 * len(entities))):
                return False, f"schema covers only {overlap}/{len(entities)} entities", {}
            if not schema.get("traps"):
                return False, "schema must declare at least one trap", {}
            return True, f"{overlap} entities, {len(schema['traps'])} traps", {}

        if name == "generate_data_script":
            csv_dir = output_dir / "starter/data"
            entities = state.get("bootstrapped_entities") or cfg.entities or []
            for entity in entities:
                if not (csv_dir / f"{entity}.csv").exists():
                    return False, f"{entity}.csv not generated", {}
            if len(entities) >= 2:
                import pandas as pd

                fact = pd.read_csv(csv_dir / f"{entities[0]}.csv", sep=None, engine="python")
                dim = pd.read_csv(csv_dir / f"{entities[1]}.csv", sep=None, engine="python")
                fk = next(
                    (c for c in fact.columns if entities[1].rstrip("s").lower() in c.lower()), None
                )
                pk = next((c for c in dim.columns if "id" in c.lower()), None)
                if fk and pk:
                    orphans = set(fact[fk].dropna()) - set(dim[pk])
                    if orphans:
                        return (
                            False,
                            f"FK violated: {len(orphans)} orphans "
                            f"{entities[0]}.{fk}→{entities[1]}.{pk}",
                            {},
                        )
            return True, "CSVs generated, FK heuristic passes", {}

        if name == "generate_starter":
            starter = output_dir / "starter"
            fmt = cfg.deliverable_format.upper()
            if fmt == "PBIP":
                required = [
                    "Assessment.pbip",
                    "Assessment.SemanticModel/definition.pbism",
                    "Assessment.SemanticModel/definition/model.tmdl",
                    "Assessment.Report/definition.pbir",
                    "Assessment.Report/report.json",
                ]
            elif fmt == "TWBX":
                required = ["Assessment.tds", "Assessment.twb"]
            else:
                files = list(starter.rglob("*")) if starter.exists() else []
                if len(files) < 2:
                    return False, "starter looks empty", {}
                return True, f"{len(files)} files (generic check)", {}
            for r in required:
                if not (starter / r).exists():
                    return False, f"missing {r}", {}
                if r.endswith((".json", ".pbir", ".pbip", ".pbism")):
                    try:
                        json.loads((starter / r).read_text())
                    except json.JSONDecodeError as e:
                        return False, f"{r} invalid JSON: {e}", {}
            pbism_path = starter / "Assessment.SemanticModel/definition.pbism"
            if fmt == "PBIP" and pbism_path.exists():
                pbism = json.loads(pbism_path.read_text())
                if pbism.get("version") != "4.0":
                    return (
                        False,
                        f"definition.pbism version must be '4.0', got '{pbism.get('version')}'",
                        {},
                    )
            return True, f"{fmt} starter OK", {}

        if name == "write_recruiter_solution":
            f = output_dir / "solution/recruiter_solution.md"
            if not f.exists():
                return False, "solution not created", {}
            content = f.read_text()
            if len(content) < 3000:
                return False, f"solution too short ({len(content)} chars)", {}
            calc_kw = dax_or_calc(cfg.tool).split()[0]
            if calc_kw.lower() not in content.lower():
                return False, f"solution missing {calc_kw} content", {}
            return True, "solution OK", {}

        if name == "final_assembly":
            expected = [
                "scenario.json",
                "bootstrap.json",
                "brief/candidate_brief.md",
                "brief/data_schema.json",
                "solution/recruiter_solution.md",
                "starter/generate_data.py",
            ]
            missing = [p for p in expected if not (output_dir / p).exists()]
            if missing:
                return False, f"missing: {missing}", {}
            return True, "all artifacts present", {}

        return False, f"unknown task {name}", {}
    except Exception as e:
        return False, f"validator crashed: {type(e).__name__}: {e}", {}
