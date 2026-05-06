import json
from pathlib import Path

import yaml

from ..utils import fmt
from .entity import DomainConfig

_PROJECT_DIR = Path(__file__).parent.parent.parent.parent
_PROMPTS = yaml.safe_load(
    (_PROJECT_DIR / "conf" / "prompt_templates.yml").read_text(encoding="utf-8")
)


# ──────────────────────────────────────────────────────────────────
# Domain-aware system prompt
# ──────────────────────────────────────────────────────────────────


def build_system(cfg: DomainConfig) -> str:
    return fmt(_PROMPTS["system"], **vars(cfg))


# ──────────────────────────────────────────────────────────────────
# Sub-task prompts
# ──────────────────────────────────────────────────────────────────


def _is_python(tool: str) -> bool:
    t = tool.lower()
    return "python" in t or "jupyter" in t or "notebook" in t


def calc_language(tool: str) -> str:
    t = tool.lower()
    if "power bi" in t:
        return "DAX"
    if _is_python(tool):
        return "Python (pandas / matplotlib / sklearn)"
    return "calculation"


def _prep_layer(tool: str) -> str:
    t = tool.lower()
    if "power bi" in t:
        return "Power Query M"
    if _is_python(tool):
        return "pandas ETL / feature engineering pipeline"
    return "data preparation layer"


def _bootstrap_prompt(cfg: DomainConfig) -> str:
    p = _PROMPTS["tasks"]["bootstrap_domain"]
    has_dims = cfg.dimensions is not None
    has_entities = cfg.entities is not None

    if has_dims and has_entities:
        return fmt(
            p["fully_specified"],
            dimensions_json=json.dumps(cfg.dimensions, ensure_ascii=False),
            entities_json=json.dumps(cfg.entities, ensure_ascii=False),
        )

    parts = [p["infer_preamble"]]
    if not has_dims:
        parts.append(p["infer_dims_section"])
    if not has_entities:
        parts.append(p["infer_entities_section"])
    parts.append(p["infer_postamble"])
    return fmt(
        "\n\n".join(parts),
        industry=cfg.industry,
        location=cfg.location,
        use_dims_note="Use the dimensions from config." if has_dims else "",
        use_entities_note="Use the entities from config." if has_entities else "",
    )


def _starter_prompt(cfg: DomainConfig, data_schema_json: str | None = None) -> str:
    p = _PROMPTS["tasks"]["generate_starter"]
    tool = cfg.tool.lower()
    deliverable_fmt = cfg.deliverable_format.upper()
    schema = data_schema_json or "(not yet available)"

    if "power bi" in tool or deliverable_fmt == "PBIP":
        return fmt(p["power_bi"], data_schema_json=schema)
    if _is_python(cfg.tool) or deliverable_fmt == "IPYNB":
        key = "python_ds" if "scientist" in cfg.role.lower() else "python_da"
        return fmt(p[key], data_schema_json=schema)
    return fmt(p["generic"], tool=cfg.tool, deliverable_format=cfg.deliverable_format)


def task_prompt(
    name: str,
    cfg: DomainConfig,
    retry_feedback: str | None,
    rolled: dict | None = None,
    scenario_json: str | None = None,
    bootstrap_json: str | None = None,
    data_schema_json: str | None = None,
    candidate_brief: str | None = None,
) -> str:
    p = _PROMPTS["tasks"]
    ctx = vars(cfg)
    _na = "(not yet available)"
    prompts = {
        "bootstrap_domain": _bootstrap_prompt(cfg),
        "roll_scenario": fmt(
            p["roll_scenario"],
            rolled_json=json.dumps(rolled, ensure_ascii=False, indent=2),
        )
        if rolled
        else p["roll_scenario"],
        "write_brief": fmt(
            p["write_brief"],
            **ctx,
            scenario_json=scenario_json or _na,
            bootstrap_json=bootstrap_json or _na,
        ),
        "design_data_schema": fmt(
            p["design_data_schema"],
            **ctx,
            bootstrap_json=bootstrap_json or _na,
        ),
        "generate_data_script": fmt(
            p["generate_data_script"],
            **ctx,
            data_schema_json=data_schema_json or _na,
            candidate_brief=candidate_brief or _na,
        ),
        "generate_starter": _starter_prompt(cfg, data_schema_json),
        "write_recruiter_solution": fmt(
            p["write_recruiter_solution"],
            **ctx,
            calc_language=calc_language(cfg.tool),
            prep_layer=_prep_layer(cfg.tool),
            scenario_json=scenario_json or _na,
            data_schema_json=data_schema_json or _na,
            candidate_brief=candidate_brief or _na,
        ),
        "final_assembly": fmt(p["final_assembly"], **ctx),
    }
    prompt = prompts[name]
    if retry_feedback:
        prompt += "\n\n" + fmt(_PROMPTS["retry_suffix"], retry_feedback=retry_feedback)
    return prompt
