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


def dax_or_calc(tool: str) -> str:
    t = tool.lower()
    if "power bi" in t:
        return "DAX"
    if "tableau" in t:
        return "Tableau calculation"
    if "looker" in t:
        return "LookML measure"
    return "calculation"


def _prep_layer(tool: str) -> str:
    t = tool.lower()
    if "power bi" in t:
        return "Power Query M"
    if "tableau" in t:
        return "Tableau Prep / data source filters"
    if "looker" in t:
        return "PDT / derived tables"
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


def _starter_prompt(cfg: DomainConfig) -> str:
    p = _PROMPTS["tasks"]["generate_starter"]
    tool = cfg.tool.lower()
    fmt = cfg.deliverable_format.upper()

    if "power bi" in tool or fmt == "PBIP":
        return p["power_bi"]
    if "tableau" in tool or fmt == "TWBX":
        return p["tableau"]
    if "looker" in tool or "lookml" in fmt.lower():
        return p["looker"]
    return fmt(p["generic"], tool=cfg.tool, deliverable_format=cfg.deliverable_format)


def task_prompt(name: str, cfg: DomainConfig, retry_feedback: str | None) -> str:
    p = _PROMPTS["tasks"]
    ctx = vars(cfg)
    prompts = {
        "bootstrap_domain": _bootstrap_prompt(cfg),
        "roll_scenario": p["roll_scenario"],
        "write_brief": fmt(p["write_brief"], **ctx),
        "design_data_schema": fmt(p["design_data_schema"], **ctx),
        "generate_data_script": p["generate_data_script"],
        "generate_starter": _starter_prompt(cfg),
        "write_recruiter_solution": fmt(
            p["write_recruiter_solution"],
            **ctx,
            dax_or_calc=dax_or_calc(cfg.tool),
            prep_layer=_prep_layer(cfg.tool),
        ),
        "final_assembly": fmt(p["final_assembly"], **ctx),
    }
    prompt = prompts[name]
    if retry_feedback:
        prompt += "\n\n" + fmt(_PROMPTS["retry_suffix"], retry_feedback=retry_feedback)
    return prompt
