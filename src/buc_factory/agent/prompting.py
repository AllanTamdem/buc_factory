import importlib.resources
import json
from typing import Any, cast

import yaml

from ..utils import fmt
from .deliverables import get_deliverable
from .entity import DomainConfig

_PROMPTS_CACHE: dict[str, dict[str, Any]] = {}
_DELIVERABLE_CACHE: dict[str, dict[str, Any]] = {}


def _load_yaml(fname: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.safe_load(
            importlib.resources.files("buc_factory")
            .joinpath(f"conf/{fname}")
            .read_text(encoding="utf-8")
        ),
    )


def _load_deliverable_yaml(deliverable: str, filename: str) -> dict[str, Any]:
    pkg = f"buc_factory.agent.deliverables.{deliverable}"
    try:
        content = importlib.resources.files(pkg).joinpath(filename).read_text(encoding="utf-8")
        return cast(dict[str, Any], yaml.safe_load(content))
    except Exception:
        return {}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, val in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _prompts(provider: str = "claude") -> dict[str, Any]:
    if provider not in _PROMPTS_CACHE:
        base = _load_yaml("prompt_templates.yml")
        if provider != "claude":
            try:
                patch = _load_yaml(f"prompt_templates_{provider}_patch.yml")
                base = _deep_merge(base, patch)
            except Exception:
                pass
        _PROMPTS_CACHE[provider] = base
    return _PROMPTS_CACHE[provider]


def _deliverable_templates(deliverable: str, provider: str = "claude") -> dict[str, Any]:
    """Load and cache the generate_starter templates for a deliverable subpackage."""
    cache_key = f"{deliverable}:{provider}"
    if cache_key not in _DELIVERABLE_CACHE:
        base = _load_deliverable_yaml(deliverable, "templates.yml")
        if provider != "claude":
            patch = _load_deliverable_yaml(deliverable, f"templates_{provider}_patch.yml")
            if patch:
                base = _deep_merge(base, patch)
        _DELIVERABLE_CACHE[cache_key] = base
    return _DELIVERABLE_CACHE[cache_key]


# ── Domain-aware system prompt ─────────────────────────────────────


def raw_system_template(provider: str = "claude") -> str:
    """Return the unrendered system prompt template (for versioning in MLflow)."""
    return str(_prompts(provider)["system"])


def build_system(cfg: DomainConfig, provider: str = "claude") -> str:
    return fmt(_prompts(provider)["system"], **vars(cfg))


# ── Tool classification helpers ────────────────────────────────────


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


# ── Sub-task prompts ───────────────────────────────────────────────


def _bootstrap_prompt(cfg: DomainConfig, p: dict[str, Any]) -> str:
    """Build the bootstrap_domain prompt for the infer (partial/no-config) path only."""
    parts = [p["infer_preamble"]]
    if cfg.dimensions is None:
        parts.append(p["infer_dims_section"])
    if cfg.entities is None:
        parts.append(p["infer_entities_section"])
    parts.append(p["infer_postamble"])
    return fmt(
        "\n\n".join(parts),
        industry=cfg.industry,
        location=cfg.location,
        use_dims_note="Use the dimensions from config." if cfg.dimensions is not None else "",
        use_entities_note="Use the entities from config." if cfg.entities is not None else "",
    )


def _starter_prompt(cfg: DomainConfig, provider: str, data_schema_json: str | None = None) -> str:
    """Return the generate_starter prompt from the appropriate deliverable templates."""
    schema = data_schema_json or "(not yet available)"
    deliverable = get_deliverable(cfg)
    tmpl = _deliverable_templates(deliverable, provider)

    if deliverable == "powerbi":
        return fmt(tmpl["power_bi"], data_schema_json=schema)
    if deliverable == "notebook":
        key = "python_ds" if "scientist" in cfg.role.lower() else "python_da"
        return fmt(tmpl[key], data_schema_json=schema)
    return fmt(tmpl.get("generic", ""), tool=cfg.tool, deliverable_format=cfg.deliverable_format)


def task_prompt(
    name: str,
    cfg: DomainConfig,
    retry_feedback: str | None,
    rolled: dict[str, Any] | None = None,
    scenario_json: str | None = None,
    bootstrap_json: str | None = None,
    data_schema_json: str | None = None,
    candidate_brief: str | None = None,
    provider: str = "claude",
) -> str:
    raw = _prompts(provider)
    p = raw["tasks"]
    ctx = vars(cfg)
    _na = "(not yet available)"
    prompts = {
        "bootstrap_domain": _bootstrap_prompt(cfg, p["bootstrap_domain"]),
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
        "generate_starter": _starter_prompt(cfg, provider, data_schema_json),
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
        prompt += "\n\n" + fmt(raw["retry_suffix"], retry_feedback=retry_feedback)
    return prompt
