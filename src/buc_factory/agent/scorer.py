"""Score a candidate simulation against the recruiter solution and the brief's criteria."""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Any, cast

import anthropic
import yaml

from buc_factory.llm.base import BaseLLM
from buc_factory.llm.claudeai import AnthropicLLM
from buc_factory.llm.gptai import OpenAILLM

LOGGER = logging.getLogger(__name__)

_SCORER_MODEL = "claude-opus-4-7"
_SCORER_OPENAI_FALLBACK = "gpt-5.5"
_SCORER_MAX_TOKENS = 4096
_MAX_SECTION_CHARS = 12_000  # per content block sent to the LLM

_PROMPTS: dict[str, Any] | None = None


def _prompts() -> dict[str, Any]:
    global _PROMPTS
    if _PROMPTS is None:
        _PROMPTS = cast(
            dict[str, Any],
            yaml.safe_load(
                importlib.resources.files("buc_factory")
                .joinpath("conf/sim_prompt_templates.yml")
                .read_text(encoding="utf-8")
            ),
        )
    return _PROMPTS


def _truncate(text: str, max_chars: int = _MAX_SECTION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n[... {len(text) - max_chars} chars truncated ...]\n\n" + text[-half:]


def _collect_pbip_work(starter_dir: Path) -> str:
    parts: list[str] = []

    model_tmdl = starter_dir / "Assessment.SemanticModel/definition/model.tmdl"
    if model_tmdl.exists():
        parts.append(f"### model.tmdl\n```\n{_truncate(model_tmdl.read_text())}\n```")

    tables_dir = starter_dir / "Assessment.SemanticModel/definition/tables"
    if tables_dir.exists():
        for f in sorted(tables_dir.glob("*.tmdl")):
            parts.append(f"### {f.name}\n```\n{_truncate(f.read_text())}\n```")

    pages_dir = starter_dir / "Assessment.Report/pages"
    if pages_dir.exists():
        for f in sorted(pages_dir.rglob("*.json")):
            rel = f.relative_to(starter_dir / "Assessment.Report")
            parts.append(f"### {rel}\n```json\n{_truncate(f.read_text())}\n```")
    else:
        report_json = starter_dir / "Assessment.Report/report.json"
        if report_json.exists():
            parts.append(f"### report.json\n```json\n{_truncate(report_json.read_text())}\n```")

    return "\n\n".join(parts)


def _collect_ipynb_work(starter_dir: Path) -> str:
    nb = starter_dir / "notebook.ipynb"
    if not nb.exists():
        return ""
    return f"### notebook.ipynb\n```json\n{_truncate(nb.read_text())}\n```"


def score_submission(
    output_dir: Path,
    recruiter_solution: str,
    deliverable_format: str,
) -> str | None:
    """Score the simulation output against the recruiter solution.

    Returns a markdown string with the scoring table, or None on failure.
    """
    brief_path = output_dir / "brief/candidate_brief.md"
    if not brief_path.exists():
        LOGGER.warning("scoring skipped: brief/candidate_brief.md not found")
        return None

    starter_dir = output_dir / "starter"
    if deliverable_format == "IPYNB":
        sim_work = _collect_ipynb_work(starter_dir)
    else:
        sim_work = _collect_pbip_work(starter_dir)

    if not sim_work:
        LOGGER.warning("scoring skipped: no simulation work found in starter/")
        return None

    brief = _truncate(brief_path.read_text(encoding="utf-8"))
    ref = _truncate(recruiter_solution)
    p = _prompts()["scoring"]
    system = p["system"]
    user = p["user"].format(
        brief=brief,
        recruiter_solution=ref,
        sim_work=sim_work,
    )

    try:
        llm: BaseLLM = AnthropicLLM(model=_SCORER_MODEL, max_tokens=_SCORER_MAX_TOKENS)
        result = llm.complete(user, system=system, max_tokens=_SCORER_MAX_TOKENS)
        LOGGER.info("  scoring complete via %s", _SCORER_MODEL)
        return result
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
    ) as exc:
        if isinstance(exc, anthropic.BadRequestError) and "credit balance" not in str(exc).lower():
            LOGGER.error("scoring LLM call failed: %s", exc, exc_info=True)
            return None
        LOGGER.warning(
            "Anthropic unavailable for scoring (%s); falling back to %s",
            exc,
            _SCORER_OPENAI_FALLBACK,
        )
    except Exception as exc:
        LOGGER.error("scoring LLM call failed: %s", exc, exc_info=True)
        return None

    try:
        llm = OpenAILLM(model=_SCORER_OPENAI_FALLBACK, max_tokens=_SCORER_MAX_TOKENS)
        result = llm.complete(user, system=system, max_tokens=_SCORER_MAX_TOKENS)
        LOGGER.info("  scoring complete via %s", _SCORER_OPENAI_FALLBACK)
        return result or None
    except Exception as exc:
        LOGGER.error("scoring OpenAI fallback failed: %s", exc, exc_info=True)
        return None
