"""Standalone scoring engine.

Evaluates any candidate submission (manual upload or simulation output) against
a recruiter solution and brief, returning a structured ScoringResult.

The scoring is:
  - Tool-specific: different dimensions for Power BI, Python DA, and Python DS
  - Methodology-first: assesses reasoning and approach, not value correctness
  - Delivery-aware: measures structure, naming, communication maturity

Usage
-----
from buc_factory.scorer import score_submission
from buc_factory.scorer.models import ScoringResult

result = score_submission(
    output_dir=Path("..."),
    recruiter_solution="...",
    deliverable_format="PBIP",
    role="Data Analyst",
)
if result:
    print(result.to_markdown())
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import anthropic
import openai
from pydantic import ValidationError

from .models import ScoringResult, ToolType

LOGGER = logging.getLogger(__name__)

_SCORER_MODEL = "gpt-5.5"
_SCORER_ANTHROPIC_FALLBACK = "claude-opus-4-8"
_SCORER_MAX_TOKENS = 4096
_MAX_SECTION_CHARS = 12_000


# ── tool-type resolution ───────────────────────────────────────────


def _tool_type(deliverable_format: str, role: str) -> ToolType:
    if deliverable_format.upper() == "PBIP":
        return "powerbi"
    if "scientist" in role.lower():
        return "python_ds"
    return "python_da"


def _get_scorer_prompts(tool: ToolType) -> tuple[str, str]:
    """Return (system_prompt, user_template) for the given tool type."""
    if tool == "powerbi":
        from .powerbi import load_prompt as _powerbi_load_prompt

        return _powerbi_load_prompt()
    role_key: Literal["python_da", "python_ds"] = (
        "python_ds" if tool == "python_ds" else "python_da"
    )
    from .notebook import load_prompt as _notebook_load_prompt

    return _notebook_load_prompt(role_key)


# ── candidate work collection ──────────────────────────────────────


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
    # Power BI Desktop 2.154+: pages live under definition/pages/
    pages_dir = starter_dir / "Assessment.Report/definition/pages"
    if not pages_dir.exists():
        pages_dir = starter_dir / "Assessment.Report/pages"  # pre-2.154 fallback
    if pages_dir.exists():
        for f in sorted(pages_dir.rglob("*.json")):
            rel = f.relative_to(starter_dir / "Assessment.Report")
            parts.append(f"### {rel}\n```json\n{_truncate(f.read_text())}\n```")
    return "\n\n".join(parts)


def _collect_ipynb_work(starter_dir: Path) -> str:
    nb = starter_dir / "notebook.ipynb"
    if not nb.exists():
        return ""
    return f"### notebook.ipynb\n```json\n{_truncate(nb.read_text())}\n```"


# ── JSON extraction and validation ─────────────────────────────────


def _parse_response(raw: str, tool: ToolType) -> ScoringResult | None:
    """Extract a ScoringResult from a (potentially noisy) LLM response."""
    text = raw.strip()

    # Strip markdown code fence if the model wrapped the JSON
    if text.startswith("```"):
        inner = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if inner:
            text = inner.group(1).strip()

    # Try direct parse first, then locate the outermost JSON object
    for candidate in [text, _extract_json_object(text)]:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            data["tool_type"] = tool  # enforce correct tool_type regardless of LLM output
            return ScoringResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError):
            continue

    LOGGER.warning("scorer: could not parse LLM response as ScoringResult")
    return None


def _extract_json_object(text: str) -> str | None:
    """Find the first complete {...} JSON object in text using brace counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ── LLM call (raw SDKs — reasoning model manages its own determinism) ─
#
# GPT-5.5 is a reasoning model: it does not accept temperature and is
# internally consistent on the same input.  We call it directly so we
# stay outside the BaseLLM abstraction and avoid any LangChain overhead.
# Fallback: Claude Sonnet (different provider → better availability).


def _openai_call(user: str, system: str) -> str:
    client = openai.OpenAI()
    return (
        client.chat.completions.create(
            model=_SCORER_MODEL,
            max_completion_tokens=_SCORER_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        .choices[0]
        .message.content
        or ""
    )


def _anthropic_fallback_call(user: str, system: str) -> str:
    """Fallback: Claude Sonnet with temperature=0 for consistency."""
    client = anthropic.Anthropic()
    kwargs: dict[str, Any] = {
        "model": _SCORER_ANTHROPIC_FALLBACK,
        "max_tokens": _SCORER_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        return str(client.messages.create(temperature=0, **kwargs).content[0].text)
    except anthropic.BadRequestError as exc:
        if "temperature" in str(exc).lower():
            return str(client.messages.create(**kwargs).content[0].text)
        raise


def _call_llm(user: str, system: str) -> str | None:
    try:
        result = _openai_call(user, system)
        LOGGER.info("scoring complete via %s", _SCORER_MODEL)
        return result or None
    except openai.AuthenticationError:
        LOGGER.warning(
            "OpenAI unavailable for scoring; falling back to %s", _SCORER_ANTHROPIC_FALLBACK
        )
    except Exception as exc:
        LOGGER.error("scoring LLM call failed: %s", exc, exc_info=True)
        return None

    try:
        result = _anthropic_fallback_call(user, system)
        LOGGER.info("scoring complete via %s (fallback)", _SCORER_ANTHROPIC_FALLBACK)
        return result or None
    except Exception as exc:
        LOGGER.error("scoring Anthropic fallback failed: %s", exc, exc_info=True)
        return None


# ── public API ─────────────────────────────────────────────────────


def score_submission(
    output_dir: Path,
    recruiter_solution: str,
    deliverable_format: str,
    role: str = "",
) -> ScoringResult | None:
    """Score a candidate submission against the recruiter solution.

    Works identically for manual uploads and simulation outputs.
    Returns a structured ScoringResult, or None on failure.

    Parameters
    ----------
    output_dir:           run/simulation workspace root (must contain brief/)
    recruiter_solution:   full text of the recruiter answer key
    deliverable_format:   "PBIP" or "IPYNB"
    role:                 candidate role string — used to distinguish DA vs DS for IPYNB
    """
    brief_path = output_dir / "brief/candidate_brief.md"
    if not brief_path.exists():
        LOGGER.warning("scoring skipped: brief/candidate_brief.md not found")
        return None

    starter_dir = output_dir / "starter"
    submission_work = (
        _collect_ipynb_work(starter_dir)
        if deliverable_format.upper() == "IPYNB"
        else _collect_pbip_work(starter_dir)
    )
    if not submission_work:
        LOGGER.warning("scoring skipped: no candidate work found in starter/")
        return None

    tool = _tool_type(deliverable_format, role)
    system, user_tmpl = _get_scorer_prompts(tool)

    role_label = "Data Scientist" if tool == "python_ds" else "Data Analyst"
    user = user_tmpl.replace("{role_label}", role_label).format(
        brief=_truncate(brief_path.read_text(encoding="utf-8")),
        recruiter_solution=_truncate(recruiter_solution),
        submission_work=submission_work,
    )

    raw = _call_llm(user, system)
    if raw is None:
        return None
    return _parse_response(raw, tool)
