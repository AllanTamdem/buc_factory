"""LangGraph for simulating a candidate completing a technical assessment.

Graph topology:

    prepare_simulate → run_simulate → validate_simulate ──► END         (success)
                                                        ├──► prepare_simulate  (retry)
                                                        └──► fail_simulate ──► END
"""

from __future__ import annotations

import importlib.resources
import logging
import time
from pathlib import Path
from typing import Literal, TypedDict

import anthropic as _anthropic
import yaml
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..llm.claudeai import AnthropicLLM
from ..llm.gptai import OpenAILLM
from ..utils import fmt
from .tool import make_tools

LOGGER = logging.getLogger(__name__)

_SIM_MODEL = "claude-sonnet-4-6"
_SIM_OPENAI_FALLBACK = "gpt-5.4"
_SIM_MAX_TOKENS = 16_000
_SIM_MAX_RETRIES = 3

# Proficiency tier boundaries
_TIER_EXPERT = 0.9
_TIER_SENIOR = 0.7
_TIER_MID = 0.5
_TIER_JUNIOR_MID = 0.35


# ── state ──────────────────────────────────────────────────────────


class CandidateState(TypedDict):
    messages: list[BaseMessage]
    output_dir: str
    mode: str  # "perfect" | "random"
    proficiency: float  # 0.0–1.0
    deliverable_format: str  # "PBIP" | "IPYNB"
    retry_count: int
    validation_error: str | None
    failed: bool


# ── prompting ──────────────────────────────────────────────────────

_SIM_PROMPTS: dict | None = None


def _sim_prompts() -> dict:
    global _SIM_PROMPTS
    if _SIM_PROMPTS is None:
        _SIM_PROMPTS = yaml.safe_load(
            importlib.resources.files("buc_factory")
            .joinpath("conf/sim_prompt_templates.yml")
            .read_text(encoding="utf-8")
        )
    return _SIM_PROMPTS


def _persona_and_instructions(mode: str, proficiency: float) -> tuple[str, str]:
    tiers = _sim_prompts()["tiers"]
    pct = int(proficiency * 100)
    if mode == "perfect" or proficiency >= _TIER_EXPERT:
        t = tiers["expert"]
        return t["persona"], t["instructions"].rstrip()
    if proficiency >= _TIER_SENIOR:
        t = tiers["senior"]
        return t["persona"], fmt(
            t["instructions"], proficiency_pct=pct, skip_pct=int((1.0 - proficiency) * 10)
        ).rstrip()
    if proficiency >= _TIER_MID:
        t = tiers["mid"]
        return t["persona"], fmt(
            t["instructions"],
            proficiency_pct=pct,
            todo_count=max(1, int((1.0 - proficiency) * 6)),
        ).rstrip()
    if proficiency >= _TIER_JUNIOR_MID:
        t = tiers["junior_mid"]
        return t["persona"], fmt(
            t["instructions"],
            proficiency_pct=pct,
            todo_count=max(2, int((1.0 - proficiency) * 9)),
        ).rstrip()
    t = tiers["junior"]
    return t["persona"], fmt(t["instructions"], proficiency_pct=pct).rstrip()


def _build_system_prompt(mode: str, proficiency: float, deliverable_format: str) -> str:
    p = _sim_prompts()
    persona, instructions = _persona_and_instructions(mode, proficiency)
    guidance_key = "notebook" if deliverable_format == "IPYNB" else "pbip"
    format_guidance = p["format_guidance"][guidance_key].rstrip()
    return fmt(
        p["system_prompt"],
        persona=persona,
        deliverable_format=deliverable_format,
        instructions=instructions,
        format_guidance=format_guidance,
    )


def _build_task_prompt(
    mode: str,
    proficiency: float,
    deliverable_format: str,
    validation_error: str | None = None,
) -> str:
    p = _sim_prompts()
    format_label = (
        "Jupyter Notebook" if deliverable_format == "IPYNB" else "Power BI Desktop (PBIP)"
    )
    prof_desc = "perfect (expert)" if mode == "perfect" else f"{proficiency:.0%} proficiency"
    prompt = fmt(p["task_prompt"], mode=mode, prof_desc=prof_desc, format_label=format_label)
    if validation_error:
        prompt += "\n" + fmt(p["task_prompt_retry_suffix"], validation_error=validation_error)
    return prompt


# ── validation ─────────────────────────────────────────────────────


def _validate_simulation(output_dir: Path, deliverable_format: str) -> tuple[bool, str]:
    """Check that the agent produced non-trivial output in the starter project."""
    starter_dir = output_dir / "starter"
    if not starter_dir.exists():
        return False, "starter/ directory missing — source artifacts may not have been copied"

    if deliverable_format == "IPYNB":
        notebook = starter_dir / "notebook.ipynb"
        if not notebook.exists():
            return False, "starter/notebook.ipynb not found"
        size = len(notebook.read_bytes())
        if size < 500:
            return False, f"notebook.ipynb too small ({size} bytes) — does not appear modified"
    else:  # PBIP
        if not list(starter_dir.glob("*.pbip")):
            return False, "No .pbip file found in starter/"
        has_tmdl = bool(list(starter_dir.rglob("*.tmdl")))
        has_report = (starter_dir / "Assessment.Report/pages/pages.json").exists() or (
            starter_dir / "Assessment.Report/report.json"
        ).exists()
        if not has_tmdl and not has_report:
            return False, "No TMDL or report files in starter/ — project was not modified"

    return True, "simulation output validated"


# ── Anthropic auth error detection ─────────────────────────────────


def _is_anthropic_auth_error(exc: Exception) -> bool:
    _AUTH_FRAGMENTS = ("authentication method", "api_key", "anthropic_api_key", "x-api-key")
    cause: Exception | None = exc
    for _ in range(5):
        if isinstance(cause, (_anthropic.AuthenticationError, _anthropic.PermissionDeniedError)):
            return True
        if isinstance(cause, _anthropic.BadRequestError) and (
            "credit balance" in str(cause).lower()
        ):
            return True
        if isinstance(cause, (TypeError, ValueError)) and any(
            f in str(cause).lower() for f in _AUTH_FRAGMENTS
        ):
            return True
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
    return False


# ── graph ──────────────────────────────────────────────────────────


def build_candidate_graph(
    output_dir: Path,
    mode: str,
    proficiency: float,
    deliverable_format: str,
):
    """Compile and return a LangGraph for simulating a candidate solving an assessment.

    The graph runs a single agentic task with up to SIM_MAX_RETRIES retries.
    Nodes are closures over output_dir, mode, proficiency, and deliverable_format.
    """
    _allowed_tools = {"write_file", "read_file", "list_files", "mark_subtask_complete"}
    all_schemas, all_dispatch = make_tools(output_dir)
    schemas = [s for s in all_schemas if s["name"] in _allowed_tools]
    dispatch = {k: v for k, v in all_dispatch.items() if k in _allowed_tools}

    system_prompt = _build_system_prompt(mode, proficiency, deliverable_format)
    llm_claude = AnthropicLLM()
    llm_openai = OpenAILLM()
    _use_openai: list[bool] = [False]

    def _invoke(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int, int, str]:
        if _use_openai[0]:
            msgs, in_tok, out_tok = llm_openai.run_agent_loop(
                messages=messages,
                system=system_prompt,
                tools=schemas,
                dispatch=dispatch,
                max_tokens=_SIM_MAX_TOKENS,
                model=_SIM_OPENAI_FALLBACK,
            )
            return msgs, in_tok, out_tok, _SIM_OPENAI_FALLBACK

        try:
            msgs, in_tok, out_tok = llm_claude.run_agent_loop(
                messages=messages,
                system=system_prompt,
                tools=schemas,
                dispatch=dispatch,
                max_tokens=_SIM_MAX_TOKENS,
                model=_SIM_MODEL,
            )
            return msgs, in_tok, out_tok, _SIM_MODEL
        except Exception as exc:
            if _is_anthropic_auth_error(exc):
                _use_openai[0] = True
                LOGGER.warning(
                    "Anthropic unavailable for simulation; falling back to %s",
                    _SIM_OPENAI_FALLBACK,
                )
                msgs, in_tok, out_tok = llm_openai.run_agent_loop(
                    messages=messages,
                    system=system_prompt,
                    tools=schemas,
                    dispatch=dispatch,
                    max_tokens=_SIM_MAX_TOKENS,
                    model=_SIM_OPENAI_FALLBACK,
                )
                return msgs, in_tok, out_tok, _SIM_OPENAI_FALLBACK
            raise

    _task_start: list[float] = [0.0]

    def prepare_simulate(
        state: CandidateState,
    ) -> Command[Literal["run_simulate"]]:
        _task_start[0] = time.perf_counter()
        attempt = state["retry_count"] + 1
        LOGGER.info("\n━━━ [simulate_candidate] attempt %d/%d ━━━", attempt, _SIM_MAX_RETRIES)
        prompt = _build_task_prompt(
            state["mode"],
            state["proficiency"],
            state["deliverable_format"],
            state.get("validation_error"),
        )
        return Command(
            goto="run_simulate",
            update={"messages": [HumanMessage(content=prompt)], "validation_error": None},
        )

    def run_simulate(
        state: CandidateState,
    ) -> Command[Literal["validate_simulate"]]:
        msgs, in_tok, out_tok, model = _invoke(state["messages"])
        elapsed = time.perf_counter() - _task_start[0]
        LOGGER.info(
            "  simulate complete: %d+%d tokens in %.1fs using %s",
            in_tok,
            out_tok,
            elapsed,
            model,
        )
        return Command(goto="validate_simulate", update={"messages": msgs})

    def validate_simulate(
        state: CandidateState,
    ) -> Command[Literal["prepare_simulate", "fail_simulate", "__end__"]]:
        ok, msg = _validate_simulation(Path(state["output_dir"]), state["deliverable_format"])
        elapsed = time.perf_counter() - _task_start[0]
        LOGGER.info("  validation: %s %s (%.1fs)", "✓" if ok else "✗", msg, elapsed)
        if ok:
            return Command(goto=END)
        new_retry = state["retry_count"] + 1
        if new_retry >= _SIM_MAX_RETRIES:
            LOGGER.error("  max retries reached — failing simulation")
            return Command(
                goto="fail_simulate",
                update={"validation_error": msg, "retry_count": new_retry},
            )
        return Command(
            goto="prepare_simulate",
            update={"validation_error": msg, "retry_count": new_retry},
        )

    def fail_simulate(
        state: CandidateState,
    ) -> Command[Literal["__end__"]]:
        LOGGER.error(
            "\n✗ [simulate_candidate] failed after %d attempts: %s",
            _SIM_MAX_RETRIES,
            state.get("validation_error"),
        )
        return Command(goto=END, update={"failed": True})

    builder = StateGraph(CandidateState)
    builder.add_node("prepare_simulate", prepare_simulate)
    builder.add_node("run_simulate", run_simulate)
    builder.add_node("validate_simulate", validate_simulate)
    builder.add_node("fail_simulate", fail_simulate)
    builder.set_entry_point("prepare_simulate")
    return builder.compile()
