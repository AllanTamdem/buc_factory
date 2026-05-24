"""
LangGraph 1.x orchestration for the buc_factory agent.

Graph topology:

    prepare_task → run_task → validate_task ──► prepare_task       (sequential retry / next)
                                            ├──► run_parallel_group ──► prepare_task
                                            |                        └──► fail_task ──► END
                                            ├──► fail_task ──► END
                                            └──► END                (all tasks done)

Parallel groups (both tasks run concurrently via ThreadPoolExecutor):
  • After roll_scenario   (idx 1): write_brief ∥ design_data_schema  → advance to idx 4
  • After generate_data_script (idx 4): generate_starter ∥ write_recruiter_solution → idx 7
"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic as _anthropic
import mlflow.langchain
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..llm.claudeai import AnthropicLLM
from ..llm.gptai import OpenAILLM
from ..tracking import get_run_id, log_prompt, log_task_result, register_system_prompt, set_run_id
from .entity import PLAN, BucState, DomainConfig
from .prompting import build_system, raw_system_template, task_prompt
from .tool import make_tools
from .validator import MAX_RETRIES_PER_TASK, TASKS_WITH_RUN_PYTHON, validate_and_extract

LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 8000
_LARGE_MAX_TOKENS = 16000

_TASK_MODEL: dict[str, str] = {
    "bootstrap_domain": "claude-sonnet-4-6",
    "roll_scenario": "claude-haiku-4-5-20251001",
    "write_brief": "claude-sonnet-4-6",
    "design_data_schema": "gpt-5.3-chat-latest",
    "generate_data_script": "o4-mini",
    "generate_starter": "claude-opus-4-7",
    "write_recruiter_solution": "claude-opus-4-7",
    "final_assembly": "claude-haiku-4-5-20251001",
}

# Keys are the new_index values that trigger run_parallel_group (i.e. the PLAN index
# of the first task in the group, reached after the preceding sequential task completes).
_PARALLEL_GROUPS: dict[int, list[str]] = {
    2: ["write_brief", "design_data_schema"],  # after roll_scenario (idx 1)
    5: ["generate_starter", "write_recruiter_solution"],  # after generate_data_script (idx 4)
}
# Derived so _PARALLEL_GROUPS is the sole source of truth.
_PARALLEL_GROUP_NEXT: dict[int, int] = {
    k: max(PLAN.index(n) for n in v) + 1 for k, v in _PARALLEL_GROUPS.items()
}

# All context for these tasks is pre-loaded into the prompt; read_file serves no purpose.
_TASKS_WITHOUT_READ_FILE: set[str] = {
    "bootstrap_domain",
    "roll_scenario",
    "write_brief",
    "design_data_schema",
}


def _task_config(task_name: str) -> tuple[int, str, str]:
    """Return (max_tokens, model, provider) for a task."""
    _large = {"write_recruiter_solution", "generate_data_script", "generate_starter"}
    max_tokens = _LARGE_MAX_TOKENS if task_name in _large else _DEFAULT_MAX_TOKENS
    model = _TASK_MODEL[task_name]
    provider = "openai" if not model.startswith("claude") else "claude"
    return max_tokens, model, provider


# ── Anthropic → OpenAI fallback ────────────────────────────────────

# Tier-based equivalents (haiku→mini, sonnet→5.4, opus→5.5).
_CLAUDE_TO_OPENAI: dict[str, str] = {
    "claude-haiku-4-5-20251001": "gpt-5-mini",
    "claude-sonnet-4-6": "gpt-5.4",
    "claude-opus-4-7": "gpt-5.5",
}
# Per-task override where the default tier mapping isn't the best choice.
_TASK_OPENAI_OVERRIDE: dict[str, str] = {
    "generate_starter": "gpt-5.3-codex",  # code-heavy task benefits from the codex variant
}

# Flipped to True on the first Anthropic auth/billing error; never reset.
# All subsequent Claude tasks immediately reroute to OpenAI.
_anthropic_unavailable: list[bool] = [False]


def _openai_fallback(task_name: str, claude_model: str) -> str:
    """Return the OpenAI model to use when Anthropic is unavailable for this task."""
    return _TASK_OPENAI_OVERRIDE.get(task_name) or _CLAUDE_TO_OPENAI.get(claude_model, "gpt-5.4")


def _is_anthropic_unavailable_err(exc: Exception) -> bool:
    """True for any Anthropic auth / config error, even when wrapped by LangChain.

    Covers four distinct failure modes:
    - Wrong key (HTTP 401)        → anthropic.AuthenticationError
    - Billing / perms (HTTP 403)  → anthropic.PermissionDeniedError
    - Insufficient credits (400)  → anthropic.BadRequestError ("credit balance is too low")
    - Missing / empty key         → TypeError from the SDK
                                    ("Could not resolve authentication method")
      or pydantic.ValidationError from langchain_anthropic ("ANTHROPIC_API_KEY not set")
    """
    _AUTH_FRAGMENTS = ("authentication method", "api_key", "anthropic_api_key", "x-api-key")
    cause: Exception | None = exc
    for _ in range(5):
        if isinstance(cause, (_anthropic.AuthenticationError, _anthropic.PermissionDeniedError)):
            return True
        if isinstance(cause, _anthropic.BadRequestError) and (
            "credit balance" in str(cause).lower()
        ):
            return True
        if isinstance(cause, (TypeError, ValueError)):
            msg = str(cause).lower()
            if any(frag in msg for frag in _AUTH_FRAGMENTS):
                return True
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
    return False


def _save_progress(state: BucState, new_task_index: int, extra: dict) -> None:
    tasks_remaining = PLAN[new_task_index:]
    progress = {
        "task_index": new_task_index,
        "current_task": tasks_remaining[0] if tasks_remaining else None,
        "tasks_remaining": tasks_remaining,
        "bootstrapped_dimensions": extra.get(
            "bootstrapped_dimensions", state.get("bootstrapped_dimensions")
        ),
        "bootstrapped_entities": extra.get(
            "bootstrapped_entities", state.get("bootstrapped_entities")
        ),
        "scenario": extra.get("scenario", state.get("scenario")),
    }
    Path(state["output_dir"], "state.json").write_text(json.dumps(progress, indent=2))


def build_graph(output_dir: Path, cfg: DomainConfig):
    """
    Compile and return a LangGraph 1.x graph for the buc_factory agent.

    All nodes are closures over ``output_dir``, ``cfg``, and a shared
    ``AnthropicLLM`` instance. The compiled graph only needs a ``BucState``
    dict at invoke time.
    """
    mlflow.langchain.autolog()
    schemas, dispatch = make_tools(output_dir)
    system_prompt_claude = build_system(cfg, provider="claude")
    system_prompt_openai = build_system(cfg, provider="openai")
    register_system_prompt(raw_system_template("claude"), name="buc-factory-system-claude")
    register_system_prompt(raw_system_template("openai"), name="buc-factory-system-openai")
    llm_claude = AnthropicLLM()
    llm_openai = OpenAILLM()

    def _active_schemas(task_name: str) -> list[dict]:
        return [
            s
            for s in schemas
            if (s["name"] != "run_python" or task_name in TASKS_WITH_RUN_PYTHON)
            and (s["name"] != "validate_csv_integrity" or task_name in TASKS_WITH_RUN_PYTHON)
            and (s["name"] != "read_file" or task_name not in _TASKS_WITHOUT_READ_FILE)
        ]

    def _active_dispatch(task_name: str) -> dict:
        return {
            k: v
            for k, v in dispatch.items()
            if (k != "run_python" or task_name in TASKS_WITH_RUN_PYTHON)
            and (k != "validate_csv_integrity" or task_name in TASKS_WITH_RUN_PYTHON)
            and (k != "read_file" or task_name not in _TASKS_WITHOUT_READ_FILE)
        }

    def _invoke(task_name: str, messages: list) -> tuple[list, int, int, str]:
        """Invoke the LLM for a task. Returns (messages, in_tok, out_tok, actual_model).

        If Anthropic is already known to be unavailable, reroutes to OpenAI immediately.
        On first auth/billing error from Anthropic, sets the module-level flag and
        transparently retries the same call via the OpenAI fallback model.
        """
        max_tok, model, provider = _task_config(task_name)

        # Fast path: Anthropic already known unavailable → reroute now.
        if _anthropic_unavailable[0] and provider == "claude":
            model = _openai_fallback(task_name, model)
            provider = "openai"
            LOGGER.info("  [fallback] Anthropic unavailable — using %s for [%s]", model, task_name)

        llm = llm_openai if provider == "openai" else llm_claude
        system = system_prompt_openai if provider == "openai" else system_prompt_claude

        try:
            msgs, in_tok, out_tok = llm.run_agent_loop(
                messages=messages,
                system=system,
                tools=_active_schemas(task_name),
                dispatch=_active_dispatch(task_name),
                max_tokens=max_tok,
                model=model,
            )
        except Exception as exc:
            if provider == "claude" and _is_anthropic_unavailable_err(exc):
                _anthropic_unavailable[0] = True
                fallback_model = _openai_fallback(task_name, model)
                LOGGER.warning(
                    "  Anthropic API unavailable (%s); falling back to %s for [%s]"
                    " — all remaining Claude tasks will use OpenAI",
                    type(exc).__name__,
                    fallback_model,
                    task_name,
                )
                msgs, in_tok, out_tok = llm_openai.run_agent_loop(
                    messages=messages,
                    system=system_prompt_openai,
                    tools=_active_schemas(task_name),
                    dispatch=_active_dispatch(task_name),
                    max_tokens=max_tok,
                    model=fallback_model,
                )
                model = fallback_model
            else:
                raise

        return msgs, in_tok, out_tok, model

    # ── nodes ──────────────────────────────────────────────────────

    _task_start: list[float] = [0.0]  # mutable cell shared across closures
    _task_tokens: dict[str, list[int]] = {}  # task_name -> [total_input, total_output]
    # task_name -> model actually used (may differ from _TASK_MODEL on fallback)
    _task_actual_models: dict[str, str] = {}

    def _fmt(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"

    def prepare_task(state: BucState) -> Command[Literal["run_task", "validate_task"]]:
        _task_start[0] = time.perf_counter()
        task_name = PLAN[state["task_index"]]
        attempt = f"{state['retry_count'] + 1}/{MAX_RETRIES_PER_TASK}"
        LOGGER.info(f"\n━━━ [{task_name}] attempt {attempt} ━━━")

        update: dict = {"validation_error": None, "current_task": task_name}

        # Fully-specified bootstrap: write bootstrap.json directly and skip the LLM.
        # Calling the model just to echo back config values is wasteful.
        if task_name == "bootstrap_domain" and cfg.dimensions and cfg.entities:
            bootstrap = {
                "dimensions": cfg.dimensions,
                "entities": cfg.entities,
                "rationale": "Fully specified via config.",
            }
            (Path(state["output_dir"]) / "bootstrap.json").write_text(
                json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("  bootstrap_domain: fully specified — skipped LLM")
            update["messages"] = []
            return Command(goto="validate_task", update=update)

        # Pre-roll scenario values in Python so randomness is independent of model
        # temperature. Only roll on the first attempt so retries use the same values.
        rolled: dict | None = None
        if task_name == "roll_scenario":
            if state["retry_count"] == 0:
                dims = state.get("bootstrapped_dimensions") or cfg.dimensions or {}
                rolled = {dim: random.choice(vals) for dim, vals in dims.items()}
                update["rolled"] = rolled
                LOGGER.info(f"  rolled scenario: {rolled}")
            else:
                rolled = state.get("rolled")

        _data_schema_json: str | None = None
        _candidate_brief: str | None = None
        if task_name == "generate_data_script":
            base = Path(state["output_dir"])
            schema_path = base / "brief" / "data_schema.json"
            brief_path = base / "brief" / "candidate_brief.md"
            if schema_path.exists():
                _data_schema_json = schema_path.read_text(encoding="utf-8")
            if brief_path.exists():
                _candidate_brief = brief_path.read_text(encoding="utf-8")

        _, _, _provider = _task_config(task_name)
        prompt = task_prompt(
            task_name,
            cfg,
            state.get("validation_error"),
            rolled=rolled,
            data_schema_json=_data_schema_json,
            candidate_brief=_candidate_brief,
            provider=_provider,
        )
        log_prompt(prompt, f"prompts/{task_name}.md")
        update["messages"] = [HumanMessage(content=prompt)]
        return Command(goto="run_task", update=update)

    def run_task(state: BucState) -> Command[Literal["validate_task"]]:
        task_name = PLAN[state["task_index"]]
        final_messages, in_tok, out_tok, actual_model = _invoke(task_name, state["messages"])
        acc = _task_tokens.setdefault(task_name, [0, 0])
        acc[0] += in_tok
        acc[1] += out_tok
        _task_actual_models[task_name] = actual_model
        return Command(goto="validate_task", update={"messages": final_messages})

    def validate_task(
        state: BucState,
    ) -> Command[Literal["prepare_task", "run_parallel_group", "fail_task", "__end__"]]:
        task_name = PLAN[state["task_index"]]
        ok, msg, extracted = validate_and_extract(task_name, cfg, output_dir, state)
        elapsed_s = time.perf_counter() - _task_start[0]
        LOGGER.info(f"  validation: {'✓' if ok else '✗'} {msg} ({_fmt(elapsed_s)})")

        if ok:
            new_index = state["task_index"] + 1
            _save_progress(state, new_index, extracted)
            tok = _task_tokens.get(task_name, [0, 0])
            _, task_model, _ = _task_config(task_name)
            log_task_result(
                task_name,
                ok=True,
                elapsed_s=elapsed_s,
                retries=state["retry_count"],
                input_tokens=tok[0],
                output_tokens=tok[1],
                model=_task_actual_models.get(task_name, task_model),
            )
            update = {
                "validation_error": None,
                "task_index": new_index,
                "tasks_remaining": PLAN[new_index:],
                "retry_count": 0,
                **extracted,
            }
            next_node = (
                END
                if new_index >= len(PLAN)
                else "run_parallel_group"
                if new_index in _PARALLEL_GROUPS
                else "prepare_task"
            )
            return Command(goto=next_node, update=update)

        new_retry = state["retry_count"] + 1
        attempt = f"{state['retry_count'] + 1}/{MAX_RETRIES_PER_TASK}"
        LOGGER.warning(f"  validation failed (attempt {attempt}): {msg} ({_fmt(elapsed_s)})")
        update = {"validation_error": msg, "retry_count": new_retry}
        goto = "fail_task" if new_retry >= MAX_RETRIES_PER_TASK else "prepare_task"
        if goto == "fail_task":
            tok = _task_tokens.get(task_name, [0, 0])
            _, task_model, _ = _task_config(task_name)
            log_task_result(
                task_name,
                ok=False,
                elapsed_s=elapsed_s,
                retries=new_retry,
                input_tokens=tok[0],
                output_tokens=tok[1],
                model=_task_actual_models.get(task_name, task_model),
            )
        return Command(goto=goto, update=update)

    def fail_task(state: BucState) -> Command[Literal["__end__"]]:
        task_name = PLAN[min(state["task_index"], len(PLAN) - 1)]
        elapsed = _fmt(time.perf_counter() - _task_start[0])
        LOGGER.error(
            f"\n✗ [{task_name}] failed after {MAX_RETRIES_PER_TASK} attempts"
            f": {state.get('validation_error')} ({elapsed})"
        )
        return Command(goto=END, update={"failed": True})

    # ── parallel helpers ───────────────────────────────────────────

    def _run_one_task_full(task_name: str, state: BucState) -> tuple[bool, str, dict, dict]:
        """Run one task with retries in a thread. Returns (ok, msg, extracted, tok_info)."""
        retry_feedback: str | None = None
        in_total = out_total = 0
        t0 = time.perf_counter()

        # Pre-load context files once so they aren't fetched via read_file round-trips.
        _scenario_json: str | None = None
        _bootstrap_json: str | None = None
        _data_schema_json: str | None = None
        _candidate_brief: str | None = None
        base = Path(state["output_dir"])
        if task_name in {"write_brief", "design_data_schema"}:
            bootstrap_path = base / "bootstrap.json"
            if bootstrap_path.exists():
                _bootstrap_json = bootstrap_path.read_text(encoding="utf-8")
            if task_name == "write_brief":
                scenario_path = base / "scenario.json"
                if scenario_path.exists():
                    _scenario_json = scenario_path.read_text(encoding="utf-8")
        elif task_name in {"generate_starter", "write_recruiter_solution"}:
            schema_path = base / "brief" / "data_schema.json"
            brief_path = base / "brief" / "candidate_brief.md"
            if schema_path.exists():
                _data_schema_json = schema_path.read_text(encoding="utf-8")
            if brief_path.exists():
                _candidate_brief = brief_path.read_text(encoding="utf-8")
            if task_name == "write_recruiter_solution":
                scenario_path = base / "scenario.json"
                if scenario_path.exists():
                    _scenario_json = scenario_path.read_text(encoding="utf-8")

        _, _, _provider = _task_config(task_name)
        first_prompt: str | None = None
        last_model: str = _TASK_MODEL[task_name]
        for attempt in range(MAX_RETRIES_PER_TASK):
            prompt = task_prompt(
                task_name,
                cfg,
                retry_feedback,
                rolled=None,
                scenario_json=_scenario_json,
                bootstrap_json=_bootstrap_json,
                data_schema_json=_data_schema_json,
                candidate_brief=_candidate_brief,
                provider=_provider,
            )
            if first_prompt is None:
                first_prompt = prompt
            _, in_tok, out_tok, last_model = _invoke(task_name, [HumanMessage(content=prompt)])
            in_total += in_tok
            out_total += out_tok
            ok, msg, extracted = validate_and_extract(task_name, cfg, output_dir, state)
            elapsed = time.perf_counter() - t0
            status = "✓" if ok else "✗"
            LOGGER.info(f"  [{task_name}] attempt {attempt + 1}: {status} {msg} ({_fmt(elapsed)})")
            if ok:
                tok_info = {
                    "input_tokens": in_total,
                    "output_tokens": out_total,
                    "elapsed": elapsed,
                    "retries": attempt,
                    "prompt": first_prompt,
                    "model": last_model,
                }
                return True, msg, extracted, tok_info
            retry_feedback = msg

        tok_info = {
            "input_tokens": in_total,
            "output_tokens": out_total,
            "elapsed": time.perf_counter() - t0,
            "retries": MAX_RETRIES_PER_TASK,
            "prompt": first_prompt,
            "model": last_model,
        }
        return False, retry_feedback or "", {}, tok_info

    def run_parallel_group(
        state: BucState,
    ) -> Command[Literal["prepare_task", "fail_task"]]:
        task_names = _PARALLEL_GROUPS[state["task_index"]]
        LOGGER.info(f"\n━━━ [parallel: {' ∥ '.join(task_names)}] ━━━")

        parent_run_id = get_run_id()

        def _worker(name: str) -> tuple:
            set_run_id(parent_run_id)
            return _run_one_task_full(name, state)

        with ThreadPoolExecutor(max_workers=len(task_names)) as executor:
            futures = {executor.submit(_worker, name): name for name in task_names}

        results = {}
        for future, name in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = (False, f"exception: {exc}", {}, {})

        # MLflow is not thread-safe — log everything after all threads complete
        for task_name in task_names:
            tok = results[task_name][3]
            if tok.get("prompt"):
                log_prompt(tok["prompt"], f"prompts/{task_name}.md")

        merged_extracted: dict = {}
        for task_name in task_names:
            ok, _, extracted, tok = results[task_name]
            _, default_model, _ = _task_config(task_name)
            log_task_result(
                task_name,
                ok=ok,
                elapsed_s=tok.get("elapsed", 0.0),
                retries=tok.get("retries", 0),
                input_tokens=tok.get("input_tokens", 0),
                output_tokens=tok.get("output_tokens", 0),
                model=tok.get("model", default_model),
            )
            if ok:
                merged_extracted.update(extracted)

        failed = [n for n in task_names if not results[n][0]]
        if failed:
            err = {n: results[n][1] for n in failed}
            LOGGER.error(f"  parallel group failed: {err}")
            return Command(goto="fail_task", update={"validation_error": str(err)})

        new_index = _PARALLEL_GROUP_NEXT[state["task_index"]]
        _save_progress(state, new_index, merged_extracted)
        LOGGER.info(f"  ✓ parallel group done — advancing to [{PLAN[new_index]}]")
        return Command(
            goto="prepare_task",
            update={
                "task_index": new_index,
                "tasks_remaining": PLAN[new_index:],
                "retry_count": 0,
                "validation_error": None,
                **merged_extracted,
            },
        )

    # ── graph assembly ─────────────────────────────────────────────

    builder = StateGraph(BucState)
    builder.add_node("prepare_task", prepare_task)
    builder.add_node("run_task", run_task)
    builder.add_node("validate_task", validate_task)
    builder.add_node("run_parallel_group", run_parallel_group)
    builder.add_node("fail_task", fail_task)
    builder.set_entry_point("prepare_task")
    return builder.compile()
