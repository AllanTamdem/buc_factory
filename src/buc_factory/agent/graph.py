"""
LangGraph 1.x orchestration for the buc_factory agent.

Graph topology:

    prepare_task → run_task → validate_task ──► prepare_task  (retry / next task)
                                            ├──► fail_task ──► END
                                            └──► END           (all tasks done)

Each node returns a ``Command`` that carries both the state update and the
next-node destination, replacing conditional edges and routing functions.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..llm.claudeai import AnthropicLLM
from ..tracking import log_prompt, log_task_result, register_system_prompt
from .entity import PLAN, BucState, DomainConfig
from .prompting import build_system, task_prompt
from .tool import make_tools
from .validator import MAX_RETRIES_PER_TASK, TASKS_WITH_RUN_PYTHON, validate_and_extract

LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 8000
_LARGE_MAX_TOKENS = 16000  # write_recruiter_solution produces one large file


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
    schemas, dispatch = make_tools(output_dir)
    system_prompt = build_system(cfg)
    register_system_prompt(system_prompt)
    llm = AnthropicLLM()

    def _active_schemas(task_name: str) -> list[dict]:
        return [
            s for s in schemas if s["name"] != "run_python" or task_name in TASKS_WITH_RUN_PYTHON
        ]

    def _active_dispatch(task_name: str) -> dict:
        return {
            k: v
            for k, v in dispatch.items()
            if k != "run_python" or task_name in TASKS_WITH_RUN_PYTHON
        }

    # ── nodes ──────────────────────────────────────────────────────

    _task_start: list[float] = [0.0]  # mutable cell shared across closures
    _task_tokens: dict[str, list[int]] = {}  # task_name -> [total_input, total_output]

    def _fmt(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"

    def prepare_task(state: BucState) -> Command[Literal["run_task"]]:
        _task_start[0] = time.perf_counter()
        task_name = PLAN[state["task_index"]]
        attempt = f"{state['retry_count'] + 1}/{MAX_RETRIES_PER_TASK}"
        LOGGER.info(f"\n━━━ [{task_name}] attempt {attempt} ━━━")

        update: dict = {"validation_error": None, "current_task": task_name}

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

        prompt = task_prompt(task_name, cfg, state.get("validation_error"), rolled=rolled)
        log_prompt(prompt, f"prompts/{task_name}.md")
        update["messages"] = [HumanMessage(content=prompt)]
        return Command(goto="run_task", update=update)

    def run_task(state: BucState) -> Command[Literal["validate_task"]]:
        task_name = PLAN[state["task_index"]]
        _large_tasks = {"write_recruiter_solution", "generate_data_script"}
        max_tokens = _LARGE_MAX_TOKENS if task_name in _large_tasks else _DEFAULT_MAX_TOKENS
        final_messages, in_tok, out_tok = llm.run_agent_loop(
            messages=state["messages"],
            system=system_prompt,
            tools=_active_schemas(task_name),
            dispatch=_active_dispatch(task_name),
            max_tokens=max_tokens,
        )
        acc = _task_tokens.setdefault(task_name, [0, 0])
        acc[0] += in_tok
        acc[1] += out_tok
        return Command(goto="validate_task", update={"messages": final_messages})

    def validate_task(
        state: BucState,
    ) -> Command[Literal["prepare_task", "fail_task", "__end__"]]:
        task_name = PLAN[state["task_index"]]
        ok, msg, extracted = validate_and_extract(task_name, cfg, output_dir, state)
        elapsed_s = time.perf_counter() - _task_start[0]
        LOGGER.info(f"  validation: {'✓' if ok else '✗'} {msg} ({_fmt(elapsed_s)})")

        if ok:
            new_index = state["task_index"] + 1
            _save_progress(state, new_index, extracted)
            tok = _task_tokens.get(task_name, [0, 0])
            log_task_result(
                task_name,
                ok=True,
                elapsed_s=elapsed_s,
                retries=state["retry_count"],
                step=state["task_index"],
                input_tokens=tok[0],
                output_tokens=tok[1],
            )
            update = {
                "validation_error": None,
                "task_index": new_index,
                "tasks_remaining": PLAN[new_index:],
                "retry_count": 0,
                **extracted,
            }
            return Command(goto=END if new_index >= len(PLAN) else "prepare_task", update=update)

        new_retry = state["retry_count"] + 1
        attempt = f"{state['retry_count'] + 1}/{MAX_RETRIES_PER_TASK}"
        LOGGER.warning(f"  validation failed (attempt {attempt}): {msg} ({_fmt(elapsed_s)})")
        update = {"validation_error": msg, "retry_count": new_retry}
        goto = "fail_task" if new_retry >= MAX_RETRIES_PER_TASK else "prepare_task"
        if goto == "fail_task":
            tok = _task_tokens.get(task_name, [0, 0])
            log_task_result(
                task_name,
                ok=False,
                elapsed_s=elapsed_s,
                retries=new_retry,
                step=state["task_index"],
                input_tokens=tok[0],
                output_tokens=tok[1],
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

    # ── graph assembly ─────────────────────────────────────────────

    builder = StateGraph(BucState)
    builder.add_node("prepare_task", prepare_task)
    builder.add_node("run_task", run_task)
    builder.add_node("validate_task", validate_task)
    builder.add_node("fail_task", fail_task)
    builder.set_entry_point("prepare_task")
    return builder.compile()
