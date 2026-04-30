import json
import logging
import time
from pathlib import Path

from .entity import AgentState, DomainConfig
from .prompting import dax_or_calc, task_prompt

LOGGER = logging.getLogger(__name__)

MAX_RETRIES_PER_TASK = 3
MODEL = "claude-opus-4-7"  # best for most business use cases and deeper tool use

# ──────────────────────────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────────────────────────


def validate_subtask(
    name: str, cfg: DomainConfig, output_dir: Path, state: AgentState
) -> tuple[bool, str]:
    try:
        if name == "bootstrap_domain":
            data = json.loads((output_dir / "bootstrap.json").read_text())
            if not data.get("dimensions") or not data.get("entities"):
                return False, "bootstrap.json must contain non-empty 'dimensions' and 'entities'"
            if len(data["entities"]) < 3:
                return False, f"need at least 3 entities, got {len(data['entities'])}"
            state.bootstrapped_dimensions = data["dimensions"]
            state.bootstrapped_entities = data["entities"]
            return True, f"{len(data['dimensions'])} dimensions, {len(data['entities'])} entities"

        if name == "roll_scenario":
            data = json.loads((output_dir / "scenario.json").read_text())
            expected = set((state.bootstrapped_dimensions or cfg.dimensions or {}).keys())
            missing = expected - data.keys()
            if missing:
                return False, f"scenario.json missing dimensions: {missing}"
            state.scenario = data
            return True, "scenario rolled"

        if name == "write_brief":
            f = output_dir / "brief/candidate_brief.md"
            if not f.exists():
                return False, "brief not created"
            content = f.read_text()
            if len(content) < 1500:
                return False, f"brief too short ({len(content)} chars)"
            return True, "brief OK"

        if name == "design_data_schema":
            schema = json.loads((output_dir / "brief/data_schema.json").read_text())
            files_in_schema = {f["filename"].replace(".csv", "") for f in schema.get("files", [])}
            entities = set(state.bootstrapped_entities or cfg.entities or [])
            overlap = len(files_in_schema & entities)
            if overlap < max(3, int(0.8 * len(entities))):
                return False, f"schema covers only {overlap}/{len(entities)} entities"
            if not schema.get("traps"):
                return False, "schema must declare at least one trap"
            return True, f"{overlap} entities, {len(schema['traps'])} traps"

        if name == "generate_data_script":
            csv_dir = output_dir / "starter/data"
            entities = state.bootstrapped_entities or cfg.entities or []
            for entity in entities:
                if not (csv_dir / f"{entity}.csv").exists():
                    return False, f"{entity}.csv not generated"
            if len(entities) >= 2:
                import pandas as pd

                fact = pd.read_csv(csv_dir / f"{entities[0]}.csv")
                dim = pd.read_csv(csv_dir / f"{entities[1]}.csv")
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
                        )
            return True, "CSVs generated, FK heuristic passes"

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
                    return False, "starter looks empty"
                return True, f"{len(files)} files (generic check)"
            for r in required:
                if not (starter / r).exists():
                    return False, f"missing {r}"
                if r.endswith((".json", ".pbir", ".pbip")):
                    try:
                        json.loads((starter / r).read_text())
                    except json.JSONDecodeError as e:
                        return False, f"{r} invalid JSON: {e}"
            return True, f"{fmt} starter OK"

        if name == "write_recruiter_solution":
            f = output_dir / "solution/recruiter_solution.md"
            if not f.exists():
                return False, "solution not created"
            content = f.read_text()
            if len(content) < 3000:
                return False, f"solution too short ({len(content)} chars)"
            calc_kw = dax_or_calc(cfg.tool).split()[0]
            if calc_kw.lower() not in content.lower():
                return False, f"solution missing {calc_kw} content"
            return True, "solution OK"

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
                return False, f"missing: {missing}"
            return True, "all artifacts present"

        return False, f"unknown task {name}"
    except Exception as e:
        return False, f"validator crashed: {type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────
# Execution loop
# ──────────────────────────────────────────────────────────────────


_TASKS_WITH_RUN_PYTHON = {"generate_data_script"}


def run_subtask(client, name, cfg, state, output_dir, tool_schemas, dispatch, system) -> bool:
    task = state.task(name)
    task.status = "running"
    state.save()
    retry_feedback: str | None = None
    t0 = time.perf_counter()

    active_tools = (
        tool_schemas
        if name in _TASKS_WITH_RUN_PYTHON
        else [t for t in tool_schemas if t["name"] != "run_python"]
    )

    for attempt in range(1, MAX_RETRIES_PER_TASK + 1):
        task.attempts = attempt
        LOGGER.info(f"\n━━━ [{name}] attempt {attempt}/{MAX_RETRIES_PER_TASK} ━━━")

        messages = [{"role": "user", "content": task_prompt(name, cfg, retry_feedback)}]

        for _ in range(40):
            # write_recruiter_solution produces a large markdown file in one write_file call
            _max_tokens = 16000 if name == "write_recruiter_solution" else 8000
            response = client.messages.create(
                model=MODEL,
                max_tokens=_max_tokens,
                system=system,
                tools=active_tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break

            tool_results = []
            done_signal = False
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = dispatch.get(block.name)
                try:
                    result = fn(**block.input) if fn else f"ERROR: unknown tool {block.name}"
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                LOGGER.info(f"  → {block.name} → {str(result)[:120]}")
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
                )
                if block.name == "mark_subtask_complete":
                    done_signal = True
            messages.append({"role": "user", "content": tool_results})
            if done_signal:
                break

        ok, msg = validate_subtask(name, cfg, output_dir, state)
        LOGGER.info(f"  validation: {'✓' if ok else '✗'} {msg}")

        if ok:
            elapsed = time.perf_counter() - t0
            LOGGER.info(f"  [{name}] completed in {elapsed:.1f}s")
            task.status = "done"
            task.last_error = None
            state.save()
            return True

        retry_feedback = msg
        task.last_error = msg
        state.save()

    elapsed = time.perf_counter() - t0
    LOGGER.info(f"  [{name}] failed after {elapsed:.1f}s")
    task.status = "failed"
    state.save()
    return False
