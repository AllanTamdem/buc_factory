"""MLflow tracking utilities for buc_factory agent runs."""

import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.genai
import mlflow.langchain
from mlflow.entities.assessment import Feedback

from .agent.entity import DomainConfig
from .agent.prompting import dax_or_calc

LOGGER = logging.getLogger(__name__)

_EXPERIMENT = "buc-factory"
_task_log: list[dict] = []  # accumulated per-task results, reset each run

# Pricing in $ per 1M tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
_AGENT_MODEL = "claude-opus-4-7"


def _token_cost(input_tokens: int, output_tokens: int, model: str = _AGENT_MODEL) -> float:
    in_price, out_price = _MODEL_PRICING.get(model, _MODEL_PRICING[_AGENT_MODEL])
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def setup_mlflow(experiment_name: str = _EXPERIMENT) -> None:
    """Set experiment and enable LangGraph trace autologging.

    When running against a local file-based store, ensures the experiment's
    artifact_location points to a writable project-local path — soft-deleting and
    recreating if a stale path is detected. When MLFLOW_TRACKING_URI points to an
    HTTP server (e.g. Docker Compose), the server owns the artifact location and
    the local path check is skipped.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri.startswith("http"):
        artifact_root = (Path.cwd() / "mlflow_data" / "artifacts").as_uri()
        client = mlflow.MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            client.create_experiment(experiment_name, artifact_location=artifact_root)
        elif experiment.artifact_location != artifact_root:
            LOGGER.info(
                "Fixing experiment artifact location: %s → %s",
                experiment.artifact_location,
                artifact_root,
            )
            client.delete_experiment(experiment.experiment_id)
            client.create_experiment(experiment_name, artifact_location=artifact_root)

    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog(log_traces=True, silent=True)
    _register_judge_prompt()
    _task_log.clear()


def log_config(cfg: DomainConfig) -> None:
    """Log all DomainConfig fields as MLflow run parameters."""
    mlflow.log_params(
        {
            "industry": cfg.industry,
            "role": cfg.role,
            "tool": cfg.tool,
            "language": cfg.language,
            "seniority": cfg.seniority,
            "location": cfg.location,
            "duration_minutes": cfg.duration_minutes,
            "deliverable_format": cfg.deliverable_format,
        }
    )


def log_task_result(
    task_name: str,
    *,
    ok: bool,
    elapsed_s: float,
    retries: int,
    step: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Log per-task outcome metrics; step enables timeline views in the MLflow UI."""
    mlflow.log_metrics(
        {
            f"task.{task_name}.success": float(ok),
            f"task.{task_name}.duration_s": round(elapsed_s, 2),
            f"task.{task_name}.retries": float(retries),
            f"task.{task_name}.input_tokens": float(input_tokens),
            f"task.{task_name}.output_tokens": float(output_tokens),
        },
        step=step,
    )
    _task_log.append(
        {
            "task": task_name,
            "attempts": retries + 1 if ok else retries,
            "status": "✓" if ok else "✗",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": _token_cost(input_tokens, output_tokens),
        }
    )


def log_tasks_summary() -> None:
    """Write a markdown table of task outcomes as an MLflow artifact."""
    if not _task_log:
        return
    total_in = sum(e["input_tokens"] for e in _task_log)
    total_out = sum(e["output_tokens"] for e in _task_log)
    total_cost = sum(e["cost"] for e in _task_log)

    lines = [
        "# Task Summary\n",
        "| Task | Attempts | Status | Input tok | Output tok | Cost ($) |",
        "|------|:--------:|:------:|----------:|-----------:|---------:|",
    ]
    for e in _task_log:
        lines.append(
            f"| {e['task']} | {e['attempts']} | {e['status']} |"
            f" {e['input_tokens']:,} | {e['output_tokens']:,} | ${e['cost']:.4f} |"
        )
    lines += [
        "|------|:--------:|:------:|----------:|-----------:|---------:|",
        f"| **Total** | | | **{total_in:,}** | **{total_out:,}** | **${total_cost:.4f}** |",
    ]
    mlflow.log_metrics(
        {
            "total_input_tokens": float(total_in),
            "total_output_tokens": float(total_out),
            "total_cost_usd": total_cost,
        }
    )
    try:
        mlflow.log_text("\n".join(lines) + "\n", "task_summary.md")
    except Exception as exc:
        LOGGER.warning("task summary logging skipped: %s", exc)


def log_run_summary(*, total_s: float, failed: bool, tasks_completed: int) -> None:
    """Log end-of-run rollup metrics."""
    mlflow.log_metrics(
        {
            "total_duration_s": round(total_s, 2),
            "run_success": float(not failed),
            "tasks_completed": float(tasks_completed),
        }
    )


def register_system_prompt(template: str) -> None:
    """Register the system prompt in the MLflow Prompt Registry.

    Creates a new version only when the template text changes.
    """
    try:
        existing = mlflow.genai.load_prompt("buc-factory-system", allow_missing=True)
        if existing is not None and existing.template == template:
            return
        mlflow.genai.register_prompt(
            name="buc-factory-system",
            template=template,
            commit_message="system prompt",
        )
    except Exception as exc:
        LOGGER.warning("system prompt registration skipped: %s", exc)


def log_prompt(text: str, artifact_file: str) -> None:
    """Log a prompt string as a text artifact under prompts/."""
    try:
        mlflow.log_text(text, artifact_file)
    except Exception as exc:
        LOGGER.warning("prompt logging skipped (%s): %s", artifact_file, exc)


def log_output_artifacts(output_dir: Path) -> None:
    """Upload the entire output directory as MLflow artifacts under 'outputs/'.

    If the server's artifact store is read-only or unreachable, falls back to
    tagging the local output path so it remains discoverable in the MLflow UI.
    """
    if not output_dir.exists():
        return
    try:
        mlflow.log_artifacts(str(output_dir), artifact_path="outputs")
    except Exception as exc:
        LOGGER.warning("Artifact upload failed (%s); tagging local path instead.", exc)
        mlflow.set_tag("output_dir_local", str(output_dir.resolve()))


def evaluate_outputs(output_dir: Path, cfg: DomainConfig) -> None:
    """Log quality metrics and run the solution_relevancy LLM judge.

    Metrics logged:
    - bootstrap.dimension_count / bootstrap.entity_count
    - schema.table_count / schema.trap_count
    - {artifact}.char_count / .min_chars_met / .has_calc_keyword
    - solution_relevancy_score (1–10, via Claude Haiku judge)
    """
    # ── direct scalar metrics from structured artifacts ───────────
    bootstrap_file = output_dir / "bootstrap.json"
    if bootstrap_file.exists():
        try:
            data = json.loads(bootstrap_file.read_text())
            mlflow.log_metrics(
                {
                    "bootstrap.dimension_count": float(len(data.get("dimensions", {}))),
                    "bootstrap.entity_count": float(len(data.get("entities", []))),
                }
            )
        except Exception as exc:
            LOGGER.warning("bootstrap metrics skipped: %s", exc)

    schema_file = output_dir / "brief/data_schema.json"
    if schema_file.exists():
        try:
            schema = json.loads(schema_file.read_text())
            mlflow.log_metrics(
                {
                    "schema.table_count": float(len(schema.get("files", []))),
                    "schema.trap_count": float(len(schema.get("traps", []))),
                }
            )
        except Exception as exc:
            LOGGER.warning("schema metrics skipped: %s", exc)

    # ── text artifact evaluators ───────────────────────────────────
    calc_kw = dax_or_calc(cfg.tool).split()[0].lower()

    rows: list[dict] = []

    brief = output_dir / "brief/candidate_brief.md"
    if brief.exists():
        rows.append(
            {
                "artifact": "candidate_brief",
                "text": brief.read_text(encoding="utf-8"),
                "min_chars": 1500,
                "keyword": "",
            }
        )

    solution = output_dir / "solution/recruiter_solution.md"
    if solution.exists():
        rows.append(
            {
                "artifact": "recruiter_solution",
                "text": solution.read_text(encoding="utf-8"),
                "min_chars": 3000,
                "keyword": calc_kw,
            }
        )

    if not rows:
        return

    metrics: dict[str, float] = {}
    for row in rows:
        name = row["artifact"]
        text = row["text"]
        metrics[f"{name}.char_count"] = float(len(text))
        metrics[f"{name}.min_chars_met"] = 1.0 if len(text) >= row["min_chars"] else 0.0
        kw = row["keyword"]
        metrics[f"{name}.has_calc_keyword"] = 1.0 if (not kw or kw in text.lower()) else 0.0

    mlflow.log_metrics(metrics)

    brief_file = output_dir / "brief/candidate_brief.md"
    solution_file = output_dir / "solution/recruiter_solution.md"
    if brief_file.exists() and solution_file.exists():
        try:
            result = mlflow.genai.evaluate(
                data=[
                    {
                        "inputs": {"brief": brief_file.read_text(encoding="utf-8")},
                        "outputs": solution_file.read_text(encoding="utf-8"),
                    }
                ],
                scorers=[solution_relevancy],
            )
            table = result.tables.get("eval_results_table")
            score = None
            if table is not None and not table.empty and "solution_relevancy" in table.columns:
                score = table["solution_relevancy"].iloc[0]
            else:
                score = result.metrics.get("solution_relevancy/mean")
            if score is not None:
                mlflow.log_metric("solution_relevancy_score", float(score))
        except Exception as exc:
            LOGGER.warning("solution relevancy scoring failed: %s", exc)


# ── judge prompt ───────────────────────────────────────────────────────────────

_JUDGE_PROMPT_NAME = "buc-factory-solution-relevancy-judge"

# MLflow template syntax: {{variable}} for substitution, single braces are literal.
_RELEVANCY_PROMPT = """\
You are evaluating a recruiter solution for a BI technical assessment.

## CANDIDATE BRIEF
{{brief}}

## RECRUITER SOLUTION
{{solution}}

Score how well the recruiter solution covers and directly answers every requirement, \
question, and deliverable stated in the candidate brief.

Respond with a JSON object only — no prose outside the JSON:
{"score": <integer 1-10>, "reasoning": "<one paragraph>"}

Scoring guide:
- 9-10: all requirements fully addressed with concrete specifics
- 7-8:  most requirements addressed, minor gaps
- 5-6:  roughly half the requirements addressed
- 3-4:  only surface-level coverage
- 1-2:  solution largely ignores the brief
"""


def _register_judge_prompt() -> None:
    """Register the judge prompt; creates a new version only when the template changes."""
    from mlflow.entities.model_registry.prompt_version import PromptModelConfig

    try:
        existing = mlflow.genai.load_prompt(_JUDGE_PROMPT_NAME, allow_missing=True)
        if existing is not None and existing.template == _RELEVANCY_PROMPT:
            return
        mlflow.genai.register_prompt(
            name=_JUDGE_PROMPT_NAME,
            template=_RELEVANCY_PROMPT,
            commit_message="solution relevancy judge",
            tags={"role": "judge", "task": "solution_relevancy"},
            model_config=PromptModelConfig(
                provider="anthropic",
                model_name="claude-haiku-4-5-20251001",
                max_tokens=512,
                temperature=0.0,
            ),
        )
    except Exception as exc:
        LOGGER.warning("judge prompt registration skipped: %s", exc)


# ── scorer ─────────────────────────────────────────────────────────────────────


@mlflow.genai.scorer(
    name="solution_relevancy",
    description="Scores (1–10) how well the recruiter solution covers the candidate brief.",
    aggregations=["mean"],
)
def solution_relevancy(*, inputs: dict, outputs: str, **_: object) -> Feedback:
    """LLM-as-judge scorer backed by Claude Haiku and the registered judge prompt."""
    import re

    import anthropic

    brief = inputs["brief"]
    prompt_version = mlflow.genai.load_prompt(_JUDGE_PROMPT_NAME, allow_missing=True)
    prompt_text = (
        prompt_version.format(brief=brief, solution=outputs)
        if prompt_version is not None
        else _RELEVANCY_PROMPT.replace("{{brief}}", brief).replace("{{solution}}", outputs)
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt_text}],
    )
    raw = response.content[0].text.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    parsed = json.loads(match.group()) if match else json.loads(raw)
    score = float(parsed["score"])
    reasoning = str(parsed["reasoning"])

    LOGGER.info("  solution relevancy score: %.0f/10 — %s", score, reasoning)
    return Feedback(name="solution_relevancy", value=score, rationale=reasoning)
