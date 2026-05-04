"""MLflow tracking utilities for buc_factory agent runs."""

import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.genai

from .agent.entity import DomainConfig

LOGGER = logging.getLogger(__name__)

_EXPERIMENT = "buc-factory"
_task_log: list[dict] = []  # accumulated per-task results, reset each run

# Pricing in $ per 1M tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
_AGENT_MODEL = "claude-sonnet-4-6"


def _token_cost(input_tokens: int, output_tokens: int, model: str = _AGENT_MODEL) -> float:
    in_price, out_price = _MODEL_PRICING.get(model, _MODEL_PRICING[_AGENT_MODEL])
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def setup_mlflow(experiment_name: str = _EXPERIMENT) -> None:
    """Set experiment and configure artifact location.

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
    _register_judge_prompt()
    _task_log.clear()


def reset_run() -> None:
    """Clear per-run state. Must be called at the start of every agent run."""
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
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = _AGENT_MODEL,
) -> None:
    """Accumulate per-task outcome for the run summary table."""
    m, s = divmod(elapsed_s, 60)
    _task_log.append(
        {
            "task": task_name,
            "attempts": retries + 1 if ok else retries,
            "status": "✓" if ok else "✗",
            "duration": f"{int(m)}m {s:.1f}s",
            "elapsed_s": elapsed_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": _token_cost(input_tokens, output_tokens, model),
        }
    )


def log_tasks_summary(wall_clock_s: float | None = None) -> None:
    """Write a markdown table of task outcomes as an MLflow artifact."""
    if not _task_log:
        return
    total_in = sum(e["input_tokens"] for e in _task_log)
    total_out = sum(e["output_tokens"] for e in _task_log)
    total_cost = sum(e["cost"] for e in _task_log)

    cpu_s = sum(e["elapsed_s"] for e in _task_log)
    cm, cs = divmod(cpu_s, 60)
    cpu_str = f"{int(cm)}m {cs:.1f}s"

    if wall_clock_s is not None:
        wm, ws = divmod(wall_clock_s, 60)
        duration_cell = f"{int(wm)}m {ws:.1f}s wall / {cpu_str} cpu"
    else:
        duration_cell = cpu_str

    lines = [
        "# Task Summary\n",
        "| Task | Attempts | Status | Duration | Input tok | Output tok | Cost ($) |",
        "|------|:--------:|:------:|:--------:|----------:|-----------:|---------:|",
    ]
    for e in _task_log:
        lines.append(
            f"| {e['task']} | {e['attempts']} | {e['status']} | {e['duration']} |"
            f" {e['input_tokens']:,} | {e['output_tokens']:,} | ${e['cost']:.4f} |"
        )
    lines += [
        "|------|:--------:|:------:|:--------:|----------:|-----------:|---------:|",
        f"| **Total** | | | **{duration_cell}** |"
        f" **{total_in:,}** | **{total_out:,}** | **${total_cost:.4f}** |",
    ]
    try:
        mlflow.log_text("\n".join(lines) + "\n", "task_summary.md")
    except Exception as exc:
        LOGGER.warning("task summary logging skipped: %s", exc)


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


def evaluate_outputs(output_dir: Path) -> None:
    """Run the solution_relevancy LLM judge and log the score."""
    brief_file = output_dir / "brief/candidate_brief.md"
    solution_file = output_dir / "solution/recruiter_solution.md"
    if brief_file.exists() and solution_file.exists():
        try:
            import re

            from .llm.claudeai import AnthropicLLM

            brief_text = brief_file.read_text(encoding="utf-8")
            solution_text = solution_file.read_text(encoding="utf-8")
            prompt_text = _RELEVANCY_PROMPT.replace("{{brief}}", brief_text)
            prompt_text = prompt_text.replace("{{solution}}", solution_text)
            raw = AnthropicLLM().complete(
                prompt_text, model="claude-haiku-4-5-20251001", max_tokens=512
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else json.loads(raw)
            score = float(parsed["score"])
            reasoning = str(parsed["reasoning"])
            LOGGER.info("  solution relevancy score: %.0f/10 — %s", score, reasoning)
            mlflow.log_metric("solution_relevancy_score", score)
            mlflow.set_tag("solution_relevancy_reasoning", reasoning)
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

Write the reasoning in the same language as the brief and solution. \
Keep it to 2-3 sentences: what is well covered, what is missing or weak.

Respond with a JSON object only — no prose outside the JSON:
{"score": <integer 1-10>, "reasoning": "<2-3 sentences>"}

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
