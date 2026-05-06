"""
Industry-agnostic recruitment assessment agent — LangGraph edition.

Generates a complete BI assessment package (candidate brief, starter project,
recruiter solution) for any industry, defined via a DomainConfig YAML file.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m buc_factory --config conf/p&c_insurance_france.yml
    python -m buc_factory --config conf/p&c_insurance_france.yml --output-dir ./run_001  # resume
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from .agent.entity import PLAN, BucState, DomainConfig
from .agent.graph import build_graph
from .tracking import (
    evaluate_outputs,
    log_config,
    log_output_artifacts,
    log_tasks_summary,
    reset_run,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("log/agent.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


def _load_initial_state(output_dir: Path) -> BucState:
    """Return a fresh or resumed BucState from the output directory."""
    state_file = output_dir / "state.json"
    saved: dict = {}
    if state_file.exists():
        saved = json.loads(state_file.read_text())
        task_index = saved.get("task_index", 0)
        if task_index >= len(PLAN):
            LOGGER.info("✓ all sub-tasks already complete — nothing to do")
            sys.exit(0)
        LOGGER.info(f"resuming from task [{PLAN[task_index]}] (index {task_index})")
    else:
        task_index = 0

    return BucState(
        messages=[],
        output_dir=str(output_dir),
        task_index=task_index,
        current_task=PLAN[task_index],
        tasks_remaining=saved.get("tasks_remaining", PLAN[task_index:]),
        retry_count=0,
        bootstrapped_dimensions=saved.get("bootstrapped_dimensions"),
        bootstrapped_entities=saved.get("bootstrapped_entities"),
        scenario=saved.get("scenario"),
        rolled=None,
        validation_error=None,
        failed=False,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate a BI recruitment assessment package.")
    parser.add_argument("--config", required=True, type=Path, help="Path to domain YAML config.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Omit to use a temp dir (no resume).",
    )
    args = parser.parse_args()

    cfg = DomainConfig.from_yaml(args.config)

    setup_mlflow()
    run_name = f"{cfg.industry}__{cfg.role}".replace(" ", "_").lower()

    with ExitStack() as stack:
        if args.output_dir is not None:
            output_dir: Path = args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        with mlflow.start_run(run_name=run_name):
            log_config(cfg)
            mlflow.set_tags({"model": "claude-sonnet-4-6 (mixed)", "task_count": len(PLAN)})

            reset_run()
            LOGGER.info(f"agent: industry={cfg.industry!r}, role={cfg.role!r}, tool={cfg.tool!r}")
            LOGGER.info(f"  output: {output_dir}")

            initial_state = _load_initial_state(output_dir)
            graph = build_graph(output_dir, cfg)

            t0 = time.perf_counter()
            final_state: BucState = graph.invoke(initial_state)
            total = time.perf_counter() - t0

            m, s = divmod(total, 60)
            fmt_total = f"{int(m)}m {s:.1f}s"

            failed = bool(final_state.get("failed"))

            log_tasks_summary(wall_clock_s=total)
            log_output_artifacts(output_dir)

            if not failed:
                evaluate_outputs(output_dir)
                LOGGER.info(f"\n✓ all sub-tasks complete — total {fmt_total}")
            else:
                mlflow.set_tag("failure_task", final_state.get("current_task", "unknown"))
                mlflow.MlflowClient().set_terminated(
                    mlflow.active_run().info.run_id, status="FAILED"
                )
                LOGGER.error(f"\n✗ agent failed after {fmt_total}")
                sys.exit(1)


if __name__ == "__main__":
    main()
