"""
Industry-agnostic recruitment assessment agent.

Generates a complete BI assessment package (candidate brief, starter project,
recruiter solution) for any industry, defined via a DomainConfig YAML file.

Usage:
    export ANTHROPIC_API_KEY=...
    python agent.py --config configs/insurance_fr.yaml --output-dir ./run_001
    python agent.py --config configs/retail_us.yaml --output-dir ./run_002
    python agent.py --config configs/insurance_fr.yaml --output-dir ./run_001  # resumes
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from .agent.entity import AgentState, DomainConfig
from .agent.prompting import build_system
from .agent.task import run_subtask
from .agent.tool import make_tools
from .llm.claudeai import AnthropicLLM

# Configure logging to display info messages in a readable format with timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("log/agent.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("buc_factory")


def main():

    load_dotenv()  # Load environment variables from .env file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cfg = DomainConfig.from_yaml(args.config)
    state = AgentState.load_or_init(args.output_dir, args.config)
    llm = AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tool_schemas, dispatch = make_tools(args.output_dir)
    system = build_system(cfg)

    LOGGER.info(f"agent: industry={cfg.industry}, role={cfg.role}, tool={cfg.tool}")
    LOGGER.info(f"  output: {args.output_dir}")

    t0 = time.perf_counter()
    while True:
        nxt = state.next_pending()
        if nxt is None:
            LOGGER.info("\n✓ all sub-tasks complete")
            break
        if nxt.status == "failed":
            LOGGER.error(f"\n✗ {nxt.name} failed after {nxt.attempts} attempts: {nxt.last_error}")
            sys.exit(1)
        if not run_subtask(
            llm, nxt.name, cfg, state, args.output_dir, tool_schemas, dispatch, system
        ):
            sys.exit(1)

    total = time.perf_counter() - t0
    LOGGER.info(f"\n=== final state (total: {total:.1f}s) ===")
    for t in state.tasks:
        LOGGER.info(f"  {t.status:8s} {t.name:30s} (attempts: {t.attempts})")


if __name__ == "__main__":
    main()

    # setup_mlflow()
    # with mlflow.start_run(run_name="example_run"):
    #     lr = 0.01
    #     acc = random.uniform(0.8, 0.95)

    #     mlflow.log_param("learning_rate", lr)
    #     mlflow.log_metric("accuracy", acc)

    # print("Run logged.")
