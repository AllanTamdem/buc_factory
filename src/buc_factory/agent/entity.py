from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import yaml
from langchain_core.messages import BaseMessage

# ──────────────────────────────────────────────────────────────────
# Plan
# ──────────────────────────────────────────────────────────────────

PLAN = [
    "bootstrap_domain",
    "roll_scenario",
    "write_brief",
    "design_data_schema",
    "generate_data_script",
    "generate_starter",
    "write_recruiter_solution",
    "final_assembly",
]

# ──────────────────────────────────────────────────────────────────
# DomainConfig
# ──────────────────────────────────────────────────────────────────


@dataclass
class DomainConfig:
    industry: str
    company_context: str
    location: str
    language: str
    role: str
    seniority: str
    tool: str
    duration_minutes: int
    dimensions: dict[str, list[str]] | None = None
    entities: list[str] | None = None
    deliverable_format: str = "PBIP"

    @classmethod
    def from_yaml(cls, path: Path) -> "DomainConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**data)


# ──────────────────────────────────────────────────────────────────
# LangGraph state
# ──────────────────────────────────────────────────────────────────


class BucState(TypedDict):
    messages: list[BaseMessage]  # current task conversation, reset per task
    output_dir: str
    task_index: int  # index into PLAN (0–7)
    current_task: str  # name of the task currently being executed
    tasks_remaining: list[str]  # task names not yet completed (including current)
    retry_count: int  # retries for the current task
    bootstrapped_dimensions: dict[str, list[str]] | None
    bootstrapped_entities: list[str] | None
    scenario: dict[str, str] | None
    rolled: dict[str, str] | None  # pre-rolled scenario values (roll_scenario task)
    validation_error: str | None  # last validator error; doubles as retry feedback
    failed: bool
