import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

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
# DomainConfig: everything industry-specific lives here
# ──────────────────────────────────────────────────────────────────


@dataclass
class DomainConfig:
    """
    Defines a domain (industry + locale + role + tool).

    Required fields are explicit. Optional fields (dimensions, entities)
    default to None — the agent's bootstrap step will infer them at runtime.
    Fill them in if you want deterministic dimensions across runs.
    """

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
# State
# ──────────────────────────────────────────────────────────────────


@dataclass
class TaskState:
    name: str
    status: str = "pending"
    attempts: int = 0
    last_error: str | None = None


@dataclass
class AgentState:
    output_dir: str
    config_path: str
    bootstrapped_dimensions: dict | None = None
    bootstrapped_entities: list[str] | None = None
    scenario: dict | None = None
    tasks: list[TaskState] = field(default_factory=list)

    @classmethod
    def load_or_init(cls, output_dir: Path, config_path: Path) -> "AgentState":
        state_file = output_dir / "state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            return cls(
                output_dir=data["output_dir"],
                config_path=data["config_path"],
                bootstrapped_dimensions=data.get("bootstrapped_dimensions"),
                bootstrapped_entities=data.get("bootstrapped_entities"),
                scenario=data.get("scenario"),
                tasks=[TaskState(**t) for t in data["tasks"]],
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            output_dir=str(output_dir),
            config_path=str(config_path),
            tasks=[TaskState(name=n) for n in PLAN],
        )

    def save(self) -> None:
        Path(self.output_dir, "state.json").write_text(json.dumps(asdict(self), indent=2))

    def task(self, name: str) -> TaskState:
        return next(t for t in self.tasks if t.name == name)

    def next_pending(self) -> TaskState | None:
        return next((t for t in self.tasks if t.status != "done"), None)
