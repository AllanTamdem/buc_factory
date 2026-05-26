# buc-factory

![Use Case Factory](docs/logo_lockup.svg)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Powered by Claude](https://img.shields.io/badge/powered%20by-Claude-D97706?style=flat&logo=anthropic&logoColor=white)](https://anthropic.com/claude)
[![Powered by OpenAI](https://img.shields.io/badge/powered%20by-OpenAI-412991?style=flat)](https://openai.com)
[![SQLite](https://img.shields.io/badge/embeddings-SQLite-003B57?style=flat&logo=sqlite&logoColor=white)](https://sqlite.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)

An AI agent that generates complete, role-specific data recruitment assessments — candidate brief, synthetic datasets, a tool-specific starter project, and a recruiter solution — from a single YAML config file.

Each run produces a unique scenario by randomly combining business dimensions so no two assessments are identical.

> Documentation: **[🇺🇸 English](docs/specification_en.md)** · **[🇫🇷 Français](docs/specification_fr.md)**

---

## Installation

**Prerequisites:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo>
cd buc_factory
make install-dev
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

---

## Usage

### CLI

```bash
# New run
uv run python -m buc_factory --config conf/industry_spec/p&c_insurance_france.yml

# Persist outputs locally (enables resume on interruption)
uv run python -m buc_factory \
  --config conf/industry_spec/p&c_insurance_france.yml \
  --output-dir data/run_001
```

### API server

```bash
make api        # starts on http://localhost:8000 — docs at /docs
make ui         # Streamlit UI on http://localhost:8501
make mlflow     # MLflow UI on http://localhost:5000
```

### Docker Compose

```bash
make build
make up
```

| Service | URL |
|---------|-----|
| API | `http://localhost:8000/docs` |
| Streamlit UI | `http://localhost:8501` |
| MLflow | `http://localhost:5001` |

```bash
make logs         # tail all services
make down         # stop containers
make clean-docker # full teardown
```

---

## Project structure

```
buc_factory/
├── conf/industry_spec/          # domain YAML configs (one per industry/role)
├── src/buc_factory/
│   ├── conf/
│   │   ├── prompt_templates.yml              # system + task prompts (Claude)
│   │   └── prompt_templates_openai_patch.yml # overrides for OpenAI tasks
│   ├── __main__.py      # CLI entry point
│   ├── tracking.py      # MLflow helpers, LLM judge
│   ├── app/
│   │   ├── api.py       # FastAPI: runs, simulations, scoring, search
│   │   ├── ui.py        # Streamlit UI
│   │   ├── search.py    # HyDE semantic search + SQLite embeddings
│   │   ├── backfill.py  # index existing MLflow runs into search DB
│   │   └── models.py    # Pydantic request/response models
│   ├── agent/
│   │   ├── entity.py    # BucState, DomainConfig, PLAN
│   │   ├── graph.py     # LangGraph nodes and parallel groups
│   │   ├── prompting.py # prompt builder
│   │   ├── scorer.py    # LLM-based scoring (simulations + real submissions)
│   │   ├── tool.py      # agent tools (write_file, run_python, …)
│   │   └── validator.py # per-task output validators
│   └── llm/
│       ├── base.py      # BaseLLM, shared agentic loop
│       ├── claudeai.py  # Anthropic provider
│       └── gptai.py     # OpenAI provider
├── data/                # generated outputs, MLflow DB, embeddings (gitignored)
└── log/                 # rotating run logs (gitignored)
```

---

## Adding a new industry

1. Create `conf/industry_spec/<name>.yml` with `industry`, `company_context`, `location`, `language`, `role`, `seniority`, `tool`, `duration_minutes`, `deliverable_format`. Optionally pin `dimensions` and `entities`.
2. Point the agent at it. No code changes needed.

## Adding a new deliverable format

1. Add a case in `prompting._starter_prompt()`.
2. Add validation in `validator.validate_and_extract()` under `generate_starter`.
3. Add a prompt block in `prompt_templates.yml` under `tasks.generate_starter`.
