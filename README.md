# buc-factory

An AI agent that generates complete, industry-specific BI recruitment assessments — candidate brief, synthetic datasets, Power BI starter project, and recruiter solution — from a single YAML config file.

Each run produces a unique scenario by randomly combining business dimensions (analytical angle, data volume, industry twist, delivery format) so no two assessments are identical.

---

## What it produces

For every run the agent writes a self-contained output directory:

```
<output-dir>/
├── bootstrap.json                  # inferred/pinned dimensions and entity list
├── scenario.json                   # randomly rolled scenario values
├── state.json                      # resume checkpoint (task index + context)
│
├── brief/
│   ├── candidate_brief.md          # the assessment given to the candidate
│   └── data_schema.json            # CSV schema with column semantics and injected traps
│
├── starter/
│   ├── generate_data.py            # reproducible synthetic data generator (seed=42)
│   ├── data/
│   │   ├── <entity>.csv            # one CSV per entity (300-800 rows each)
│   │   └── ...
│   ├── Assessment.pbip             # Power BI project file (PBIP format)
│   ├── Assessment.SemanticModel/   # semantic model definition (TMDL)
│   └── Assessment.Report/         # report scaffold
│
└── solution/
    └── recruiter_solution.md       # full answer key with KPIs, DAX, trap explanations
```

---

## Agent workflow

The agent is a **LangGraph state machine** with five nodes that loop until all eight sub-tasks complete or a task exhausts its retry budget.

```
prepare_task → run_task → validate_task ──► prepare_task       (sequential retry / next)
                                        ├──► run_parallel_group ──► prepare_task
                                        │                        └──► fail_task ──► END
                                        ├──► fail_task ──► END
                                        └──► END                (all tasks done)
```

Two task pairs run concurrently via `ThreadPoolExecutor`:

- After `roll_scenario`: **`write_brief` ∥ `design_data_schema`**
- After `generate_data_script`: **`generate_starter` ∥ `write_recruiter_solution`**

### The eight sub-tasks (in order)

| # | Task | Model | What the model does | Validation |
|---|------|-------|---------------------|------------|
| 1 | `bootstrap_domain` | claude-sonnet-4-6 | Writes `bootstrap.json` with dimensions and entity list (inferred from industry or pinned from config) | Non-empty dimensions and ≥ 3 entities |
| 2 | `roll_scenario` | claude-haiku-4-5 | Writes `scenario.json` with one value randomly sampled per dimension (rolled in Python, not by the model) | All dimension keys present |
| 3 | `write_brief` | claude-sonnet-4-6 | Writes `brief/candidate_brief.md` in the target language, 600-900 words, anchored to the rolled scenario | File exists, ≥ 1 500 chars |
| 4 | `design_data_schema` | gpt-5.3-chat-latest | Writes `brief/data_schema.json` with column definitions, FK relationships, and 2-4 realistic data traps | ≥ 80% of entities covered, at least one trap declared |
| 5 | `generate_data_script` | o4-mini | Writes `starter/generate_data.py` (schema pre-loaded in prompt), runs it, verifies FK integrity | All entity CSVs present, FK heuristic passes |
| 6 | `generate_starter` | claude-opus-4-7 | Builds the tool-specific starter project under `starter/` (PBIP or Python notebook; schema pre-loaded in prompt) | Required files present and valid JSON |
| 7 | `write_recruiter_solution` | claude-opus-4-7 | Writes `solution/recruiter_solution.md` (brief, scenario, and schema pre-loaded in prompt) | File exists, ≥ 3 000 chars, contains DAX/calc keyword |
| 8 | `final_assembly` | claude-haiku-4-5 | Checks all expected artifacts are present | All required paths exist |

### Retry logic

Each task has a budget of **3 attempts** (`MAX_RETRIES_PER_TASK`). On failure the validator error is injected into the next attempt's prompt as feedback. After 3 failures the graph enters `fail_task` and halts.

### Resume / checkpointing

`state.json` is written after every successfully validated task. If a run is interrupted, re-running with the same `--output-dir` resumes from the last completed task — no work is lost.

---

## Configuration

### Domain config (YAML)

Every run requires a domain config file. Three examples are provided under `conf/industry_spec/`.

**Fully specified** (pinned dimensions and entities — deterministic across runs):

```yaml
# conf/industry_spec/p&c_insurance_france.yml
industry: "P&C insurance"
company_context: "A mid-sized Paris-based P&C insurer..."
location: "Paris, France"
language: "French"
role: "Data Analyst"
seniority: "Mid-Senior"
tool: "Power BI Desktop"
duration_minutes: 75
deliverable_format: "PBIP"

dimensions:
  branche: ["MRH (Multirisque Habitation)", "Auto particuliers", ...]
  angle: ["Pilotage de la sinistralité", ...]
  ...

entities:
  - "polices"
  - "sinistres"
  - "clients"
  - "produits"
  - "geographie"
```

**Minimal** (omit `dimensions` and `entities` — the model infers them during `bootstrap_domain`):

```yaml
# conf/industry_spec/retail_usa.yml
industry: "fashion retail (omnichannel)"
company_context: "A mid-sized US apparel retailer..."
location: "New York, NY"
language: "English"
role: "BI Analyst"
seniority: "Mid-Senior"
tool: "Power BI Desktop"
duration_minutes: 90
deliverable_format: "PBIP"
```

### Deliverable formats

| `deliverable_format` | Starter output |
|----------------------|----------------|
| `PBIP` | Power BI Desktop project (default) |
| `IPYNB` | Python Jupyter notebook (DA or DS variant, selected by `role`) |
| *(other)* | Generic file set |

### Prompt templates

System and task prompts live in `src/buc_factory/conf/prompt_templates.yml`. Provider-specific overrides are applied via a deep-merge patch file: `src/buc_factory/conf/prompt_templates_openai_patch.yml` is merged on top for tasks routed to OpenAI. Edit the base file (or the patch) to change tone, constraints, or evaluation criteria without touching Python code.

---

## Installation

**Prerequisites:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo>
cd buc_factory

uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

`.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## Usage

### CLI

```bash
# New run — outputs go to MLflow, no local directory kept
uv run python -m buc_factory \
  --config conf/industry_spec/p&c_insurance_france.yml

# Persist outputs locally and allow resume (--output-dir is optional)
uv run python -m buc_factory \
  --config conf/industry_spec/p&c_insurance_france.yml \
  --output-dir data/run_001

# Resume an interrupted run (same --output-dir)
uv run python -m buc_factory \
  --config conf/industry_spec/p&c_insurance_france.yml \
  --output-dir data/run_001
```

Logs are written to both the console and `log/agent.log`.

---

### API server

```bash
buc-factory-api
# or
uvicorn buc_factory.app.api:app --reload
```

The server starts on `http://localhost:8000`. Interactive docs are available at **`http://localhost:8000/docs`**.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/runs` | List all runs from MLflow with status, parameters, and scenario |
| `POST` | `/runs` | Submit a new agent run (returns `202` immediately) |
| `GET` | `/runs/{run_id}` | Run details from MLflow: status, parameters, scenario |
| `GET` | `/runs/{run_id}/recruiter.zip` | Download `brief/` + `solution/` (from MLflow artifacts) |
| `GET` | `/runs/{run_id}/candidate.zip` | Download `brief/` + `starter/` (from MLflow artifacts, without `generate_data.py`) |

#### Example: submit a run

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "industry": "assurance vie",
    "company_context": "Un assureur vie français de taille intermédiaire...",
    "location": "Paris, France",
    "language": "French",
    "role": "Data Analyst",
    "seniority": "Mid-Senior",
    "tool": "Power BI Desktop",
    "duration_minutes": 75,
    "deliverable_format": "PBIP",
    "dimensions": {
      "gamme": ["Fonds euros garanti", "Unités de compte (UC)"],
      "angle": ["Collecte nette", "Comportement de rachat"],
      "historique_mois": ["24", "36"],
      "volumetrie": ["100k_contrats", "500k_contrats"]
    },
    "entities": ["contrats", "clients", "versements", "rachats", "produits"]
  }'
# → {"run_id": "run_010", "status": "queued"}
```

`dimensions` and `entities` are optional — omit them to let the model infer them during `bootstrap_domain`.

#### Poll status and download

```bash
# Poll until "done"
curl http://localhost:8000/runs/run_010

# Download packages once complete
curl -O http://localhost:8000/runs/run_010/recruiter.zip
curl -O http://localhost:8000/runs/run_010/candidate.zip
```

---

### MLflow UI

Start the tracking server before (or after) running the agent:

```bash
bash .vscode/launch_mlflow.sh
# then open http://127.0.0.1:5000
```

The server uses a local SQLite backend (`mlflow_data/mlflow.db`) and stores artifacts under `mlflow_data/artifacts/`. Both paths are created automatically on first use.

---

### Docker

**CLI (one-shot run):**

```bash
docker build -t buc-factory .

docker run --env-file .env \
  -v $(pwd)/conf:/app/conf \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/log:/app/log \
  buc-factory \
  uv run python -m buc_factory \
    --config conf/industry_spec/p&c_insurance_france.yml \
    --output-dir data/run_001
```

**API server:**

```bash
docker run --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/conf:/app/conf \
  -v $(pwd)/log:/app/log \
  buc-factory \
  buc-factory-api
```

---

### Docker Compose

Compose provides three services: `mlflow` (tracking server), `api` (HTTP server), and `buc_factory` (CLI one-shot runner).

```bash
# Start MLflow + API server
docker compose up mlflow api -d

# API is now available at http://localhost:8000
# MLflow UI at http://localhost:5001

# Run a one-shot CLI assessment
docker compose run --rm buc_factory \
  --config conf/industry_spec/p&c_insurance_france.yml \
  --output-dir data/run_001
```

Add `--build` on first run or after code changes: `docker compose up --build mlflow api -d`.

All services inherit `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from `.env` and connect to MLflow over the internal Docker network (`http://mlflow:5000`).

---

## Project structure

```
buc_factory/
├── conf/
│   └── industry_spec/                      # domain YAML configs (one per industry)
│
├── src/buc_factory/
│   ├── conf/
│   │   ├── prompt_templates.yml                # base system + task prompts (Claude)
│   │   └── prompt_templates_openai_patch.yml   # deep-merge overrides for OpenAI tasks
│   ├── __main__.py             # CLI entry point, MLflow run, state loading, resume logic
│   ├── tracking.py             # MLflow helpers: metrics, artifacts, prompt registry, LLM judge
│   ├── app/
│   │   ├── api.py              # FastAPI app: POST /runs, GET /runs, zip download endpoints
│   │   └── models.py           # Pydantic I/O models: RunRequest, RunResponse, Dimensions, …
│   ├── agent/
│   │   ├── entity.py           # BucState (TypedDict) and PLAN (task order)
│   │   ├── graph.py            # LangGraph nodes: prepare_task, run_task, validate_task, fail_task
│   │   ├── prompting.py        # builds system prompt and per-task prompts from templates
│   │   ├── tool.py             # make_tools() — write_file, read_file, list_files, run_python, ...
│   │   └── validator.py        # validate_and_extract() — one validator per task
│   └── llm/
│       ├── base.py             # BaseLLM (ABC) — shared agentic loop, client cache
│       ├── claudeai.py         # AnthropicLLM — ChatAnthropic + cache_control system message
│       └── gptai.py            # OpenAILLM — ChatOpenAI + Anthropic→OpenAI tool schema adapter
│
├── mlflow_data/
│   ├── mlflow.db               # SQLite tracking backend (gitignored)
│   └── artifacts/              # run artifacts: prompts, outputs, task_summary.md (gitignored)
│
├── notebook/
│   └── claude_ai.ipynb         # standalone examples of AnthropicLLM usage
│
├── data/                       # generated output dirs (gitignored)
└── log/
    └── agent.log               # run logs
```

---

## Key internals

### LLM providers (`llm/`)

All task execution goes through `BaseLLM.run_agent_loop` — a shared agentic loop that invokes the model, dispatches tool calls, and repeats until the `mark_subtask_complete` tool fires or the step cap is reached:

```
invoke model → dispatch tool calls → append results → repeat
until: no tool calls | done_tool fired | max_steps reached
```

Returns `(list[BaseMessage], total_input_tokens, total_output_tokens)` — token counts are summed across every model call in the loop (including retries) and forwarded to MLflow.

Each task is routed to a specific model by `_TASK_MODEL` in `graph.py`; the provider is derived from the model name (`startswith("claude")` → Anthropic, otherwise → OpenAI). `AnthropicLLM` passes tool schemas as-is (Anthropic format) and adds `cache_control` to the system message. `OpenAILLM` converts schemas to OpenAI function-calling format and skips `temperature` for o-series reasoning models.

Token budgets by task:
- Default tasks: 8 000 tokens
- `generate_data_script`, `generate_starter`, `write_recruiter_solution`: 16 000 tokens (large file output)

### `make_tools()` (`agent/tool.py`)

Returns `(schemas, dispatch)`. The model only sees `run_python` during `generate_data_script`; all other tasks see the base tool set:

| Tool | Description |
|------|-------------|
| `write_file(path, content)` | Write text to a file inside the output dir |
| `read_file(path)` | Read a previously written file (not available for `bootstrap_domain`, `roll_scenario`, `write_brief`, `design_data_schema` — their context is pre-loaded in the prompt) |
| `list_files(directory)` | List files under a directory |
| `run_python(script_path)` | Execute a Python script; returns stdout/stderr/exit code (only available during `generate_data_script`) |
| `validate_csv_integrity(spec_json)` | Check FK integrity between a fact and dimension CSV (only available during `generate_data_script`) |
| `validate_json(path)` | Verify a file contains valid JSON |
| `mark_subtask_complete(summary)` | Signal task completion (exits the tool-use loop) |

### `BucState` (`agent/entity.py`)

LangGraph state dict carried across all nodes:

```python
class BucState(TypedDict):
    messages: list[BaseMessage]       # current task conversation (reset per task)
    output_dir: str
    task_index: int                   # index into PLAN (0–7)
    current_task: str                 # name of the running task
    tasks_remaining: list[str]        # tasks not yet completed
    retry_count: int
    bootstrapped_dimensions: dict | None
    bootstrapped_entities: list | None
    scenario: dict | None
    rolled: dict | None               # pre-rolled scenario values
    validation_error: str | None      # last validator error (fed back as retry prompt)
    failed: bool
```

### MLflow tracking (`tracking.py`)

Every agent run is wrapped in an `mlflow.start_run()`. The following are recorded automatically:

| What | MLflow location |
|------|----------------|
| Domain config fields (`industry`, `role`, `tool`, …) | Run parameters |
| Per-task success, duration, retries, input/output tokens | Run metrics (step = task index) |
| Total input tokens, output tokens, estimated cost (USD) | Run metrics |
| Task outcome table with per-task token counts and cost | Artifact `task_summary.md` |
| Task prompts (last attempt per task) | Artifacts under `prompts/` |
| System prompt | Prompt Registry (`buc-factory-system`) |
| All output files from `--output-dir` | Artifacts under `outputs/` |
| Solution relevancy score (1–10, GPT-5.3 LLM judge) | Metric `solution_relevancy_score`; reasoning in tag `solution_relevancy_reasoning` |

The judge prompt is registered in the Prompt Registry as `buc-factory-solution-relevancy-judge` with `PromptModelConfig` pointing to `gpt-5.3-chat-latest`. Using a cross-provider judge (OpenAI evaluating Claude output) avoids self-leniency bias. The prompt is role-aware: it extracts requirements differently for Power BI / Python DA vs Python DS deliverables.

---

## Adding a new industry

1. Create `conf/industry_spec/<name>.yml` — set `industry`, `company_context`, `location`, `language`, `role`, `seniority`, `tool`, `duration_minutes`, `deliverable_format`. Optionally pin `dimensions` and `entities`.
2. Run the agent pointing at the new config. No code changes needed.

## Adding a new deliverable format

1. Add a case in `prompting._starter_prompt()` for the new format identifier.
2. Add validation logic in `validator.validate_and_extract()` under `generate_starter`.
3. Add a prompt block in `src/buc_factory/conf/prompt_templates.yml` under `tasks.generate_starter`.
