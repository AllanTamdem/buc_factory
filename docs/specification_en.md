# BUC Factory — Complete Specification

![Use Case Factory](logo_lockup.svg)

## What is BUC Factory?

BUC Factory is an automated system that generates **data recruitment assessments**. From a single configuration file describing a role, an industry, and a tool, it produces everything needed for a technical interview:

- A **candidate brief** (the assessment prompt)
- Realistic, coherent **synthetic datasets**
- A ready-to-use **starter project** (Power BI or Python)
- A detailed **recruiter solution** with formulas, code, and a scoring rubric

Each generation is unique: themes, data volumes, business contexts, and data traps are randomly combined so no two candidates receive the same assessment.

---

## Table of Contents

1. [How it works — overview](#1-how-it-works--overview)
2. [Configuring an assessment](#2-configuring-an-assessment)
3. [The generation pipeline — 8 steps](#3-the-generation-pipeline--8-steps)
4. [Prompts and the template system](#4-prompts-and-the-template-system)
5. [The agent and its tools](#5-the-agent-and-its-tools)
6. [The Judge — automated solution evaluator](#6-the-judge--automated-solution-evaluator)
7. [The Candidate Simulator](#7-the-candidate-simulator)
8. [Scoring](#8-scoring)
9. [Submitting via the API](#9-submitting-via-the-api)
10. [The Streamlit interface](#10-the-streamlit-interface)
11. [Experiment tracking with MLflow](#11-experiment-tracking-with-mlflow)
12. [Semantic search](#12-semantic-search)
13. [Language models used](#13-language-models-used)
14. [Output file structure](#14-output-file-structure)
15. [Environment variables](#15-environment-variables)
16. [Deployment and infrastructure](#16-deployment-and-infrastructure)
    - [16.1 Local and Docker Compose](#161-local-and-docker-compose)
    - [16.2 Managed cloud — Render.com](#162-managed-cloud--rendercom)
    - [16.3 Managed cloud — Railway.com](#163-managed-cloud--railwaycom)
    - [16.4 OVH Public Cloud](#164-ovh-public-cloud)
    - [16.5 OVH AI Deploy](#165-ovh-ai-deploy)

---

## 1. How it works — overview

End-to-end flow:

```
YAML file (role configuration)
         │
         ▼
┌─────────────────────────────┐
│  Generation pipeline        │  ← 8 automated steps driven by LLMs
│  (LangGraph agent)          │
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
Candidate brief   Recruiter solution
CSV datasets      Power BI / Python starter project
     │
     ▼
┌─────────────────────────────┐
│  Candidate simulator        │  ← Simulates a real candidate completing the assessment
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Automated scoring          │  ← Grades the simulated work against the solution
└─────────────────────────────┘
```

Everything can be triggered from the CLI, the REST API, or the Streamlit web interface.

---

## 2. Configuring an assessment

Every run starts from a YAML file describing the company context and role.

### Example configuration file

```yaml
industry: "P&C insurance"
company_context: >
  A mid-sized Paris-based P&C insurer specializing in motor and home insurance.
  It manages 1.2 million active policies across France.
location: "Paris, France"
language: "French"
role: "Data Analyst"
seniority: "Mid-Senior"
tool: "Power BI Desktop"
duration_minutes: 75
deliverable_format: "PBIP"
```

### Field descriptions

| Field | Description | Example |
|---|---|---|
| `industry` | Industry sector | `"P&C insurance"`, `"Retail"` |
| `company_context` | Company description (2–4 sentences) | See above |
| `location` | City and country | `"Paris, France"` |
| `language` | Language for the brief and solution | `"French"`, `"English"` |
| `role` | Role being assessed | `"Data Analyst"`, `"Data Scientist"` |
| `seniority` | Seniority level | `"Junior"`, `"Mid-Senior"`, `"Senior"`, `"Staff"` |
| `tool` | Tool or platform | `"Power BI Desktop"`, `"Python (Notebook)"` |
| `duration_minutes` | Expected assessment duration | `75` |
| `deliverable_format` | Output format | `"PBIP"` (Power BI), `"IPYNB"` (Jupyter) |

### Optional dimensions and entities

**Dimensions** (the scenario axes) and **entities** (the data tables) can be pinned directly in the YAML:

```yaml
dimensions:
  branche: ["MRH", "Auto", "Santé"]
  angle: ["Claims monitoring", "Portfolio analysis"]

entities:
  - "policies"
  - "claims"
  - "customers"
```

If omitted, the system **infers them automatically** via the LLM during the first step. When both are provided, the inference step is bypassed entirely and the pipeline runs deterministically.

---

## 3. The generation pipeline — 8 steps

A complete assessment is generated through **8 sequential steps**, some running in parallel to save time. Each step is driven by a specific LLM and automatically validated before the next one begins.

```
Step 1: Domain bootstrap        (Claude Sonnet)
Step 2: Scenario roll           (Claude Haiku)
         ┌──────────────┐
Step 3:  │  Brief       │  ← parallel
Step 4:  │  Data schema │  ← parallel
         └──────────────┘
Step 5: Data script generation  (OpenAI o4-mini)
         ┌──────────────────────┐
Step 6:  │  Starter project     │  ← parallel
Step 7:  │  Recruiter solution  │  ← parallel
         └──────────────────────┘
Step 8: Final assembly          (Claude Haiku)
```

### Step 1 — Domain bootstrap

**Purpose:** Infer the scenario dimensions and business entities suited to the industry.

**Output:** `bootstrap.json`

```json
{
  "dimensions": {
    "branche": ["MRH", "Auto", "Health"],
    "angle": ["Claims monitoring", "Portfolio analysis"],
    "historique_mois": ["12", "24", "36"],
    "volumetrie": ["50k_policies", "120k_policies"],
    "twist": ["Parts inflation", "Rising fraud"],
    "restitution": ["Management dashboard", "Monthly report"]
  },
  "entities": ["policies", "claims", "customers", "agents", "guarantees"],
  "rationale": "These entities correspond to the core objects of a P&C insurer..."
}
```

**Note:** If `dimensions` and `entities` are already provided in the YAML, no LLM is called — the file is written directly.

---

### Step 2 — Scenario roll

**Purpose:** Randomly pick one value per dimension, building a unique context.

**Output:** `scenario.json`

```json
{
  "branche": "Private motor",
  "angle": "Claims monitoring",
  "historique_mois": "36",
  "volumetrie": "120k_policies",
  "twist": "Parts inflation",
  "restitution": "Technical management dashboard"
}
```

The roll is performed **in Python** before the LLM call, so retries are deterministic — the scenario does not change if the same step is re-attempted.

---

### Step 3 — Candidate brief

**Purpose:** Write the assessment prompt given to the candidate during the interview.

**Output:** `brief/candidate_brief.md`

The brief contains:
- Company and role context
- The candidate's mission
- KPIs to compute / visualisations to build
- Description of the available datasets
- Evaluation criteria

**Validation constraint:** The brief must be at least 1,500 characters long.

---

### Step 4 — Data schema design

**Purpose:** Define the precise structure of all CSV tables the candidate will work with.

**Output:** `brief/data_schema.json`

The schema describes each table's columns, types, business meaning, and most importantly the **data traps** — intentional anomalies hidden in the data to test the candidate's rigor.

**Example trap:**
```json
{
  "column": "claim_amount",
  "trap": "Amounts use '.' as a decimal separator but the file is encoded
           in fr-FR locale, which may cause type errors if the candidate
           does not adjust the Power Query locale setting."
}
```

**Validation constraint:** At least one trap declared and at least 80% of entities covered.

---

### Step 5 — Data script generation

**Purpose:** Write and execute a Python script that generates the synthetic CSV files.

**Output:** `starter/generate_data.py` + all `starter/data/*.csv` files

The script:
- Uses `seed=42` for reproducibility
- Generates 300 to 800 rows per table
- Respects inter-table relationships (foreign keys)
- Embeds the traps defined in step 4

**Automated validation:** The script is executed within the pipeline. The agent verifies all CSVs were created and that there are no orphan foreign keys.

---

### Step 6 — Starter project generation

**Purpose:** Create the project the candidate will receive and must complete.

**Output:** A `starter/` directory in Power BI (PBIP) or Jupyter (IPYNB) format

**For Power BI (PBIP):**
- `.pbip` entry point file
- TMDL v4 semantic model with columns, relationships, and partial measures
- Report scaffold with pages and visuals

**For Python (IPYNB):**
- Jupyter notebook with at least 5 cells
- Imports, data loading, and sections left for the candidate to complete

---

### Step 7 — Recruiter solution

**Purpose:** Write the full answer key the recruiter will use to evaluate the candidate.

**Output:** `solution/recruiter_solution.md`

The solution contains:
- All expected DAX formulas (Power BI) or Python code
- Explanations for each trap and how to detect it
- A scoring rubric with proficiency levels
- Suggested follow-up interview questions

**Validation constraint:** At least 3,000 characters and presence of concrete formulas or calculations.

---

### Step 8 — Final assembly

**Purpose:** Verify that all expected files are present before closing the generation.

**Required files:**
- `scenario.json`
- `bootstrap.json`
- `brief/candidate_brief.md`
- `brief/data_schema.json`
- `solution/recruiter_solution.md`
- `starter/generate_data.py`

---

### Parallelism — what it actually delivers

Two pairs of steps run in parallel via a Python `ThreadPoolExecutor` with 2 workers:

- **Group 1:** brief writing ∥ data schema design (steps 3 and 4)
- **Group 2:** starter project generation ∥ recruiter solution writing (steps 6 and 7)

In theory, running two tasks on two cores should cut the time in half. In practice, **the gain is small, sometimes imperceptible**. Here is why.

#### The real bottleneck: the LLM API, not the CPU

Each step spends 95% of its time **waiting for the LLM's response** — a network request to the Anthropic or OpenAI API. The local CPU does almost nothing during that wait. Adding more CPU cores changes nothing about that latency.

When two parallel tasks call the same API simultaneously, two effects arise:

1. **Token quota contention:** LLM APIs cap throughput in *tokens per minute* (TPM). Two concurrent tasks consume this quota twice as fast. If the quota is hit, one task is throttled — which cancels the parallelism benefit.

2. **Server-side latency:** A model like Claude Opus generates tokens sequentially. Processing two requests in parallel on Anthropic's side takes nearly as long as processing them sequentially, because GPU capacity is shared between both requests.

#### Why threads and not multiprocessing?

Python has a mechanism called the **GIL (Global Interpreter Lock)** that prevents two threads from executing pure Python code simultaneously within the same process. For CPU-intensive work, Python threads therefore do nothing in parallel.

This is not a problem here, because the tasks are *I/O-bound* (waiting on the network, not the CPU): Python releases the GIL automatically during network operations, allowing threads to truly coexist. But that does not change the API-side constraint.

Using `multiprocessing` (true parallel processes) would not help either — the bottleneck is the remote API, not Python.

#### When parallelism actually helps

Despite these limitations, parallelism has a real effect when:
- The two tasks call **different APIs** (e.g. steps 3+4, where one calls Claude Sonnet and the other calls GPT) — they do not share the same quota
- The TPM quota is generous (high-tier account) — both requests genuinely fire at the same time
- One of the two tasks is significantly shorter — it finishes while the longer one is still running

Under typical conditions, the observed real gain is **15 to 30%** off total pipeline duration, well short of the theoretical 50%.

---

### Retry mechanism

If a step fails (missing file, insufficient content, invalid JSON, etc.), the pipeline **automatically re-runs the same step**, injecting the exact validation error into the next attempt's prompt. This lets the LLM self-correct without human intervention.

**Maximum 3 attempts** per step. After that the pipeline halts and the run is marked as failed.

---

## 4. Prompts and the template system

### How a prompt works

Each step receives a **system prompt** (describing the role and context) and a **task prompt** (describing what is expected at that specific step).

The system prompt is built from the configuration file:

```
You are a senior data consultant specializing in the {industry} sector,
based in {location}, working with {tool}.
You write in {language}.
You are generating an assessment for a {role} position at {seniority} level.
```

### Prompt templates

All templates live in `src/buc_factory/conf/prompt_templates.yml`. They contain variables such as `{industry}`, `{language}`, `{scenario_json}`, etc., which are dynamically replaced at runtime.

A `prompt_templates_openai_patch.yml` file contains **OpenAI-specific overrides** — some phrasings work better with GPT than with Claude, and vice versa. These are deep-merged on top of the base file for tasks routed to OpenAI.

### Pre-loaded context

To avoid the agent reading files itself (which would be slow), prior step outputs are **injected directly into the prompt**:

- Step 3 (brief): receives `scenario.json` and `bootstrap.json`
- Step 5 (data): receives `data_schema.json` and the brief
- Step 6 (starter): receives `data_schema.json`
- Step 7 (solution): receives the brief, `scenario.json`, and `data_schema.json`

### Retry prompt suffix

When a step fails and must be retried, a suffix is appended to the prompt:

```
PREVIOUS ATTEMPT FEEDBACK:
The step failed with the following error:
[exact error message]
Address this issue in your next attempt.
```

### Prompt versioning in MLflow

The system prompt template is registered in the **MLflow Prompt Registry** under the name `buc-factory-system`. Each template change creates a new version, making it possible to trace which prompts produced which results.

---

## 5. The agent and its tools

### What is an agent?

An **agent** is an LLM equipped with tools (functions it can call) to accomplish a task. Rather than just generating text, the agent can read files, write files, execute code, and so on. It chooses which tools to use and in what order.

### Available tools

| Tool | Description |
|---|---|
| `write_file(path, content)` | Creates or overwrites a text file |
| `read_file(path)` | Reads a file's content |
| `list_files(directory)` | Lists all files in a directory |
| `run_python(script)` | Executes a Python script and returns its output |
| `validate_csv_integrity(...)` | Checks foreign key integrity between CSV tables |
| `validate_json(path)` | Verifies a file contains valid JSON |
| `mark_subtask_complete(summary)` | Signals task completion |

Not all tools are available at every step. For example, `run_python` is only accessible during the data generation step.

### The agentic loop

The agent operates in a loop:

```
1. LLM call → the model responds with tool calls
2. Tools are executed
3. Results are sent back to the LLM
4. → Repeat until the agent calls mark_subtask_complete
                          or exceeds 40 iterations
```

### LangGraph — the orchestrator

![Pipeline architecture](factory_architecture.svg)

The pipeline is implemented with **LangGraph**, a framework that models the workflow as a state graph:

```
prepare_task → run_task → validate_task ──► prepare_task (retry)
                                        ├──► run_parallel_group
                                        ├──► fail_task
                                        └──► END
```

Two groups of steps run in parallel (via a thread pool):
- Steps 3+4: brief and data schema
- Steps 6+7: starter project and recruiter solution

---

## 6. The Judge — automated solution evaluator

### What is the Judge?

After each generation, an automatic **Judge** evaluates the quality of the recruiter solution produced. It verifies that the solution concretely addresses every requirement in the brief.

### How it works

1. It reads the candidate brief (the assessment requirements)
2. It reads the generated recruiter solution
3. It extracts **all required elements** (KPIs, visualisations, data model, traps, scoring rubric)
4. For each element, it assesses whether the solution is **concrete** (with formulas, code, precise explanations) or **superficial**
5. It assigns a score from 1 to 10 and writes a justification

### Judge scoring rubric

| Score | Meaning |
|---|---|
| 9–10 | Every element covered with concrete formulas / code |
| 7–8 | All main elements covered; 1–2 items remain thin |
| 5–6 | Core analytical elements present but rubric or interview questions missing |
| 3–4 | Only a subset covered; data model or preparation missing |
| 1–2 | Most elements absent |

### Model used

The Judge uses an **OpenAI GPT model** (rather than Anthropic) to avoid self-leniency bias — a Claude model does not grade the output of another Claude model.

The Judge prompt is also versioned in the MLflow Prompt Registry under `buc-factory-solution-relevancy-judge`.

---

## 7. The Candidate Simulator

### What is the simulator for?

The simulator tests a generated assessment **by simulating a real candidate completing it**. Use cases:
- Verify the assessment is completable within the allotted time
- Test the automated scoring
- Produce example outputs at different proficiency levels

### Proficiency levels

The simulator receives a `proficiency` parameter between 0.0 and 1.0 that determines the simulated candidate's level:

| Level | Threshold | Behaviour |
|---|---|---|
| Expert | ≥ 0.90 | Completes EVERYTHING accurately, clean code, no errors |
| Senior | ≥ 0.70 | Misses ~(1−score)% of subtle requirements, 1–2 minor bugs, 1 missed trap |
| Mid-level | ≥ 0.50 | Completes ~score% of requirements, 1–6 TODOs, 1–2 bugs |
| Junior-Mid | ≥ 0.35 | Many TODOs, multiple bugs, falls into most traps |
| Junior | < 0.35 | Attempts only 1–2 basic tasks, mostly placeholders |

### Two simulation modes

**`perfect` mode:** the simulator always plays an expert candidate (proficiency = 1.0). Useful for quickly validating that an assessment is complete.

**`random` mode:** the simulator draws a proficiency value randomly (or receives one explicitly). A `seed` parameter can be provided to reproduce an exact simulation.

### How the simulator works

Like the generation agent, the simulator is an LLM with access to the `write_file`, `read_file`, and `list_files` tools. It receives:

1. The starter project (the Power BI file or notebook to complete)
2. The candidate brief
3. Behavioural instructions tailored to its level (persona + rules)

**Example instructions for a Mid-level candidate (60%):**
```
You are an intermediate-level data analyst (60% proficiency).
- Complete 60% of the brief's requirements
- Leave 3 sections with a TODO comment
- Introduce 1–2 minor bugs in the code
- Fall into 1–2 data traps without correcting them
```

The simulator modifies the starter project (writing directly into TMDL files or the notebook) to produce a realistic output.

### Simulation validation

A simulation is only accepted if the starter project has been **substantially modified**. An attempt that makes almost no changes is rejected and retried (up to 3 times).

---

## 8. Scoring

### What is scoring for?

Scoring automatically evaluates a candidate's work against the recruiter solution. It operates in **two distinct use cases**, both using the same `score_submission()` function:

1. **After a simulation** — scoring is triggered automatically once the simulator has completed the starter project
2. **On a real submission** — a recruiter uploads the ZIP delivered by a real candidate, and the system grades it immediately

### The scoring function

`score_submission(output_dir, recruiter_solution, deliverable_format)`:

1. Reads `brief/candidate_brief.md` to extract requirements
2. Collects the candidate's work based on the format:
   - **PBIP:** reads all `.tmdl` files from the semantic model + all `.json` files from report pages
   - **IPYNB:** reads `notebook.ipynb` directly
3. Reads the recruiter solution (supplied as text)
4. Truncates each section to **12,000 characters** maximum to fit within the LLM's context window
5. Calls Claude Opus with a prompt that compares the work to the requirements, using the solution as a reference
6. Returns a structured markdown report

### What the report contains

- An overall grade summary
- For each brief requirement: covered, partially covered, or absent
- Traps detected or missed by the candidate
- Suggestions for the debrief interview

### Model used

**Claude Opus 4.7** (the most powerful model), with automatic fallback to **GPT-5.5** (OpenAI) if the Anthropic API is unavailable.

### Submitting a real candidate's solution

Beyond automated simulations, a recruiter can upload a real candidate's delivered work for grading:

```http
POST /runs/{run_id}/score
Content-Type: multipart/form-data

solution: <.zip file>
```

The ZIP must contain the candidate's completed project — either the Power BI structure (PBIP) or the Python notebook (IPYNB). A ZIP wrapped inside a single root folder is automatically unwrapped.

**Processing steps:**
1. Validate the file is a valid ZIP
2. Retrieve the expected format from MLflow (the run's `deliverable_format` parameter)
3. Extract the ZIP into a temporary `starter/` directory
4. Download the brief and recruiter solution from MLflow
5. Call `score_submission()` with these files
6. Return the scoring report as markdown (`200 OK`, `text/plain`)

This feature allows real candidates to be evaluated with the same rigor as an automated simulation.

---

## 9. Submitting via the API

### Overview

BUC Factory exposes a **REST API** (FastAPI) on port 8000 for submitting generation requests, tracking them, and retrieving results.

**Interactive docs:** `http://localhost:8000/docs`

### Submit a generation

```http
POST /runs
Content-Type: application/json

{
  "industry": "P&C insurance",
  "company_context": "A mid-sized P&C insurer...",
  "location": "Paris, France",
  "language": "French",
  "role": "Data Analyst",
  "seniority": "Mid-Senior",
  "tool": "Power BI Desktop",
  "duration_minutes": 75,
  "deliverable_format": "PBIP"
}
```

**Immediate response (202 Accepted):**
```json
{
  "run_id": "run_042",
  "status": "queued"
}
```

The generation runs **in the background** — the API responds immediately without waiting for the pipeline to finish.

### Track progress

```http
GET /runs/run_042
```

```json
{
  "run_id": "run_042",
  "status": "running",
  "scenario": {
    "branche": "Private motor",
    "angle": "Claims monitoring"
  }
}
```

Possible statuses: `queued`, `running`, `done`, `failed`.

### Retrieve deliverables

| Endpoint | Content |
|---|---|
| `GET /runs/{id}/brief` | Candidate brief (markdown) |
| `GET /runs/{id}/solution` | Recruiter solution (markdown) |
| `GET /runs/{id}/candidate.zip` | Brief + starter project (without the solution) |
| `GET /runs/{id}/recruiter.zip` | Brief + full solution |
| `POST /runs/{id}/score` | Upload a real candidate's ZIP for grading |

### Submit a simulation

```http
POST /simulations
Content-Type: application/json

{
  "run_id": "run_042",
  "mode": "random",
  "proficiency": 0.65,
  "seed": 42
}
```

**Response:**
```json
{
  "simulation_id": "sim_007",
  "status": "queued"
}
```

### Retrieve simulation results

| Endpoint | Content |
|---|---|
| `GET /simulations/{id}` | Status, mode, proficiency |
| `GET /simulations/{id}/scoring` | Scoring report (markdown) |
| `GET /simulations/{id}/solution.zip` | Completed starter project |

### Search past generations

```http
GET /runs/search?q=claims+motor+Power+BI&limit=5
```

The search uses semantic embeddings — it understands the meaning of the query, not just exact keywords. The `use_hyde=true` parameter (default) improves precision for short or ambiguous queries.

---

## 10. The Streamlit interface

The web interface is accessible on port 8501 (`http://localhost:8501`).

### What it does

**Submit an assessment:** form with all configuration fields, pre-filled defaults, and a randomisation button for quickly exploring varied profiles (15+ pre-configured industry and country profiles).

**Browse existing assessments:** list of all past generations with real-time status (color-coded), with the ability to download ZIP archives.

**View assessment details:** parameters, rolled scenario, step-by-step status.

**Launch a simulation:** pick an existing assessment, a mode, and a proficiency level, then track progress.

**Search:** semantic search bar across all generated assessments.

---

## 11. Experiment tracking with MLflow

MLflow traces **every generation and simulation** in an experiment journal.

### What is recorded per run

**Parameters** (what was requested):
- industry, role, tool, language, seniority, location, duration_minutes, deliverable_format

**Metrics** (what happened):
- Duration of each step (seconds)
- Token consumption per step (input + output)
- Estimated cost per step (USD)
- Number of attempts per step
- Solution relevancy score (1–10)

**Artifacts** (files produced):
- All generated assessment files
- Task summary (`task_summary.md`)
- Prompts used at each step (`prompts/`)

**Tags:**
- API run identifier (`api_run_id`)
- Score justification (`solution_relevancy_reasoning`)

### The two MLflow experiments

| Experiment | Content |
|---|---|
| `buc-factory` | All assessment generations |
| `buc-factory-simulations` | All candidate simulations |

### Cost calculation

Costs are estimated automatically from token counts and official model pricing. Example:

```
Claude Opus 4.7:   $5.00 / million input tokens, $25.00 output
Claude Sonnet 4.6: $3.00 / million input tokens, $15.00 output
Claude Haiku 4.5:  $1.00 / million input tokens,  $5.00 output
```

---

## 12. Semantic search

### Why semantic search?

Classic keyword search (SQL LIKE) only finds exact matches. Semantic search understands the **meaning** of a query: "motor claims France" will find an assessment titled "P&C motor claims management in France", even if no word matches exactly.

### Architecture

```
User query
      │
      ▼ (if use_hyde=True, enabled by default)
GPT-4o-mini generates a fictional description of a matching run
      │
      ▼
text-embedding-3-small → 1536-dimensional vector
      │
      ▼
Cosine similarity against all vectors stored in SQLite
      │
      ▼
Top-k results ranked by score
```

### Step 1 — Building the index text

After each successful generation, an **index text** is built and stored alongside its vector:

```
"A Mid-Senior Data Analyst working in the P&C insurance industry,
using Power BI Desktop, based in Paris, speaking French.
Scenario: branche: Private motor, angle: Claims monitoring,
historique_mois: 36, volumetrie: 120k_policies, twist: Parts inflation"
```

This text combines the run parameters (role, industry, tool, location, language) and the rolled scenario. This is what gets converted to a vector and stored.

### Step 2 — Embedding the index text

The text is sent to the **OpenAI `text-embedding-3-small`** model (1,536 dimensions). The resulting vector is JSON-serialised and stored in SQLite:

```sql
CREATE TABLE run_embeddings (
    api_run_id    TEXT PRIMARY KEY,
    mlflow_run_id TEXT,
    embed_text    TEXT,       -- human-readable index text
    embedding     TEXT        -- JSON vector [0.023, -0.41, ...]
)
```

### Step 3 — HyDE (Hypothetical Document Embeddings)

The challenge with vector search is that the user query ("data analyst motor insurance") and the indexed text ("A Mid-Senior Data Analyst working in the P&C insurance industry...") are **not in the same semantic space** — one is a short query, the other is a run description.

HyDE solves this in two steps:

**1. Generating a hypothetical document**

The raw query is sent to GPT-4o-mini with this prompt:
```
"Write one sentence describing a BUC coaching run for: {query}.
Mention role, seniority, industry, tool, location, and language where relevant."
```

Example — query `"data analyst motor insurance power bi"` → hypothetical description:
```
"A Mid-Senior Data Analyst in the automotive insurance industry
using Power BI Desktop, based in France, speaking French."
```

**2. Embedding the hypothetical document**

It is this richer description (in the same format as indexed texts) that is converted to a vector — not the raw query. This aligns the vector spaces and dramatically improves relevance for short or ambiguous queries.

To disable HyDE and query on the raw input directly:
```
GET /runs/search?q=data+analyst+insurance&use_hyde=false
```

### Step 4 — Cosine similarity

The search loads all stored vectors, then computes the **cosine similarity** between the query vector and each indexed vector:

```
score = (query_vector · document_vector) / (|query_vector| × |document_vector|)
```

In practice, both vectors are L2-normalised, reducing the computation to a simple dot product. NumPy is used to vectorise this across the entire database in a single matrix operation.

A score of 1.0 means a perfect match, 0.0 means no correlation. In practice, relevant scores fall between 0.7 and 0.95.

### Search endpoint

```http
GET /runs/search?q=data+scientist+retail+python&limit=5&use_hyde=true
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `q` | required | Free-text query |
| `limit` | 10 | Number of results (max 100) |
| `use_hyde` | `true` | Enable HyDE expansion |

**Response:** list of `SearchResult` sorted by descending score, each containing: `score` (float), `run_id`, `status`, `parameters`, `scenario`.

### Backfill — indexing existing runs

Runs generated before the search feature was added are not indexed. The following command indexes them retroactively:

```bash
uv run buc-factory-backfill
```

**Process:**
1. Fetches all `FINISHED` MLflow runs with an `api_run_id` tag
2. Filters out runs already in the `run_embeddings` table
3. For each missing run: downloads `scenario.json` from MLflow, builds the index text, computes the embedding, stores in SQLite
4. Prints a summary: X indexed, Y already present

Requires `MLFLOW_TRACKING_URI` and `OPENAI_API_KEY`.

---

## 13. Language models used

BUC Factory uses **multiple models** across steps, selecting the best quality/cost trade-off for each task.

| Step | Primary model | Reason |
|---|---|---|
| Domain bootstrap | Claude Sonnet | Strong business understanding, good value |
| Scenario roll | Claude Haiku | Simple task, fast and cheap |
| Candidate brief | Claude Sonnet | Fluent multilingual writing |
| Data schema | GPT (OpenAI) | Structured schema design |
| Data script | o4-mini (OpenAI) | Robust Python code generation |
| Starter project | Claude Opus | Complex task, long files, high quality required |
| Recruiter solution | Claude Opus | Long dense content, precise formulas required |
| Final assembly | Claude Haiku | Simple verification, low cost |
| Scoring | Claude Opus | Nuanced and detailed evaluation |
| Judge (evaluation) | GPT (OpenAI) | Cross-provider bias: Anthropic model does not grade its own output |

### Automatic Anthropic → OpenAI fallback

If the Anthropic API becomes unavailable (quota exceeded, billing issue, missing key), **all Claude calls automatically switch** to the OpenAI equivalent without interrupting the pipeline:

| Claude | OpenAI substitute |
|---|---|
| Claude Haiku | GPT mini |
| Claude Sonnet | GPT standard |
| Claude Opus | GPT premium |

---

## 14. Output file structure

Complete directory tree of a successful generation:

```
output-dir/
├── bootstrap.json          ← domain dimensions and entities
├── scenario.json           ← randomly rolled scenario
├── state.json              ← resume checkpoint
│
├── brief/
│   ├── candidate_brief.md  ← assessment prompt given to the candidate
│   └── data_schema.json    ← table structure + traps
│
├── starter/
│   ├── generate_data.py    ← data generation script (seed=42)
│   ├── data/
│   │   ├── policies.csv
│   │   ├── claims.csv
│   │   └── customers.csv
│   │
│   │   ── Power BI (PBIP) ──
│   ├── Assessment.pbip
│   ├── Assessment.SemanticModel/
│   │   └── definition/
│   │       ├── model.tmdl
│   │       └── tables/
│   │           ├── policies.tmdl
│   │           └── claims.tmdl
│   └── Assessment.Report/
│       └── pages/...
│
│   │   ── Python (IPYNB) ──
│   ├── notebook.ipynb
│   └── requirements.txt
│
└── solution/
    └── recruiter_solution.md   ← full answer key + scoring rubric
```

### The checkpoint (`state.json`)

After each validated step, `state.json` is updated. If the generation is interrupted (crash, timeout), it can be **resumed from where it left off** by rerunning with the same output directory:

```bash
python -m buc_factory --config conf/... --output-dir data/run_001
# If state.json already exists in data/run_001, the generation resumes at the next step
```

---

## 15. Environment variables

Configure in a `.env` file at the project root.

### Required

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic API key (for Claude)
OPENAI_API_KEY=sk-...          # OpenAI API key (for GPT and embeddings)
```

### Optional

```bash
# MLflow server (default: local SQLite)
MLFLOW_TRACKING_URI=http://mlflow:5000

# Semantic search database path
EMBEDDINGS_DB_PATH=data/embeddings.db

# Cloud storage for MLflow artifacts
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MLFLOW_S3_ENDPOINT_URL=...
ARTIFACT_BUCKET=...
```

---

## 16. Deployment and infrastructure

### Two ways to deploy

BUC Factory can be deployed in two different ways depending on the context:

**A — From the GitHub repository (build on the fly)**
- The platform clones the repo and builds the Docker image itself
- Advantage: simple to configure, no external registry needed
- Drawback: build takes 3–5 minutes per deployment; the repo must be accessible from the platform
- Prerequisites: access to the GitHub repository (public, or access granted to the platform)

**B — From the pre-built CI/CD image**
- GitHub Actions builds and publishes `amd64` and `arm64` images
- The `.tar.gz` archives contain the Docker image and configuration files
- Advantage: instant deployment, versioned and tested image, no source code access required
- Drawback: requires a Docker registry or unpacking the archive on the server
- Prerequisites: download the GitHub release archive (`buc-factory-stack-amd64.tar.gz` or `buc-factory-stack-macos-arm64.tar.gz`)

---

### 16.1 Local and Docker Compose

**Prerequisites:**
- Python ≥ 3.12 + [uv](https://docs.astral.sh/uv/) (development)
- Docker Desktop (Docker Compose)
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`

**Development:**
```bash
uv sync
cp .env.example .env   # fill in API keys
uv run python -m buc_factory --config conf/industry_spec/p_and_c_france.yml --output-dir data/run_001
uv run buc-factory-api   # port 8000
uv run buc-factory-ui    # port 8501
```

**Docker Compose (recommended for local use):**
```bash
make build
make up
```

| Service | Port | Role |
|---|---|---|
| `mlflow` | 5001 | Experiment tracking + artifacts |
| `api` | 8000 | FastAPI REST server |
| `ui` | 8501 | Streamlit interface |

**From the pre-built release archive:**
```bash
tar -xzf buc-factory-stack-amd64.tar.gz
cp .env.example .env   # fill in API keys
docker load -i buc-factory-image.tar
docker compose up
```

---

### 16.2 Managed cloud — Render.com

Render simplifies deployment without managing servers. The minimal architecture requires **two services**: the API and MLflow.

#### Prerequisites
- Render account (Starter plan at $7/month minimum for persistent disks)
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`
- Access to the GitHub repository **or** pre-built image archive

#### Service 1 — MLflow

MLflow stores experiments and artifacts. It must be deployed first since the API depends on it.

**From the pre-built image:**
1. New → Web Service → **Deploy an existing image**
2. Image: `ghcr.io/mlflow/mlflow:v3.12.0`
3. **Docker Command**:
   ```
   mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --serve-artifacts --artifacts-destination /mlflow/artifacts
   ```
4. Add a **Disk** mounted at `/mlflow` (512 MB minimum) — without persistent storage, all data is lost on every redeploy
5. Use `$PORT`, not a hardcoded port — Render injects its own value via the `PORT` environment variable

Once deployed, note the public URL (e.g. `https://mlflow-xxxx.onrender.com`).

**Render limitations for MLflow:**
- The Starter plan (512 MB RAM) is insufficient for MLflow v3 — use the **Standard plan (2 GB RAM, ~$25/month)**
- Render does not support persistent disks on the free plan

#### Service 2 — buc-factory API

**From the GitHub repo:**
1. New → Web Service → **Connect a repository** → select `buc_factory`
2. Runtime: **Docker** (Render detects the `Dockerfile` automatically)
3. Environment variables to configure in the dashboard:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `OPENAI_API_KEY` | `sk-...` |
| `MLFLOW_TRACKING_URI` | `https://mlflow-xxxx.onrender.com` |
| `EMBEDDINGS_DB_PATH` | `/app/data/embeddings.db` |

4. Add a **Disk** mounted at `/app/data` to persist the embeddings database and local artifacts

**From the pre-built image:**
1. New → Web Service → **Deploy an existing image**
2. Reference the image published in a registry (GitHub Container Registry, Docker Hub)
3. Configure the same environment variables

#### Key differences between the two modes on Render

| | From the repo | From the pre-built image |
|---|---|---|
| Deployment time | 3–5 min (build included) | <1 min |
| Source code access required | Yes | No |
| Updates | Automatic on each push | Manual (new image) |
| Version control | Branch/commit | Image tag |

---

### 16.3 Managed cloud — Railway.com

Railway offers a similar experience to Render, with usage-based pricing instead of fixed plans. Particularly well-suited for MLflow thanks to more generous default RAM allocation.

#### Prerequisites
- Railway account (Hobby plan at $5/month, includes $5 of usage credits)
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`
- Access to the GitHub repository **or** a published Docker image

#### Service 1 — MLflow on Railway

1. New Project → **Deploy from Docker image**
2. Image: `ghcr.io/mlflow/mlflow:v3.12.0`
3. **Start command**:
   ```
   mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --serve-artifacts --artifacts-destination /mlflow/artifacts
   ```
4. Add a **Volume** mounted at `/mlflow` for persistence
5. Railway allocates RAM dynamically — MLflow starts without issues on the Hobby plan

#### Service 2 — buc-factory API on Railway

**From the GitHub repo:**
1. New Project → **Deploy from GitHub repo** → select `buc_factory`
2. Railway auto-detects the `Dockerfile`
3. Configure environment variables under Settings → Variables

**From the pre-built image:**
1. New Project → **Deploy from Docker image**
2. Reference the image from a public or private registry

#### Railway advantages over Render

- More generous RAM by default (512 MB to 8 GB depending on usage)
- Automatic scale-to-zero (no cost when there is no traffic)
- Native PostgreSQL provisioning (alternative to SQLite for MLflow in production)
- Private networking between services in the same project (no need to expose MLflow publicly)

#### Using PostgreSQL for MLflow on Railway

Railway can provision a PostgreSQL database in one click. To use it with MLflow:

1. New → **PostgreSQL** in the same Railway project
2. Retrieve the automatically generated `DATABASE_URL` variable
3. Update the MLflow command:
   ```
   mlflow server --host 0.0.0.0 --port $PORT \
     --backend-store-uri $DATABASE_URL \
     --default-artifact-root /mlflow/artifacts \
     --serve-artifacts
   ```

This avoids SQLite's single-writer limitation for environments with multiple concurrent runs.

---

### 16.4 OVH Public Cloud

OVH Public Cloud is OVH's IaaS offering: compute instances (CPU/RAM), storage, and networking, with no managed abstraction. This is the **most flexible but most manual** deployment mode.

#### Prerequisites
- OVH Public Cloud account
- **B2-7 instance** minimum (2 vCPU, 7 GB RAM) to run MLflow + API together
- **B2-15 instance** recommended (4 vCPU, 15 GB RAM) for production
- Public IP address attached to the instance
- Docker and Docker Compose installed on the instance
- Domain name (optional but recommended for HTTPS)

#### Deployment from the pre-built image (recommended)

This is the cleanest approach: no source code is exposed on the server and the deployed version has been tested by CI.

```bash
# On the local machine — download the release archive
wget https://github.com/<org>/buc_factory/releases/download/v1.x.x/buc-factory-stack-amd64.tar.gz

# Transfer to the OVH server
scp buc-factory-stack-amd64.tar.gz ubuntu@<ovh-ip>:/opt/buc-factory/

# On the OVH server
cd /opt/buc-factory
tar -xzf buc-factory-stack-amd64.tar.gz

# Load the image into Docker
docker load -i buc-factory-image.tar

# Configure environment variables
cp .env .env.prod
nano .env.prod   # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.

# Start
docker compose --env-file .env.prod up -d
```

The archive already contains a `compose.yml` and an `.env` template — MLflow is pulled automatically from `ghcr.io` on the first `docker compose up`.

#### Deployment from the GitHub repo

```bash
# On the OVH server
git clone https://github.com/<org>/buc_factory.git /opt/buc-factory
cd /opt/buc-factory
cp .env.example .env
nano .env   # fill in API keys

docker compose build
docker compose up -d
```

#### Persistent storage on OVH

To avoid losing MLflow data and embeddings between restarts, mount an **OVH Block Storage volume**:

1. Create a volume in the OVH console → Block Storage → Attach to instance
2. Format and mount the volume:
   ```bash
   mkfs.ext4 /dev/sdb
   mount /dev/sdb /mnt/buc-data
   echo "/dev/sdb /mnt/buc-data ext4 defaults 0 2" >> /etc/fstab
   ```
3. Update the volume paths in `compose.yml` to point to `/mnt/buc-data`

#### HTTPS with Nginx and Let's Encrypt

```bash
apt install nginx certbot python3-certbot-nginx
certbot --nginx -d buc-factory.yourdomain.com
```

Configure Nginx as a reverse proxy to `localhost:8000` (API) and `localhost:8501` (UI).

---

### 16.5 OVH AI Deploy

**OVH AI Deploy** is OVH's serverless platform for deploying containerised applications with optional GPU acceleration. It suits BUC Factory well as it handles scaling and HTTPS automatically.

#### Differences from OVH Public Cloud

| | OVH Public Cloud | OVH AI Deploy |
|---|---|---|
| Server management | Manual (SSH, Docker) | None |
| Scaling | Manual | Automatic |
| HTTPS | Configure manually (Nginx) | Included |
| GPU available | No (CPU instances) | Yes (NVIDIA T4, A100) |
| Pricing | Fixed instance ($/hour) | Usage-based (per run hour) |
| Data persistence | Block Storage volume | Object Storage (Swift/S3) |

#### Prerequisites
- OVH Cloud account with AI Deploy access
- Docker image published in a registry (OVH Managed Registry, GitHub Container Registry, Docker Hub)
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`
- OVH Object Storage bucket for MLflow artifacts (replaces the local volume)

#### Publish the image to OVH Managed Registry

```bash
# Build and tag the image
docker build -t <region>.registry.ovh.net/<namespace>/buc-factory:latest .

# Authenticate and push
docker login <region>.registry.ovh.net
docker push <region>.registry.ovh.net/<namespace>/buc-factory:latest
```

Alternatively, the CI-built image can be pushed directly from GitHub Actions to OVH Managed Registry.

#### Deploy the API on AI Deploy

Via the OVH console or the `ovhai` CLI:

```bash
ovhai app run \
  --name buc-factory-api \
  --image <region>.registry.ovh.net/<namespace>/buc-factory:latest \
  --cpu 4 \
  --memory 8Gi \
  --env ANTHROPIC_API_KEY=sk-ant-... \
  --env OPENAI_API_KEY=sk-... \
  --env MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com \
  --env MLFLOW_S3_ENDPOINT_URL=https://s3.<region>.io.cloud.ovh.net \
  --env AWS_ACCESS_KEY_ID=<ovh-s3-key> \
  --env AWS_SECRET_ACCESS_KEY=<ovh-s3-secret> \
  --env ARTIFACT_BUCKET=buc-factory-artifacts \
  --port 8000
```

#### MLflow with Object Storage (OVH S3)

On AI Deploy there is no directly attachable persistent disk. MLflow must use:
- **Backend store:** OVH Managed PostgreSQL (Cloud Databases → PostgreSQL)
- **Artifact store:** OVH Object Storage (S3-compatible)

MLflow configuration for this mode:
```bash
mlflow server \
  --host 0.0.0.0 \
  --port $PORT \
  --backend-store-uri postgresql://user:pass@host:5432/mlflow \
  --default-artifact-root s3://buc-factory-artifacts/ \
  --serve-artifacts
```

#### Deployment options summary

| Platform | Setup effort | Est. monthly cost | Available RAM | Persistence | HTTPS |
|---|---|---|---|---|---|
| Local / Docker Compose | Minimal | $0 | Unlimited | Local volume | No |
| Render.com | Low | $25–50 | 2 GB (Standard) | Disk ($1/GB) | Yes |
| Railway.com | Low | $10–30 | Dynamic | Volume included | Yes |
| OVH Public Cloud | Medium | $15–40 | 7–15 GB | Block Storage | Manual |
| OVH AI Deploy | Medium | Variable (usage) | 4–32 GB | Object Storage | Yes |

### CI/CD

The GitHub Actions workflow `.github/workflows/package.yml`:
1. Validates the code (lint, format, types, tests)
2. Builds Docker images for `amd64` and `arm64`
3. Produces self-contained deployment archives (without MLflow — pulled at `docker compose up`)
4. Publishes archives as build artifacts (branches/PRs) or GitHub releases (tags `v*`)
