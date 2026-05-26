# buc-factory — Deployment Guide

This archive is self-contained. It bundles the pre-built Docker image and everything you need to run the stack on any machine with Docker installed.

## Packages

Two archives are published — download the one matching your machine:

| File | For |
|------|-----|
| `buc-factory-stack-macos-arm64.tar.gz` | Apple Silicon Mac (M1/M2/M3/M4) |
| `buc-factory-stack-amd64.tar.gz` | Windows, Linux, Intel Mac |

## Contents

Both archives have the same structure:

```
├── buc-factory-image.tar   ← pre-built application image
├── mlflow-image.tar        ← MLflow tracking server image
├── compose.yml             ← Docker Compose stack definition
├── .env                    ← environment variable template (fill in before starting)
└── README.md               ← this file
```

No internet access required after extraction.

## System requirements

All LLM inference is API-based (Anthropic / OpenAI) — no GPU required.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU      | 2 cores | 4 cores     |
| RAM      | 4 GB    | 8 GB        |
| Disk     | 6 GB    | 12 GB       |

The recommended values reflect Docker Desktop settings that run this stack comfortably. The disk floor covers the two extracted images (~2.5 GB uncompressed) plus MLflow artifacts and run outputs.

## Prerequisites

- Docker ≥ 24 with the Compose plugin (`docker compose version`)

## Quick start

**1. Extract the archive**

```bash
tar -xzf buc-factory-stack.tar.gz
```

**2. Load the images**

```bash
docker load -i buc-factory-image.tar
docker load -i mlflow-image.tar
```

**3. Fill in your credentials**

Open `.env` and set at minimum:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```

**4. Create local directories** (Docker mounts these)

```bash
mkdir -p log data/mlflow/artifacts conf
```

**5. Start the stack**

```bash
# Start MLflow + API + UI in the background
docker compose up mlflow api ui -d

# Follow logs
docker compose logs -f
```

| Service  | URL                    |
|----------|------------------------|
| API      | http://localhost:8000  |
| UI       | http://localhost:8501  |
| MLflow   | http://localhost:5001  |

## Run a one-shot assessment (CLI mode)

```bash
docker compose --profile cli run --rm buc_factory \
  --config conf/my_spec.yml \
  --output-dir data/run_001
```

## Tear down

```bash
docker compose down          # stop containers, keep data
docker compose down -v       # stop + remove volumes
docker rmi buc-factory:latest
```
