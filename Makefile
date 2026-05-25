.DEFAULT_GOAL := help

# ── Environment ────────────────────────────────────────────────────────────────
.PHONY: install install-dev venv

venv:
	uv venv --python $(shell pyenv which python)
	@test -f uv.lock || uv lock
install:
	uv sync --frozen --no-dev

install-dev:
	uv sync --frozen
# 	uv pip install -e .

# ── Code quality ───────────────────────────────────────────────────────────────
.PHONY: lint format format-check typecheck check

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/

check: lint format-check typecheck

# ── Tests ──────────────────────────────────────────────────────────────────────
.PHONY: test test-cov

test:
	uv run pytest tests/ -q --tb=short

test-cov:
	uv run pytest tests/ --cov=src/buc_factory --cov-report=term-missing -q

# ── Run locally ────────────────────────────────────────────────────────────────
.PHONY: api ui mlflow

ui:
	@buc-factory-ui

api:
	@caffeinate -i buc-factory-api

mlflow:
	uv run mlflow server \
		--host 127.0.0.1 \
		--port 5000 \
		--backend-store-uri sqlite:///data/mlflow/mlflow.db \
		--default-artifact-root data/mlflow/artifacts

# ── Docker ─────────────────────────────────────────────────────────────────────
.PHONY: build up down logs

build:
	docker compose build

up:
	docker compose up mlflow api ui -d

down:
	docker compose down

logs:
	docker compose logs -f

# Run the CLI one-shot container (pass ARGS to override config/output-dir)
# Usage: make run ARGS="--config conf/industry_spec/saas_france.yml --output-dir data/run_001"
.PHONY: run
run:
	docker compose --profile cli run --rm buc_factory $(ARGS)

# ── Housekeeping ───────────────────────────────────────────────────────────────
.PHONY: clean clean-docker

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/ .pytest_cache/; \
	echo "cleaned"

clean-docker:
	docker compose down --rmi all --volumes

# ── Help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  buc-factory — available targets"
	@echo ""
	@echo "  Environment"
	@echo "    venv          Create .venv with uv"
	@echo "    install       Install production dependencies"
	@echo "    install-dev   Install all dependencies (incl. dev)"
	@echo ""
	@echo "  Code quality"
	@echo "    lint          ruff check"
	@echo "    format        ruff format (in-place)"
	@echo "    format-check  ruff format --check"
	@echo "    typecheck     mypy"
	@echo "    check         lint + format-check + typecheck"
	@echo ""
	@echo "  Tests"
	@echo "    test          pytest (quick)"
	@echo "    test-cov      pytest with coverage report"
	@echo ""
	@echo "  Run locally"
	@echo "    api           Start FastAPI server"
	@echo "    ui            Start Streamlit UI"
	@echo "    mlflow        Start local MLflow tracking server"
	@echo ""
	@echo "  Docker"
	@echo "    build         docker compose build"
	@echo "    up            Start mlflow + api + ui in background"
	@echo "    down          Stop containers"
	@echo "    logs          Tail container logs"
	@echo "    run ARGS=...  Run CLI container (pass --config / --output-dir)"
	@echo ""
	@echo "  Housekeeping"
	@echo "    clean         Remove build/cache artifacts"
	@echo "    clean-docker  Remove containers, images, and volumes"
	@echo ""
