FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install uv
COPY . .
RUN uv sync --frozen --no-dev
RUN uv build --wheel --out-dir /dist
# Replace editable install with the wheel so site-packages is self-contained
RUN uv pip install --no-deps /dist/*.whl


FROM python:3.12-slim
WORKDIR /app
# Copy the fully-built venv (deps + wheel-installed package, no source tree)
COPY --from=builder /app/.venv /app/.venv
COPY README.md .
COPY docs/ docs/
RUN mkdir -p log data/mlflow/artifacts
EXPOSE 8000
CMD [".venv/bin/buc-factory-api"]
