FROM python:3.12-slim
WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Ensure writable runtime directories exist inside the image
RUN mkdir -p log mlflow_data/artifacts

EXPOSE 8000

# Default: CLI mode. Override with 'buc-factory-api' to start the HTTP server.
CMD ["uv", "run", "python", "-m", "buc_factory"]
