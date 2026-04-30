if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Syncing dependencies..."
    uv sync --frozen
else
    echo "Creating virtual environment..."
    uv venv --python $(pyenv which python)
    uv add dotenv mlflow
    uv add --dev pytest pytest-cov ruff mypy
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# remove the .venv directory if it exists
# if [ -d "../.venv" ]; then
#     deactivate 2>/dev/null || true
#     echo "Removing existing virtual environment..."
#     rm -rf .venv
# fi