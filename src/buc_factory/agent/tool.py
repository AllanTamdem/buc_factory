import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────


def make_tools(output_dir: Path) -> tuple[list[dict], dict[str, Callable]]:

    def write_file(path: str, content: str) -> str:
        full = output_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def read_file(path: str) -> str:
        full = output_dir / path
        if not full.exists():
            return f"ERROR: {path} does not exist"
        return full.read_text(encoding="utf-8")

    def list_files(directory: str = ".") -> str:
        full = output_dir / directory
        if not full.exists():
            return f"ERROR: {directory} does not exist"
        return "\n".join(str(p.relative_to(output_dir)) for p in full.rglob("*") if p.is_file())

    def run_python(script_path: str) -> str:
        full = output_dir / script_path
        if not full.exists():
            return f"ERROR: {full} does not exist"
        try:
            result = subprocess.run(
                [sys.executable, str(full)],
                cwd=full.parent,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return (
                f"exit={result.returncode}\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )
        except subprocess.TimeoutExpired:
            return "ERROR: script timed out after 60s"

    def validate_csv_integrity(spec_json: str) -> str:
        try:
            import pandas as pd

            spec = json.loads(spec_json)
            fact = pd.read_csv(output_dir / spec["facts"])
            dim = pd.read_csv(output_dir / spec["dim"])
            orphans = set(fact[spec["fact_fk"]]) - set(dim[spec["dim_pk"]])
            if orphans:
                return f"FAIL: {len(orphans)} orphan FK values, sample: {list(orphans)[:5]}"
            return f"OK: {len(fact)} fact rows, all FKs resolve in {len(dim)} dim rows"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def validate_json(path: str) -> str:
        try:
            json.loads((output_dir / path).read_text())
            return f"OK: valid JSON in {path}"
        except Exception as e:
            return f"FAIL: {type(e).__name__}: {e}"

    def mark_subtask_complete(summary: str) -> str:
        return f"acknowledged: {summary}"

    def _tool(
        name: str, description: str, properties: dict, required: list[str] | None = None
    ) -> dict:
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return {"name": name, "description": description, "input_schema": schema}

    schemas = [
        _tool(
            "write_file",
            "Write a text file. Overwrites if exists.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        _tool(
            "read_file",
            "Read a previously written file.",
            {"path": {"type": "string"}},
            ["path"],
        ),
        _tool(
            "list_files",
            "List files under a directory.",
            {"directory": {"type": "string"}},
        ),
        _tool(
            "run_python",
            "Execute a Python script. Returns stdout/stderr/exit.",
            {"script_path": {"type": "string"}},
            ["script_path"],
        ),
        _tool(
            "validate_csv_integrity",
            "Check FK integrity. Pass JSON: {facts, fact_fk, dim, dim_pk}.",
            {"spec_json": {"type": "string"}},
            ["spec_json"],
        ),
        _tool(
            "validate_json",
            "Verify a file contains valid JSON.",
            {"path": {"type": "string"}},
            ["path"],
        ),
        _tool(
            "mark_subtask_complete",
            "Signal sub-task completion with summary.",
            {"summary": {"type": "string"}},
            ["summary"],
        ),
    ]

    dispatch = {
        "write_file": write_file,
        "read_file": read_file,
        "list_files": list_files,
        "run_python": run_python,
        "validate_csv_integrity": validate_csv_integrity,
        "validate_json": validate_json,
        "mark_subtask_complete": mark_subtask_complete,
    }
    return schemas, dispatch
