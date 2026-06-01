"""Tests for score_submission() — verifies OpenAI/Anthropic delegation and fallback."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

from buc_factory.scorer import score_submission
from buc_factory.scorer.models import ScoringResult

_VALID_JSON = """{
  "tool_type": "python_da",
  "total_score": 7,
  "max_total_score": 10,
  "dimensions": [
    {"name": "analysis", "category": "methodology", "score": 7, "max_score": 10, "comment": "Solid"}
  ],
  "overall_verdict": "Competent analyst",
  "language": "en"
}"""


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    brief = tmp_path / "brief" / "candidate_brief.md"
    brief.parent.mkdir()
    brief.write_text("# Brief\n" + "x" * 200)

    nb = tmp_path / "starter" / "notebook.ipynb"
    nb.parent.mkdir()
    nb.write_text('{"cells": [{"cell_type": "code", "source": "import pandas"}]}')

    return tmp_path


def _auth_error() -> openai.AuthenticationError:
    resp = MagicMock()
    resp.status_code = 401
    return openai.AuthenticationError(message="invalid key", response=resp, body={})


# ── happy path ────────────────────────────────────────────────────


def test_score_returns_scoring_result(run_dir: Path) -> None:
    with patch("buc_factory.scorer._call_llm", return_value=_VALID_JSON):
        result = score_submission(run_dir, "solution text", "IPYNB")
    assert isinstance(result, ScoringResult)
    assert result.total_score == 7
    assert result.tool_type == "python_da"


def test_score_passes_solution_in_prompt(run_dir: Path) -> None:
    captured: dict[str, str] = {}

    def _fake(user: str, system: str) -> str:
        captured["user"] = user
        captured["system"] = system
        return _VALID_JSON

    with patch("buc_factory.scorer._call_llm", side_effect=_fake):
        score_submission(run_dir, "my_unique_solution_text", "IPYNB")

    assert "my_unique_solution_text" in captured["user"]
    assert len(captured["system"]) > 0


# ── OpenAI → Anthropic fallback ───────────────────────────────────


def test_score_falls_back_on_openai_auth_error(run_dir: Path) -> None:
    with (
        patch("buc_factory.scorer._openai_call", side_effect=_auth_error()),
        patch(
            "buc_factory.scorer._anthropic_fallback_call", return_value=_VALID_JSON
        ) as mock_fallback,
    ):
        result = score_submission(run_dir, "solution", "IPYNB")

    assert isinstance(result, ScoringResult)
    mock_fallback.assert_called_once()


def test_score_returns_none_on_openai_generic_error(run_dir: Path) -> None:
    with (
        patch("buc_factory.scorer._openai_call", side_effect=Exception("connection reset")),
        patch("buc_factory.scorer._anthropic_fallback_call") as mock_fallback,
    ):
        result = score_submission(run_dir, "solution", "IPYNB")

    assert result is None
    mock_fallback.assert_not_called()


def test_score_returns_none_on_full_failure(run_dir: Path) -> None:
    with (
        patch("buc_factory.scorer._openai_call", side_effect=_auth_error()),
        patch("buc_factory.scorer._anthropic_fallback_call", side_effect=Exception("down")),
    ):
        result = score_submission(run_dir, "solution", "IPYNB")

    assert result is None


# ── missing inputs ────────────────────────────────────────────────


def test_score_returns_none_when_no_brief(tmp_path: Path) -> None:
    (tmp_path / "starter").mkdir()
    result = score_submission(tmp_path, "solution", "IPYNB")
    assert result is None


def test_score_returns_none_when_no_work(tmp_path: Path) -> None:
    brief = tmp_path / "brief" / "candidate_brief.md"
    brief.parent.mkdir()
    brief.write_text("# Brief")
    (tmp_path / "starter").mkdir()

    result = score_submission(tmp_path, "solution", "IPYNB")
    assert result is None
