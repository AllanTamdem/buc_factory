"""Tests for score_submission() — verifies AnthropicLLM/OpenAILLM delegation and fallback."""

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from buc_factory.agent.scorer import score_submission


@pytest.fixture
def run_dir(tmp_path):
    brief = tmp_path / "brief" / "candidate_brief.md"
    brief.parent.mkdir()
    brief.write_text("# Brief\n" + "x" * 200)

    nb = tmp_path / "starter" / "notebook.ipynb"
    nb.parent.mkdir()
    nb.write_text('{"cells": []}')

    return tmp_path


def _mock_llm(return_text="## Score\n| Criterion | Score |\n|---|---|\n| Analysis | 8/10 |"):
    m = MagicMock()
    m.return_value.complete.return_value = return_text
    return m


# ── happy path ────────────────────────────────────────────────────


def test_score_simulation_returns_string(run_dir):
    with patch("buc_factory.agent.scorer.AnthropicLLM") as mock_cls:
        mock_cls.return_value.complete.return_value = "## Score"
        result = score_submission(run_dir, "solution text", "IPYNB")
    assert isinstance(result, str)
    assert "Score" in result


def test_score_submission_passes_system_and_user(run_dir):
    with patch("buc_factory.agent.scorer.AnthropicLLM") as mock_cls:
        instance = mock_cls.return_value
        instance.complete.return_value = "ok"
        score_submission(run_dir, "solution", "IPYNB")

    call_kwargs = instance.complete.call_args[1]
    assert "system" in call_kwargs
    assert len(call_kwargs["system"]) > 0
    positional_prompt = instance.complete.call_args[0][0]
    assert "solution" in positional_prompt


# ── fallback to OpenAI ────────────────────────────────────────────


def test_score_submission_falls_back_on_auth_error(run_dir):
    with (
        patch("buc_factory.agent.scorer.AnthropicLLM") as mock_anthropic,
        patch("buc_factory.agent.scorer.OpenAILLM") as mock_openai,
    ):
        mock_anthropic.return_value.complete.side_effect = anthropic.AuthenticationError(
            message="invalid key", response=MagicMock(), body={}
        )
        mock_openai.return_value.complete.return_value = "## OAI Score"

        result = score_submission(run_dir, "solution", "IPYNB")

    assert result == "## OAI Score"
    mock_openai.return_value.complete.assert_called_once()


def test_score_submission_falls_back_on_credit_exhausted(run_dir):
    with (
        patch("buc_factory.agent.scorer.AnthropicLLM") as mock_anthropic,
        patch("buc_factory.agent.scorer.OpenAILLM") as mock_openai,
    ):
        mock_anthropic.return_value.complete.side_effect = anthropic.BadRequestError(
            message="Your credit balance is too low to access the Anthropic API.",
            response=MagicMock(),
            body={},
        )
        mock_openai.return_value.complete.return_value = "## OAI Score"

        result = score_submission(run_dir, "solution", "IPYNB")

    assert result == "## OAI Score"
    mock_openai.return_value.complete.assert_called_once()


def test_score_submission_returns_none_on_unrelated_bad_request(run_dir):
    with (
        patch("buc_factory.agent.scorer.AnthropicLLM") as mock_anthropic,
        patch("buc_factory.agent.scorer.OpenAILLM") as mock_openai,
    ):
        mock_anthropic.return_value.complete.side_effect = anthropic.BadRequestError(
            message="max_tokens exceeds model limit",
            response=MagicMock(),
            body={},
        )

        result = score_submission(run_dir, "solution", "IPYNB")

    assert result is None
    mock_openai.return_value.complete.assert_not_called()


def test_score_submission_returns_none_on_full_failure(run_dir):
    with (
        patch("buc_factory.agent.scorer.AnthropicLLM") as mock_anthropic,
        patch("buc_factory.agent.scorer.OpenAILLM") as mock_openai,
    ):
        mock_anthropic.return_value.complete.side_effect = anthropic.AuthenticationError(
            message="invalid key", response=MagicMock(), body={}
        )
        mock_openai.return_value.complete.side_effect = Exception("openai down")

        result = score_submission(run_dir, "solution", "IPYNB")

    assert result is None


# ── missing inputs ────────────────────────────────────────────────


def test_score_submission_returns_none_when_no_brief(tmp_path):
    (tmp_path / "starter").mkdir()
    result = score_submission(tmp_path, "solution", "IPYNB")
    assert result is None


def test_score_submission_returns_none_when_no_work(tmp_path):
    brief = tmp_path / "brief" / "candidate_brief.md"
    brief.parent.mkdir()
    brief.write_text("# Brief")
    (tmp_path / "starter").mkdir()

    result = score_submission(tmp_path, "solution", "IPYNB")
    assert result is None
