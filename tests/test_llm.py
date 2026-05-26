"""Tests for AnthropicLLM.complete() and OpenAILLM.complete() — system param and routing."""

from unittest.mock import MagicMock, patch

from buc_factory.llm.claudeai import AnthropicLLM
from buc_factory.llm.gptai import OpenAILLM

# ── AnthropicLLM.complete ─────────────────────────────────────────


def _anthropic_mock(text="response"):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(content=[MagicMock(text=text)])
    return mock_client


def test_anthropic_complete_returns_text():
    with patch("buc_factory.llm.claudeai.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = _anthropic_mock("hello")
        assert AnthropicLLM().complete("prompt") == "hello"


def test_anthropic_complete_without_system_omits_key():
    with patch("buc_factory.llm.claudeai.anthropic.Anthropic") as mock_cls:
        client = _anthropic_mock()
        mock_cls.return_value = client
        AnthropicLLM().complete("prompt")

    kwargs = client.messages.create.call_args[1]
    assert "system" not in kwargs


def test_anthropic_complete_with_system_passes_it():
    with patch("buc_factory.llm.claudeai.anthropic.Anthropic") as mock_cls:
        client = _anthropic_mock()
        mock_cls.return_value = client
        AnthropicLLM().complete("prompt", system="be concise")

    kwargs = client.messages.create.call_args[1]
    assert kwargs["system"] == "be concise"


def test_anthropic_complete_model_override():
    with patch("buc_factory.llm.claudeai.anthropic.Anthropic") as mock_cls:
        client = _anthropic_mock()
        mock_cls.return_value = client
        AnthropicLLM(model="claude-opus-4-7").complete("hi", model="claude-haiku-4-5-20251001")

    assert client.messages.create.call_args[1]["model"] == "claude-haiku-4-5-20251001"


def test_anthropic_complete_uses_instance_model_when_no_override():
    with patch("buc_factory.llm.claudeai.anthropic.Anthropic") as mock_cls:
        client = _anthropic_mock()
        mock_cls.return_value = client
        AnthropicLLM(model="claude-sonnet-4-6").complete("hi")

    assert client.messages.create.call_args[1]["model"] == "claude-sonnet-4-6"


# ── OpenAILLM.complete ────────────────────────────────────────────


def _openai_mock(text="response"):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))]
    )
    return mock_client


def test_openai_complete_returns_text():
    with patch("buc_factory.llm.gptai.openai.OpenAI") as mock_cls:
        mock_cls.return_value = _openai_mock("hi")
        assert OpenAILLM().complete("prompt") == "hi"


def test_openai_complete_without_system_sends_one_message():
    with patch("buc_factory.llm.gptai.openai.OpenAI") as mock_cls:
        client = _openai_mock()
        mock_cls.return_value = client
        OpenAILLM().complete("hello")

    messages = client.chat.completions.create.call_args[1]["messages"]
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "hello"}


def test_openai_complete_with_system_prepends_system_message():
    with patch("buc_factory.llm.gptai.openai.OpenAI") as mock_cls:
        client = _openai_mock()
        mock_cls.return_value = client
        OpenAILLM().complete("hello", system="be precise")

    messages = client.chat.completions.create.call_args[1]["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "be precise"}
    assert messages[1] == {"role": "user", "content": "hello"}


def test_openai_complete_model_override():
    with patch("buc_factory.llm.gptai.openai.OpenAI") as mock_cls:
        client = _openai_mock()
        mock_cls.return_value = client
        OpenAILLM(model="gpt-5.3-chat-latest").complete("hi", model="o4-mini")

    assert client.chat.completions.create.call_args[1]["model"] == "o4-mini"


def test_openai_complete_empty_content_returns_empty_string():
    with patch("buc_factory.llm.gptai.openai.OpenAI") as mock_cls:
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))]
        )
        mock_cls.return_value = client
        assert OpenAILLM().complete("hi") == ""
