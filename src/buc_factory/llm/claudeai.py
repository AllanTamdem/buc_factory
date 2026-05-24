"""LangChain-backed Anthropic LLM for the buc_factory agent."""

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage

from .base import BaseLLM


class AnthropicLLM(BaseLLM):
    """
    ChatAnthropic provider.

    Passes tool schemas as-is (already in Anthropic format) and wraps
    the system prompt with ephemeral cache_control to avoid re-charging
    the same tokens on every turn of the loop.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        max_tokens: int = 8000,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, max_tokens, api_key)

    def _get_client(self, model: str, max_tokens: int) -> ChatAnthropic:
        key = (model, max_tokens)
        if key not in self._client_cache:
            kwargs: dict = {"model": model, "max_tokens": max_tokens}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client_cache[key] = ChatAnthropic(**kwargs)
        return self._client_cache[key]

    def _prepare_tools(self, tools: list[dict]) -> list[dict]:
        return tools

    def _make_system_message(self, system: str) -> BaseMessage:
        return SystemMessage(
            content=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Single-turn completion via the raw Anthropic SDK (no autolog side-effects)."""
        client = anthropic.Anthropic(api_key=self._api_key or None)
        kwargs: dict = {
            "model": model or self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text
