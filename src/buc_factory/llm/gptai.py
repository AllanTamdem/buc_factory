"""LangChain-backed OpenAI LLM for the buc_factory agent."""

import openai
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base import BaseLLM


def _supports_temperature(model: str) -> bool:
    """GPT-4.x and GPT-3.x are the only OpenAI models that accept a temperature parameter."""
    return model.startswith("gpt-4") or model.startswith("gpt-3")


def _to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema format to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": dict(s["input_schema"]),
            },
        }
        for s in schemas
    ]


class OpenAILLM(BaseLLM):
    """
    ChatOpenAI provider.

    Converts Anthropic-format tool schemas to OpenAI function-calling format
    before binding. o-series reasoning models (o1, o3, o4-*) are instantiated
    without a temperature parameter.
    """

    def __init__(
        self,
        model: str = "gpt-5.3-chat-latest",
        max_tokens: int = 8000,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, max_tokens, api_key)

    def _get_client(self, model: str, max_tokens: int) -> ChatOpenAI:
        key = (model, max_tokens)
        if key not in self._client_cache:
            kwargs: dict = {"model": model, "max_tokens": max_tokens}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if _supports_temperature(model):
                kwargs["temperature"] = 0.7
            self._client_cache[key] = ChatOpenAI(**kwargs)
        return self._client_cache[key]

    def _prepare_tools(self, tools: list[dict]) -> list[dict]:
        return _to_openai_tools(tools)

    def _make_system_message(self, system: str) -> BaseMessage:
        return SystemMessage(content=system)

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Single-turn completion via the raw OpenAI SDK."""
        client = openai.OpenAI(api_key=self._api_key or None)
        response = client.chat.completions.create(
            model=model or self._model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
