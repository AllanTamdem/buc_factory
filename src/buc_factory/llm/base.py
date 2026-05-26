"""
Abstract base for LangChain-backed LLM wrappers used by the buc_factory agent.

Concrete subclasses implement four provider-specific hooks:

- ``_get_client``          — return a LangChain chat client for (model, max_tokens)
- ``_prepare_tools``       — convert Anthropic-format schemas to the provider's format
- ``_make_system_message`` — wrap a system string in the right BaseMessage form
- ``complete``             — single-turn completion via the raw provider SDK

Everything else — the agentic tool-use loop, client caching, default parameters —
lives here and is shared across providers.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

LOGGER = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Provider-agnostic agentic LLM wrapper.

    Subclass and implement the four abstract methods; call ``run_agent_loop``
    the same way regardless of provider.

    Example::

        llm = AnthropicLLM()   # or OpenAILLM()

        from langchain_core.messages import HumanMessage
        msgs, in_tok, out_tok = llm.run_agent_loop(
            messages=[HumanMessage("Write bootstrap.json.")],
            system="You are a BI assessment generator.",
            tools=tool_schemas,
            dispatch=tool_dispatch,
        )
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._default_max_tokens = max_tokens
        self._api_key = api_key
        self._client_cache: dict[tuple[str, int], Any] = {}

    # ── provider hooks ─────────────────────────────────────────────

    @abstractmethod
    def _get_client(self, model: str, max_tokens: int) -> Any:
        """Return a bound LangChain chat client for the given model / token cap."""

    @abstractmethod
    def _prepare_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format tool schemas to the provider's expected format."""

    @abstractmethod
    def _make_system_message(self, system: str) -> BaseMessage:
        """Wrap a system-prompt string in the provider-appropriate BaseMessage."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Single-turn text completion without tool use."""

    # ── shared infrastructure ──────────────────────────────────────

    def _llm_with_tools(
        self, max_tokens: int, tools: list[dict[str, Any]], model: str | None = None
    ) -> Any:
        return self._get_client(model or self._model, max_tokens).bind_tools(
            self._prepare_tools(tools)
        )

    def run_agent_loop(
        self,
        messages: list[BaseMessage],
        system: str,
        tools: list[dict[str, Any]],
        dispatch: dict[str, Callable[..., Any]],
        max_tokens: int | None = None,
        max_steps: int = 40,
        done_tool: str = "mark_subtask_complete",
        model: str | None = None,
    ) -> tuple[list[BaseMessage], int, int]:
        """
        Agentic loop: invoke the model, dispatch tool calls, repeat.

        Exits when the model returns with no tool calls, the done_tool fires,
        or max_steps is reached.

        Args:
            messages:   Conversation history (at least one HumanMessage).
            system:     System prompt string.
            tools:      Anthropic-format tool schemas.
            dispatch:   Mapping {tool_name: callable}.
            max_tokens: Per-call token cap; falls back to the instance default.
            max_steps:  Hard cap on model invocations.
            done_tool:  Tool name that signals task completion.

        Returns:
            Tuple of (messages, total_input_tokens, total_output_tokens).
        """
        _max_tokens = max_tokens or self._default_max_tokens
        llm = self._llm_with_tools(_max_tokens, tools, model=model)
        system_msg = self._make_system_message(system)
        msgs = list(messages)
        total_input = 0
        total_output = 0

        for _ in range(max_steps):
            response: AIMessage = llm.invoke([system_msg] + msgs)
            msgs.append(response)

            if response.usage_metadata:
                total_input += response.usage_metadata.get("input_tokens", 0)
                total_output += response.usage_metadata.get("output_tokens", 0)

            if not response.tool_calls:
                break

            def _call_tool(tc: Any) -> ToolMessage:
                fn = dispatch.get(tc["name"])
                try:
                    result = fn(**tc["args"]) if fn else f"ERROR: unknown tool {tc['name']}"
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                LOGGER.info(f"  → {tc['name']} → {str(result)[:120]}")
                return ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])

            n = len(response.tool_calls)
            if n > 1:
                with ThreadPoolExecutor(max_workers=n) as executor:
                    tool_messages = list(executor.map(_call_tool, response.tool_calls))
            else:
                tool_messages = [_call_tool(tc) for tc in response.tool_calls]

            done_signal = any(tc["name"] == done_tool for tc in response.tool_calls)

            msgs.extend(tool_messages)
            if done_signal:
                break

        return msgs, total_input, total_output
