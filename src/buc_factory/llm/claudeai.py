"""
LangChain-backed Anthropic wrapper for the buc_factory agent.

Replaces the raw ``anthropic.Anthropic`` client with ``ChatAnthropic`` so that
the tool-use loop operates on standard LangChain ``BaseMessage`` objects rather
than the Anthropic wire-format dicts.
"""

import logging
from collections.abc import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

LOGGER = logging.getLogger(__name__)


class AnthropicLLM:
    """
    ChatAnthropic wrapper with an integrated agentic tool-use loop.

    Exposes a single primary surface:

    - ``run_agent_loop()`` — call the model, execute tools, repeat until
      end-turn, no tool calls, the done tool fires, or ``max_steps`` is hit.

    The loop returns the full ``list[BaseMessage]`` conversation so callers
    can inspect or extend it (e.g., LangGraph nodes that update graph state).

    Example::

        llm = AnthropicLLM()

        from langchain_core.messages import HumanMessage
        msgs = llm.run_agent_loop(
            messages=[HumanMessage("Write bootstrap.json with 5 dimensions.")],
            system="You are a BI assessment generator.",
            tools=tool_schemas,
            dispatch=tool_dispatch,
        )
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        max_tokens: int = 8000,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._default_max_tokens = max_tokens
        self._api_key = api_key

    def _llm_with_tools(self, max_tokens: int, tools: list[dict]) -> ChatAnthropic:
        kwargs: dict = {"model": self._model, "max_tokens": max_tokens}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return ChatAnthropic(**kwargs).bind_tools(tools)  # type: ignore[return-value]

    def run_agent_loop(
        self,
        messages: list[BaseMessage],
        system: str,
        tools: list[dict],
        dispatch: dict[str, Callable],
        max_tokens: int | None = None,
        max_steps: int = 40,
        done_tool: str = "mark_subtask_complete",
    ) -> tuple[list[BaseMessage], int, int]:
        """
        Agentic loop: invoke ChatAnthropic, dispatch tool calls, repeat.

        The loop exits when any of the following occur:

        - The model returns with no tool calls (natural end-turn).
        - The ``done_tool`` is called (explicit task-completion signal).
        - ``max_steps`` model calls have been made.

        Args:
            messages:   Conversation history (LangChain ``BaseMessage`` list).
                        Must begin with at least one ``HumanMessage``. Extended
                        in place; callers receive the same list back.
            system:     System prompt prepended to every model invocation.
            tools:      Anthropic-format tool schemas. ``ChatAnthropic.bind_tools``
                        accepts them directly alongside LangChain ``BaseTool``
                        objects.
            dispatch:   Mapping ``{tool_name: callable}``. Each callable is
                        invoked as ``fn(**args)`` when the model requests it.
                        Missing tools return an error string to the model.
            max_tokens: Per-call token cap; falls back to the instance default.
            max_steps:  Hard cap on model invocations. Defaults to 40.
            done_tool:  Tool name that signals task completion; exits the loop
                        after all tools in that turn are processed.

        Returns:
            Tuple of (messages, total_input_tokens, total_output_tokens) where
            token counts are summed across all model calls in the loop.
        """
        _max_tokens = max_tokens or self._default_max_tokens
        llm = self._llm_with_tools(_max_tokens, tools)
        msgs = list(messages)
        total_input = 0
        total_output = 0

        for _ in range(max_steps):
            response: AIMessage = llm.invoke([SystemMessage(content=system)] + msgs)
            msgs.append(response)

            if response.usage_metadata:
                total_input += response.usage_metadata.get("input_tokens", 0)
                total_output += response.usage_metadata.get("output_tokens", 0)

            if not response.tool_calls:
                break

            tool_messages: list[BaseMessage] = []
            done_signal = False
            for tc in response.tool_calls:
                fn = dispatch.get(tc["name"])
                try:
                    result = fn(**tc["args"]) if fn else f"ERROR: unknown tool {tc['name']}"
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                LOGGER.info(f"  → {tc['name']} → {str(result)[:120]}")
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
                )
                if tc["name"] == done_tool:
                    done_signal = True

            msgs.extend(tool_messages)
            if done_signal:
                break

        return msgs, total_input, total_output
