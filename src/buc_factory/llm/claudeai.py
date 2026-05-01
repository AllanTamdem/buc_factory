import logging
from collections.abc import Callable
from typing import Any

from anthropic import Anthropic

LOGGER = logging.getLogger(__name__)


class AnthropicLLM:
    """
    Thin wrapper around the Anthropic Messages API.

    Provides two surfaces:
    - ``generate()``       — single-turn completion with no tool loop.
    - ``run_agent_loop()`` — multi-step agentic loop with plain-callable tool dispatch.

    Example::

        llm = AnthropicLLM(api_key="sk-ant-...")

        # single completion
        response = llm.generate(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            system="You are a maths tutor.",
        )
        print(response.content[0].text)  # "4"

        # agentic loop
        def add(a: int, b: int) -> str:
            return str(a + b)

        tools = [{
            "name": "add",
            "description": "Add two integers.",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        }]
        messages = [{"role": "user", "content": "What is 3 + 5?"}]
        llm.run_agent_loop(messages, system="You are a maths tutor.", tools=tools,
                           dispatch={"add": add})
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-7",
        max_tokens: int = 8000,
    ):
        """
        Initialise the wrapper.

        Args:
            api_key:
                Anthropic API key. Must be a non-empty string starting with
                ``sk-ant-``. Passed directly to ``anthropic.Anthropic``.

                Example::

                    AnthropicLLM(api_key="sk-ant-api03-...")
                    # or from the environment:
                    AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY"))

            model:
                Claude model ID to use for every request made by this
                instance. Defaults to ``"claude-opus-4-7"``, the most capable
                generally available model.

                Note: ``claude-opus-4-7`` does not accept ``temperature``,
                ``top_p``, or ``top_k`` — pass those via ``generate(**kwargs)``
                only when targeting an older model.

                Example::

                    AnthropicLLM(api_key=key, model="claude-sonnet-4-6")

            max_tokens:
                Default per-response token cap used when ``generate()`` or
                ``run_agent_loop()`` are called without an explicit
                ``max_tokens`` override. Must be a positive integer.

                Anthropic requires this field on every request; the wrapper
                always sends it. Defaults to ``8000``.

                Example::

                    AnthropicLLM(api_key=key, max_tokens=16000)

        Raises:
            ValueError: If ``api_key`` is falsy (empty string or ``None``).
        """
        if not api_key:
            raise ValueError("Anthropic api_key must be provided")
        self._model = model
        self._max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key)

    def generate(
        self,
        messages: list[dict],
        system: str = "You are a helpful assistant.",
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        """
        Single API call with no tool loop.

        Sends one ``messages.create`` request and returns the raw Anthropic
        ``Message`` response object. Use this for straightforward completions
        where no tool execution is needed.

        Args:
            messages:
                Conversation turns in Anthropic wire format — a list of
                ``{"role": ..., "content": ...}`` dicts. Roles must
                alternate ``"user"`` / ``"assistant"``; the first turn must
                be ``"user"``.

                Example::

                    messages = [
                        {"role": "user", "content": "Summarise this article: ..."},
                    ]

                    # multi-turn
                    messages = [
                        {"role": "user",      "content": "My name is Alice."},
                        {"role": "assistant", "content": "Hello Alice!"},
                        {"role": "user",      "content": "What is my name?"},
                    ]

            system:
                System prompt passed separately from the conversation history,
                as required by the Anthropic API. Defaults to a generic
                helpful-assistant instruction.

                Example::

                    system = "You are a senior Python engineer. Reply only in code."

            max_tokens:
                Override the instance-level ``max_tokens`` for this call only.
                When ``None`` the instance default is used.

                Example::

                    # short classification — tight cap
                    llm.generate(messages, max_tokens=64)

                    # long report — generous cap
                    llm.generate(messages, max_tokens=16000)

            **kwargs:
                Any additional keyword argument accepted by
                ``anthropic.Anthropic().messages.create()``, forwarded
                verbatim. Useful for one-off overrides such as
                ``thinking``, ``tools``, or ``stop_sequences``.

                Note: do not pass ``temperature`` / ``top_p`` when targeting
                ``claude-opus-4-7`` — those parameters return a 400 error on
                that model.

                Example::

                    llm.generate(messages, thinking={"type": "adaptive"})
                    llm.generate(messages, stop_sequences=["STOP"])

        Returns:
            ``anthropic.types.Message`` — the raw API response. Text content
            is in ``response.content[0].text``; token usage is in
            ``response.usage``.
        """
        return self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
            **kwargs,
        )

    def run_agent_loop(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        dispatch: dict[str, Callable],
        max_tokens: int | None = None,
        max_steps: int = 40,
        done_tool: str = "mark_subtask_complete",
    ) -> list[dict]:
        """
        Agentic loop: call Claude, execute tools via dispatch, repeat
        until ``end_turn``, no tool calls, or ``done_tool`` fires.

        Each iteration appends the assistant turn and the tool-result
        user turn to ``messages`` in place, so the full conversation
        history accumulates there. The loop ends when:

        - Claude returns ``stop_reason == "end_turn"`` (no more tool calls).
        - Claude returns any ``stop_reason`` other than ``"tool_use"``.
        - The tool named ``done_tool`` is called (explicit completion signal).
        - ``max_steps`` iterations are exhausted.

        Args:
            messages:
                Mutable conversation list in Anthropic wire format. Must
                contain at least one ``"user"`` turn. The list is extended
                in place with every assistant response and tool-result round
                trip; callers can inspect the full history after the call.

                Example::

                    messages = [
                        {"role": "user", "content": "Write bootstrap.json with 5 dimensions."}
                    ]
                    llm.run_agent_loop(messages, system=system, tools=tools, dispatch=dispatch)
                    # messages now contains the full conversation

            system:
                System prompt passed to every ``messages.create`` call in the
                loop. Should describe the agent's role and constraints.

                Example::

                    system = "You are a BI assessment generator. Use the provided tools only."

            tools:
                List of Anthropic tool-schema dicts. Each dict must have at
                minimum ``"name"``, ``"description"``, and ``"input_schema"``
                keys. Only tools present here can be invoked by the model.

                Example::

                    tools = [{
                        "name": "write_file",
                        "description": "Write text to a file.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "path":    {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    }]

            dispatch:
                Mapping from tool name to a plain Python callable. When
                the model requests a tool call, the matching callable is
                invoked as ``fn(**block.input)``. If a tool name is missing
                from the mapping the result is an error string fed back to
                the model.

                Example::

                    dispatch = {
                        "write_file": lambda path, content: Path(path).write_text(content),
                        "read_file":  lambda path: Path(path).read_text(),
                    }

            max_tokens:
                Per-call token cap. Overrides the instance default for every
                request inside the loop. Useful when a specific subtask is
                known to produce unusually long output.

                Example::

                    # subtask writes a large markdown report
                    llm.run_agent_loop(messages, ..., max_tokens=16000)

            max_steps:
                Hard upper bound on the number of ``messages.create`` calls.
                Prevents infinite loops if the model never signals completion.
                Defaults to ``40``.

                Example::

                    llm.run_agent_loop(messages, ..., max_steps=10)

            done_tool:
                Name of the tool the model should call to signal that the
                subtask is complete. When this tool is invoked the loop exits
                immediately after processing all tool calls in that turn.
                Defaults to ``"mark_subtask_complete"``.

                Example::

                    llm.run_agent_loop(messages, ..., done_tool="finish")

        Returns:
            The ``messages`` list (same object passed in), now containing the
            full conversation including all assistant and tool-result turns.
        """
        _max_tokens = max_tokens or self._max_tokens

        for _ in range(max_steps):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break
            if response.stop_reason != "tool_use":
                break

            tool_results = []
            done_signal = False
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = dispatch.get(block.name)
                try:
                    result = fn(**block.input) if fn else f"ERROR: unknown tool {block.name}"
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                LOGGER.info(f"  → {block.name} → {str(result)[:120]}")
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
                )
                if block.name == done_tool:
                    done_signal = True
            messages.append({"role": "user", "content": tool_results})
            if done_signal:
                break

        return messages
