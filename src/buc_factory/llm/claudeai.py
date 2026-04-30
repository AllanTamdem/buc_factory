import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from anthropic import Anthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

LOGGER = logging.getLogger(__name__)


class AnthropicLLM(BaseChatModel):
    """
    A LangChain-compatible wrapper for Anthropic Claude chat models.

    This class provides a ChatModel abstraction similar to AzureOpenAILLM,
    adapted to Anthropic's Messages API. It supports:

    - Standard chat completions (system / user / assistant messages)
    - Tool calling via Claude's `tool_use` / `tool_result` protocol
    - Iterative tool execution with loop protection
    - Usage token accounting
    - LangChain Runnable and Agent compatibility

    The wrapper intentionally mirrors the structure and behavior of the
    AzureOpenAILLM implementation to minimize migration effort.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-7",
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic Claude LLM wrapper.

        Args:
            api_key (str):
                Anthropic API key (required). Must start with `sk-ant-`.

            model (str):
                Claude model name.
                Default: "claude-opus-4-7" (best for most business use cases and deeper tool use).

            kwargs:
                Additional generation parameters:

                temperature (float):
                    Controls randomness in output.
                    Range: 0–1
                    Default: 0.1

                max_tokens (int):
                    Maximum tokens to generate per response.
                    **Required by Anthropic**.
                    Default: 4096

                stop (list[str] | None):
                    Stop sequences that will halt generation when encountered.

                tools (list[dict] | None):
                    List of tool definitions for Claude to use. Each tool should be a dict with:
                    - `type`: "tool_search_tool_bm25" | "tool_search_tool_regex" | ...

                Any other keyword arguments are accepted for compatibility
                with LangChain's BaseChatModel interface.

        Raises:
            ValueError:
                If `api_key` is not provided.
        """
        super().__init__(**kwargs)

        if not api_key:
            raise ValueError("Anthropic api_key must be provided")

        self._model = model
        self._temperature = kwargs.get("temperature", 0.1)
        self._max_tokens = kwargs.get("max_tokens", 4096)
        self._stop = kwargs.get("stop")
        self._tools = kwargs.get("tools")  # Predifined Claude tools
        self._client = Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # LangChain required properties
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        """Return the type identifier for this LLM."""
        return "AnthropicChatModel"

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Default generation parameters passed to the Anthropic API."""
        return {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stop_sequences": self._stop,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Parameters that uniquely identify this LLM instance."""
        return {
            "model": self._model,
        }

    # ------------------------------------------------------------------
    # Request formatting
    # ------------------------------------------------------------------

    def _format_request(self, prompt, **kwargs):
        """
        Format LangChain inputs into Anthropic Messages API format.

        Anthropic requires:
        - `system` message passed separately
        - `messages` as a list of role/content dicts

        Args:
            prompt:
                Either a raw string or a list of LangChain BaseMessage objects

        Returns:
            tuple[str, list[dict]]:
                system_message, formatted_messages
        """
        system_message = kwargs.get("system_message", "You are a helpful assistant.")

        messages = []

        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, BaseMessage):
                    role = "assistant" if msg.type == "ai" else "user"
                    messages.append(
                        {
                            "role": role,
                            "content": msg.content,
                        }
                    )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

        return system_message, messages

    # ------------------------------------------------------------------
    # Response → LangChain conversion
    # ------------------------------------------------------------------

    def _create_chat_result(
        self,
        response,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """
        Convert an Anthropic API response into a LangChain ChatResult.

        Handles:
        - Text blocks
        - Tool-use blocks
        - Token usage metadata
        """
        text_output = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                text_output += block.text
            elif block.type == "tool_use":
                tool_uses.append(block)

        message = AIMessage(content=text_output)

        if response.usage:
            message.usage_metadata = UsageMetadata(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

        generation = ChatGeneration(
            message=message,
            generation_info=generation_info or {},
        )

        return ChatResult(generations=[generation])

    # ------------------------------------------------------------------
    # Core generation loop with tool execution
    # ------------------------------------------------------------------

    def _generate(self, prompt, **kwargs):
        """
        Generate a response from the Anthropic model.

        Supports iterative tool calling. If the model emits one or more
        `tool_use` blocks, those tools are executed and the results are
        fed back into the conversation via `tool_result` messages.

        Args:
            prompt:
                Input prompt or list of messages.

            max_loops (int):
                Maximum number of tool-call iterations before aborting.
                Default: 5

        Returns:
            ChatResult
        """
        system, messages = self._format_request(prompt, **kwargs)

        tools = (
            kwargs.get("tools") or self._tools
        )  # Allow tools to be passed per-call or use predefined ones
        tools_registry = kwargs.get("tools_registry", {})
        max_loops = kwargs.get("max_loops", 5)

        for _ in range(max_loops):
            response = self._client.messages.create(
                model=self._model,
                system=system,
                messages=messages,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
                stop_sequences=kwargs.get("stop", self._stop),
                tools=tools,
            )

            tool_calls = [block for block in response.content if block.type == "tool_use"]

            # No tool calls → final answer
            if not tool_calls:
                return self._create_chat_result(
                    response=response,
                    generation_info=self._default_params,
                )

            # Add assistant message containing tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )
            print(
                f"Model requested {len(tool_calls)} "
                f"tool call(s): {[call.name for call in tool_calls]}"
            )
            # Execute each tool
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.input
                impl = tools_registry.get(tool_name)

                if impl is None:
                    tool_result = f"ERROR: tool '{tool_name}' not found."
                else:
                    try:
                        tool_result = impl.invoke(tool_args)
                    except Exception as e:
                        tool_result = f"ERROR executing tool '{tool_name}': {e}"

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": tool_result,
                            }
                        ],
                    }
                )
                print(f"Executed tool '{tool_name}' with args {tool_args}, result: {tool_result}")

        LOGGER.warning("Max tool-calling loops reached, returning last response.")
        return self._create_chat_result(
            response=response,
            generation_info=self._default_params,
        )

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[dict | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """
        Bind tool definitions to this chat model.

        Tool schemas are converted using OpenAI-compatible definitions,
        which are fully supported by Claude.

        Args:
            tools:
                List of tool definitions or BaseTool instances.

        Returns:
            Runnable suitable for LangChain agents and chains.
        """
        tools_registry = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}
        anthropic_tools = [
            {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.args_schema.model_json_schema(),
            }
            for tool in tools
            if isinstance(tool, BaseTool)
        ]
        kwargs["tools_registry"] = tools_registry

        return super().bind(tools=anthropic_tools, **kwargs)
