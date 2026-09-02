"""Model-server port.

Deliberately narrow: the agent loop needs a chat completion that may emit tool
calls, and nothing else. Ollama, a hosted API or a fake all satisfy this.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from bat.domain.conversation import ToolCall


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Wire-format message handed to the model server."""

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """JSON-schema description of a tool, as advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass(frozen=True, slots=True)
class Completion:
    """A single assistant turn returned by the model server."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """Async chat completion, with and without token streaming."""

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> Completion: ...

    def stream(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas. Tool-calling turns are not streamed."""
        ...
