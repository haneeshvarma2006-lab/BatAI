"""Agent loop port.

The runner is expressed as an async event *stream* rather than a coroutine
returning a string. That single choice buys three things the SaaS needs:

* the HTTP layer can stream tokens over SSE without the loop knowing about HTTP;
* a non-streaming endpoint is a fold over the same stream, so there is one
  implementation and not two that drift;
* cancellation is structural — if the client disconnects, the generator is
  closed and in-flight work unwinds through normal ``finally`` blocks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from bat.domain.conversation import Message, Session
from bat.domain.tenancy import TenantContext
from bat.ports.tools import ToolPolicy


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TOOL_BUDGET = "tool_budget"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything one agent turn needs. No hidden globals."""

    context: TenantContext
    session: Session
    user_input: str
    history: tuple[Message, ...] = ()
    policy: ToolPolicy = field(default_factory=ToolPolicy)
    max_steps: int = 6
    #: Rounds of tool calls before the loop asks for a plain answer.
    tool_rounds_per_turn: int = 1
    #: Wall-clock ceiling for the whole turn, tool calls included.
    deadline_s: float = 120.0
    stream_tokens: bool = True


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """An incremental slice of assistant text."""

    type: Literal["token"] = "token"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """The loop is about to invoke a tool."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """A tool returned an observation (or a denial)."""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    duration_ms: float = 0.0
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class FinalEvent:
    """Terminal event of a successful run."""

    content: str
    stop_reason: StopReason = StopReason.COMPLETED
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    type: Literal["final"] = "final"


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """Terminal event of a failed run. Message is caller-safe."""

    code: str
    message: str
    type: Literal["error"] = "error"


type AgentEvent = TokenEvent | ToolCallEvent | ToolResultEvent | FinalEvent | ErrorEvent

#: Events that terminate a stream. Exactly one is always emitted.
TERMINAL_EVENTS: tuple[type, ...] = (FinalEvent, ErrorEvent)


@runtime_checkable
class AgentRunner(Protocol):
    """Executes one turn, emitting events as it goes.

    Contract for implementations:

    * exactly one terminal event (``final`` or ``error``) is emitted, always,
      including on internal failure;
    * the stream is cancellable at any await point;
    * no tool runs without passing ``request.policy``;
    * ``deadline_s`` and ``max_steps`` are both enforced.
    """

    def run(self, request: RunRequest) -> AsyncIterator[AgentEvent]: ...


def is_terminal(event: AgentEvent) -> bool:
    return isinstance(event, TERMINAL_EVENTS)


async def collect(events: AsyncIterator[AgentEvent]) -> FinalEvent | ErrorEvent:
    """Fold an event stream into its terminal event.

    This is how the non-streaming endpoint reuses the streaming loop.
    """
    buffer: list[str] = []
    terminal: FinalEvent | ErrorEvent | None = None
    async for event in events:
        if isinstance(event, TokenEvent):
            buffer.append(event.text)
        elif isinstance(event, FinalEvent | ErrorEvent):
            terminal = event
            break
    if terminal is None:
        return ErrorEvent(
            code="incomplete_stream",
            message="agent produced no terminal event",
        )
    if isinstance(terminal, FinalEvent) and not terminal.content and buffer:
        # A runner that streamed tokens but left `final.content` empty still
        # gets a correct non-streaming response.
        return FinalEvent(
            content="".join(buffer),
            stop_reason=terminal.stop_reason,
            steps=terminal.steps,
            prompt_tokens=terminal.prompt_tokens,
            completion_tokens=terminal.completion_tokens,
        )
    return terminal


__all__ = [
    "AgentEvent",
    "AgentRunner",
    "ErrorEvent",
    "FinalEvent",
    "RunRequest",
    "StopReason",
    "TokenEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "collect",
    "is_terminal",
]
