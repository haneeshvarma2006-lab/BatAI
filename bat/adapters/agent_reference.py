"""Reference :class:`AgentRunner` used until the real loop lands.

This is deliberately not "the agent". It is a conforming implementation that
exercises the whole contract the API depends on -- token streaming, terminal
events, deadlines, cancellation -- so that the routing, session and streaming
layers are testable today and the real ReAct loop is a drop-in replacement
behind the same protocol.

It performs no tool calls and contacts no model server.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from bat.ports.agent import (
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    RunRequest,
    StopReason,
    TokenEvent,
)


class ReferenceAgentRunner:
    """Echoes a deterministic reply, one token at a time."""

    __slots__ = ("_prefix", "_token_delay_s")

    def __init__(self, *, prefix: str = "BAT", token_delay_s: float = 0.0) -> None:
        self._prefix = prefix
        self._token_delay_s = token_delay_s

    async def run(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        reply = self._compose(request)
        emitted: list[str] = []
        # A monotonic deadline rather than `asyncio.timeout`: this is an async
        # generator, so it is suspended at each yield and the timeout would keep
        # running in -- and fire inside -- whichever task is consuming it.
        deadline = asyncio.get_running_loop().time() + request.deadline_s
        stop_reason = StopReason.COMPLETED
        try:
            for token in _tokenize(reply):
                if asyncio.get_running_loop().time() >= deadline:
                    stop_reason = StopReason.MAX_STEPS
                    break
                if self._token_delay_s:
                    await asyncio.sleep(self._token_delay_s)
                emitted.append(token)
                if request.stream_tokens:
                    yield TokenEvent(text=token)
            yield FinalEvent(
                content="".join(emitted),
                stop_reason=stop_reason,
                steps=1,
                prompt_tokens=_estimate_tokens(request.user_input),
                completion_tokens=len(emitted),
            )
        except asyncio.CancelledError:
            # The client hung up. Let cancellation propagate so the task really
            # dies; emitting an event here would yield during unwinding.
            raise
        except Exception as exc:  # pragma: no cover - defensive
            # The contract promises exactly one terminal event, even on a bug.
            yield ErrorEvent(code="agent_error", message=str(exc))

    def _compose(self, request: RunRequest) -> str:
        history_note = (
            f" I have {len(request.history)} earlier message(s) in this session."
            if request.history
            else ""
        )
        return (
            f"{self._prefix} reference runner acknowledging: {request.user_input}."
            f"{history_note} No tools were invoked; the production agent loop"
            f" replaces this implementation behind the same interface."
        )


def _tokenize(text: str) -> list[str]:
    """Split into whitespace-preserving pseudo-tokens for streaming."""
    parts = text.split(" ")
    return [p if i == len(parts) - 1 else p + " " for i, p in enumerate(parts)]


def _estimate_tokens(text: str) -> int:
    """Crude but monotonic; real usage numbers come from the model server."""
    return max(1, len(text) // 4)


def describe() -> dict[str, Any]:
    return {"runner": "reference", "tools": [], "model": None}
