"""Agent turn endpoints: buffered and streaming.

Both endpoints drive the *same* :class:`AgentRunner` event stream. The buffered
one folds the stream with :func:`bat.ports.agent.collect`; the streaming one
forwards each event as SSE. There is no second code path to keep in sync.

Persistence ordering
--------------------
The user message is written before the run starts, so a crash mid-run still
leaves an accurate transcript. The assistant message is written only on a
terminal ``final`` event, so a failed or cancelled run does not persist a
half-formed reply that later turns would treat as real history.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from typing import Annotated

from fastapi import APIRouter, Path, Request
from fastapi.responses import StreamingResponse

from bat.api.deps import AgentInvoke, Config, RunLimiter, Runner, Store
from bat.api.schemas import MessageResponse, SendMessageRequest, TurnResponse
from bat.domain.conversation import Message, Role, Session
from bat.domain.errors import UpstreamError
from bat.domain.tenancy import TenantContext
from bat.ports.agent import (
    AgentEvent,
    AgentRunner,
    ErrorEvent,
    FinalEvent,
    RunRequest,
    TokenEvent,
    collect,
)
from bat.ports.session_store import SessionStore
from bat.ratelimit import ConcurrencyLimiter
from bat.ports.tools import ToolPolicy
from bat.settings import Settings

logger = logging.getLogger("bat.api.messages")

router = APIRouter(prefix="/v1/sessions", tags=["agent"])

SessionId = Annotated[str, Path(pattern=r"^sess_[0-9a-f]{32}$")]

#: Sent every 15s of silence so proxies and load balancers keep the SSE
#: connection open during a long tool call.
_KEEPALIVE_INTERVAL_S = 15.0


def _policy_for(settings: Settings) -> ToolPolicy:
    """Derive the effective tool policy from configuration.

    Built per request from settings rather than captured at import time, so a
    policy change cannot be defeated by a stale module-level default.
    """
    return ToolPolicy(
        allowed=frozenset(settings.agent.enabled_tools),
        max_authority=settings.agent.max_tool_authority,
        min_code_isolation=settings.agent.min_code_isolation,
        max_calls_per_run=settings.agent.max_tool_calls_per_run,
    )


async def _prepare(
    *,
    context: TenantContext,
    store: SessionStore,
    settings: Settings,
    session_id: str,
    body: SendMessageRequest,
) -> tuple[Session, Message, RunRequest]:
    """Load the session, persist the user turn and build the run request."""
    session = await store.get_session(
        tenant_id=context.tenant_id, session_id=session_id
    )
    history = await store.get_history(
        tenant_id=context.tenant_id,
        session_id=session_id,
        limit=settings.session.history_window,
    )
    user_message = await store.append_message(
        tenant_id=context.tenant_id,
        session_id=session_id,
        message=Message.create(
            session_id=session_id,
            tenant_id=context.tenant_id,
            role=Role.USER,
            content=body.content,
            metadata=body.metadata,
        ),
    )
    run_request = RunRequest(
        context=context,
        session=session,
        user_input=body.content,
        history=history,
        policy=_policy_for(settings),
        max_steps=settings.agent.max_steps,
        deadline_s=settings.agent.deadline_s,
        tool_rounds_per_turn=settings.agent.tool_rounds_per_turn,
    )
    return session, user_message, run_request


async def _persist_reply(
    *,
    context: TenantContext,
    store: SessionStore,
    session_id: str,
    content: str,
    stop_reason: str,
) -> Message:
    return await store.append_message(
        tenant_id=context.tenant_id,
        session_id=session_id,
        message=Message.create(
            session_id=session_id,
            tenant_id=context.tenant_id,
            role=Role.ASSISTANT,
            content=content,
            metadata={"stop_reason": stop_reason},
        ),
    )


@router.post(
    "/{session_id}/messages",
    response_model=TurnResponse,
    summary="Run one agent turn and wait for the reply",
)
async def send_message(
    session_id: SessionId,
    body: SendMessageRequest,
    context: AgentInvoke,
    store: Store,
    runner: Runner,
    settings: Config,
    limiter: RunLimiter,
) -> TurnResponse:
    session, user_message, run_request = await _prepare(
        context=context,
        store=store,
        settings=settings,
        session_id=session_id,
        body=body,
    )

    async with limiter.slot(context.tenant_id):
        terminal = await collect(
            runner.run(replace(run_request, stream_tokens=False))
        )

    if isinstance(terminal, ErrorEvent):
        logger.warning(
            "agent run failed",
            extra={"session_id": session_id, "error_code": terminal.code},
        )
        raise UpstreamError(terminal.message, details={"code": terminal.code})

    assistant_message = await _persist_reply(
        context=context,
        store=store,
        session_id=session_id,
        content=terminal.content,
        stop_reason=str(terminal.stop_reason),
    )
    return TurnResponse(
        session_id=session.id,
        user_message=MessageResponse.from_domain(user_message),
        assistant_message=MessageResponse.from_domain(assistant_message),
        stop_reason=str(terminal.stop_reason),
        steps=terminal.steps,
        usage={
            "prompt_tokens": terminal.prompt_tokens,
            "completion_tokens": terminal.completion_tokens,
        },
    )


@router.post(
    "/{session_id}/messages/stream",
    summary="Run one agent turn, streaming events over SSE",
    response_class=StreamingResponse,
)
async def stream_message(
    session_id: SessionId,
    body: SendMessageRequest,
    request: Request,
    context: AgentInvoke,
    store: Store,
    runner: Runner,
    settings: Config,
    limiter: RunLimiter,
) -> StreamingResponse:
    _, _, run_request = await _prepare(
        context=context,
        store=store,
        settings=settings,
        session_id=session_id,
        body=body,
    )

    generator = _sse_stream(
        request=request,
        context=context,
        store=store,
        runner=runner,
        limiter=limiter,
        session_id=session_id,
        run_request=run_request,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Stops nginx buffering the stream into uselessness.
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_stream(
    *,
    request: Request,
    context: TenantContext,
    store: SessionStore,
    runner: AgentRunner,
    limiter: ConcurrencyLimiter,
    session_id: str,
    run_request: RunRequest,
) -> AsyncIterator[str]:
    """Forward agent events as SSE frames, persisting the reply at the end."""
    buffered: list[str] = []
    try:
        async with limiter.slot(context.tenant_id):
            events = runner.run(run_request)
            async for event in _with_keepalive(events):
                if event is None:
                    yield ": keepalive\n\n"
                    continue

                if await request.is_disconnected():
                    logger.info(
                        "client disconnected mid-stream",
                        extra={"session_id": session_id},
                    )
                    break

                yield _frame(event)

                if isinstance(event, FinalEvent):
                    content = event.content or "".join(buffered)
                    if content:
                        await _persist_reply(
                            context=context,
                            store=store,
                            session_id=session_id,
                            content=content,
                            stop_reason=str(event.stop_reason),
                        )
                    break
                if isinstance(event, ErrorEvent):
                    break
                if isinstance(event, TokenEvent):
                    buffered.append(event.text)
    except asyncio.CancelledError:
        logger.info("stream cancelled", extra={"session_id": session_id})
        raise
    except Exception as exc:
        # A stream cannot change its status code once it has begun, so failures
        # after the first byte are reported as a terminal SSE error frame.
        logger.exception("stream failed", extra={"session_id": session_id})
        yield _frame(ErrorEvent(code="stream_failed", message=str(exc)))
    finally:
        yield "event: done\ndata: {}\n\n"


async def _with_keepalive(
    events: AsyncIterator[AgentEvent],
) -> AsyncIterator[AgentEvent | None]:
    """Yield events, emitting ``None`` whenever the source goes quiet.

    The in-flight pull is kept across keepalive ticks and shielded from
    `wait_for`'s cancellation: starting a second `__anext__` while the first is
    pending raises "anext(): asynchronous generator is already running", and an
    unshielded timeout would cancel the real work rather than just the wait.
    """
    iterator = events.__aiter__()
    pending: asyncio.Future[AgentEvent] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            try:
                event = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=_KEEPALIVE_INTERVAL_S
                )
            except TimeoutError:
                yield None
                continue
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            yield event
    finally:
        if pending is not None:
            pending.cancel()
        # Async generators expose aclose(); a plain AsyncIterator need not.
        closer = getattr(iterator, "aclose", None)
        if closer is not None:
            await closer()


def _frame(event: AgentEvent) -> str:
    """Render one SSE frame. The event type doubles as the SSE event name."""
    payload = asdict(event)
    name = payload.pop("type", "message")
    return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"

