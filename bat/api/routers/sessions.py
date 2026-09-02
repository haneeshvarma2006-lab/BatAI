"""Session lifecycle endpoints.

Every handler passes ``context.tenant_id`` to the store. The store raises
``NotFoundError`` for anything outside the tenant, so a caller probing another
tenant's session ids sees the same 404 as for ids that never existed.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status

from bat.api.deps import Config, SessionsRead, SessionsWrite, Store
from bat.api.schemas import (
    CreateSessionRequest,
    MessageListResponse,
    MessageResponse,
    SessionListResponse,
    SessionResponse,
)

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

SessionId = Annotated[str, Path(min_length=5, max_length=64, pattern=r"^sess_[0-9a-f]{32}$")]


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Open a session",
)
async def create_session(
    body: CreateSessionRequest,
    context: SessionsWrite,
    store: Store,
    settings: Config,
    response: Response,
) -> SessionResponse:
    session = await store.create_session(
        tenant_id=context.tenant_id,
        principal_id=context.principal_id,
        title=body.title,
        ttl=timedelta(seconds=settings.session.ttl_seconds),
        metadata=body.metadata,
    )
    response.headers["Location"] = f"/v1/sessions/{session.id}"
    return SessionResponse.from_domain(session)


@router.get("", response_model=SessionListResponse, summary="List sessions")
async def list_sessions(
    context: SessionsRead,
    store: Store,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
    mine_only: Annotated[bool, Query()] = True,
) -> SessionListResponse:
    """Defaults to the caller's own sessions.

    Listing a whole tenant is opt-in rather than the default, so a shared
    tenant key does not casually expose one user's threads to another.
    """
    page = await store.list_sessions(
        tenant_id=context.tenant_id,
        principal_id=context.principal_id if mine_only else None,
        limit=limit,
        cursor=cursor,
    )
    return SessionListResponse(
        items=tuple(SessionResponse.from_domain(s) for s in page.items),
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )


@router.get("/{session_id}", response_model=SessionResponse, summary="Fetch a session")
async def get_session(
    session_id: SessionId, context: SessionsRead, store: Store
) -> SessionResponse:
    session = await store.get_session(
        tenant_id=context.tenant_id, session_id=session_id
    )
    return SessionResponse.from_domain(session)


@router.get(
    "/{session_id}/messages",
    response_model=MessageListResponse,
    summary="Read a transcript",
)
async def get_messages(
    session_id: SessionId,
    context: SessionsRead,
    store: Store,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> MessageListResponse:
    history = await store.get_history(
        tenant_id=context.tenant_id, session_id=session_id, limit=limit
    )
    return MessageListResponse(
        items=tuple(MessageResponse.from_domain(m) for m in history)
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session and its transcript",
)
async def delete_session(
    session_id: SessionId, context: SessionsWrite, store: Store
) -> Response:
    await store.delete_session(tenant_id=context.tenant_id, session_id=session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
