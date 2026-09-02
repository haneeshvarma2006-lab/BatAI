"""Session persistence port.

Every method takes ``tenant_id`` as its first argument and implementations MUST
filter on it. A lookup for a session belonging to another tenant raises
:class:`~bat.domain.errors.NotFoundError`, never a permission error, so the API
does not leak the existence of other tenants' data.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

from bat.domain.conversation import Message, Page, Session


@runtime_checkable
class SessionStore(Protocol):
    """Async CRUD over sessions and their message history."""

    async def create_session(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        title: str | None = None,
        ttl: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session: ...

    async def get_session(self, *, tenant_id: str, session_id: str) -> Session:
        """Return the session, or raise ``NotFoundError`` if absent/expired."""
        ...

    async def list_sessions(
        self,
        *,
        tenant_id: str,
        principal_id: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> Page[Session]:
        """Newest-first keyset page. ``principal_id`` narrows to one user."""
        ...

    async def delete_session(self, *, tenant_id: str, session_id: str) -> None:
        """Idempotent: deleting an absent session raises ``NotFoundError``."""
        ...

    async def append_message(
        self, *, tenant_id: str, session_id: str, message: Message
    ) -> Message: ...

    async def get_history(
        self, *, tenant_id: str, session_id: str, limit: int = 50
    ) -> tuple[Message, ...]:
        """Oldest-first tail of the transcript, at most ``limit`` messages."""
        ...

    async def purge_expired(self) -> int:
        """Evict expired sessions; returns how many were removed."""
        ...
