"""In-process session store.

Correct and fully tenant-isolated, but per-process: sessions vanish on restart
and are invisible to other replicas. It exists so the API layer, the tests and
local development have a real implementation to run against; production selects
a Redis or Postgres adapter behind the same :class:`SessionStore` protocol, and
:meth:`bat.settings.Settings._harden` refuses to boot production with this one.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from datetime import UTC, datetime, timedelta
from typing import Any

from bat.domain.conversation import Message, Page, Session
from bat.domain.errors import ConflictError, NotFoundError, ValidationError


def _encode_cursor(session: Session) -> str:
    """Keyset cursor over (created_at, id), which is stable under inserts.

    Offset pagination would skip or duplicate rows whenever a session is created
    mid-listing; keyset pagination does not.
    """
    raw = f"{session.created_at.isoformat()}|{session.id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        created_raw, sep, session_id = decoded.partition("|")
        if not sep:
            raise ValueError("missing separator")
        return datetime.fromisoformat(created_raw), session_id
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValidationError("malformed cursor", details={"cursor": cursor}) from exc


class InMemorySessionStore:
    """Tenant-partitioned session and transcript storage."""

    __slots__ = (
        "_history_cap",
        "_lock",
        "_max_per_principal",
        "_messages",
        "_sessions",
        "_ttl",
    )

    def __init__(
        self,
        *,
        ttl: timedelta | None = None,
        max_sessions_per_principal: int = 200,
        history_cap: int = 500,
    ) -> None:
        self._ttl = ttl
        self._max_per_principal = max_sessions_per_principal
        self._history_cap = history_cap
        # Partitioned by tenant, so forgetting to filter is a missing key rather
        # than a cross-tenant read.
        self._sessions: dict[str, dict[str, Session]] = {}
        self._messages: dict[str, dict[str, list[Message]]] = {}
        self._lock = asyncio.Lock()

    # -- internals ---------------------------------------------------------

    def _tenant_sessions(self, tenant_id: str) -> dict[str, Session]:
        return self._sessions.setdefault(tenant_id, {})

    def _tenant_messages(self, tenant_id: str) -> dict[str, list[Message]]:
        return self._messages.setdefault(tenant_id, {})

    def _load(self, tenant_id: str, session_id: str) -> Session:
        """Fetch within the tenant partition, treating expiry as absence."""
        session = self._tenant_sessions(tenant_id).get(session_id)
        if session is None or session.is_expired():
            raise NotFoundError("session not found", details={"session_id": session_id})
        return session

    # -- SessionStore protocol --------------------------------------------

    async def create_session(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        title: str | None = None,
        ttl: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        async with self._lock:
            owned = [
                s
                for s in self._tenant_sessions(tenant_id).values()
                if s.principal_id == principal_id and not s.is_expired()
            ]
            if len(owned) >= self._max_per_principal:
                raise ConflictError(
                    "session limit reached for this user",
                    details={"limit": self._max_per_principal},
                )
            session = Session.create(
                tenant_id=tenant_id,
                principal_id=principal_id,
                title=title,
                ttl=ttl or self._ttl,
                metadata=metadata,
            )
            self._tenant_sessions(tenant_id)[session.id] = session
            self._tenant_messages(tenant_id)[session.id] = []
            return session

    async def get_session(self, *, tenant_id: str, session_id: str) -> Session:
        async with self._lock:
            return self._load(tenant_id, session_id)

    async def list_sessions(
        self,
        *,
        tenant_id: str,
        principal_id: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> Page[Session]:
        if not 1 <= limit <= 100:
            raise ValidationError(
                "limit must be between 1 and 100", details={"limit": limit}
            )

        async with self._lock:
            candidates = [
                s
                for s in self._tenant_sessions(tenant_id).values()
                if not s.is_expired()
                and (principal_id is None or s.principal_id == principal_id)
            ]

        # Newest first; the id breaks ties so the ordering is total and stable.
        candidates.sort(key=lambda s: (s.created_at, s.id), reverse=True)

        if cursor is not None:
            after_created, after_id = _decode_cursor(cursor)
            candidates = [
                s for s in candidates if (s.created_at, s.id) < (after_created, after_id)
            ]

        window = candidates[: limit + 1]
        has_more = len(window) > limit
        items = tuple(window[:limit])
        return Page(
            items=items,
            next_cursor=_encode_cursor(items[-1]) if has_more and items else None,
        )

    async def delete_session(self, *, tenant_id: str, session_id: str) -> None:
        async with self._lock:
            self._load(tenant_id, session_id)
            self._tenant_sessions(tenant_id).pop(session_id, None)
            self._tenant_messages(tenant_id).pop(session_id, None)

    async def append_message(
        self, *, tenant_id: str, session_id: str, message: Message
    ) -> Message:
        if message.tenant_id != tenant_id or message.session_id != session_id:
            raise ValidationError("message does not belong to the target session")

        async with self._lock:
            session = self._load(tenant_id, session_id)
            transcript = self._tenant_messages(tenant_id).setdefault(session_id, [])
            transcript.append(message)
            if len(transcript) > self._history_cap:
                del transcript[: len(transcript) - self._history_cap]
            self._tenant_sessions(tenant_id)[session_id] = session.touch(
                message_delta=1, ttl=self._ttl
            )
            return message

    async def get_history(
        self, *, tenant_id: str, session_id: str, limit: int = 50
    ) -> tuple[Message, ...]:
        async with self._lock:
            self._load(tenant_id, session_id)
            transcript = self._tenant_messages(tenant_id).get(session_id, [])
            return tuple(transcript[-limit:]) if limit > 0 else ()

    async def purge_expired(self) -> int:
        now = datetime.now(UTC)
        removed = 0
        async with self._lock:
            for tenant_id, sessions in self._sessions.items():
                expired = [sid for sid, s in sessions.items() if s.is_expired(now=now)]
                for sid in expired:
                    sessions.pop(sid, None)
                    self._tenant_messages(tenant_id).pop(sid, None)
                removed += len(expired)
        return removed

    # -- diagnostics -------------------------------------------------------

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {
                "tenants": len(self._sessions),
                "sessions": sum(len(v) for v in self._sessions.values()),
                "messages": sum(
                    len(t) for tenant in self._messages.values() for t in tenant.values()
                ),
            }
