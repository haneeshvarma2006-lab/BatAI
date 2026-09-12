"""Postgres-backed session store.

Replaces the in-memory adapter for any deployment with more than one worker.
The in-memory one is correct but per-process: sessions vanish on restart and are
invisible to other replicas, which `Settings._harden` refuses in production.

Tenant isolation
----------------
The in-memory store got isolation structurally, by partitioning a dict per
tenant. SQL has no such shape, so isolation here is a discipline instead: every
statement carries ``tenant_id`` in its ``WHERE`` clause, and the primary keys are
composite ``(tenant_id, id)``. A query that forgets the tenant does not merely
return the wrong rows -- it fails to match the primary key index.

For defence in depth beyond this adapter, the same schema supports Postgres
row-level security: enable RLS on both tables and set ``app.tenant_id`` per
transaction. That is deliberately not on by default here, because it requires a
connection-per-tenant discipline that interacts badly with pgbouncer in
transaction mode; the notes in ``SCHEMA_RLS`` say how to turn it on.

Concurrency
-----------
``message_count`` is incremented in SQL (``count = count + 1``) rather than read,
incremented in Python and written back, so two concurrent appends to one session
cannot lose an increment.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from bat.domain.conversation import Message, Page, Role, Session, ToolCall, new_id
from bat.domain.errors import ConflictError, NotFoundError, UpstreamError, ValidationError

logger = logging.getLogger("bat.store.postgres")


SCHEMA = """
CREATE TABLE IF NOT EXISTS bat_sessions (
    tenant_id      TEXT        NOT NULL,
    id             TEXT        NOT NULL,
    principal_id   TEXT        NOT NULL,
    title          TEXT,
    created_at     TIMESTAMPTZ NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL,
    expires_at     TIMESTAMPTZ,
    message_count  INTEGER     NOT NULL DEFAULT 0,
    metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, id)
);

-- Serves the newest-first keyset listing, both with and without the
-- principal filter, without a sort.
CREATE INDEX IF NOT EXISTS bat_sessions_listing
    ON bat_sessions (tenant_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS bat_sessions_by_principal
    ON bat_sessions (tenant_id, principal_id, created_at DESC, id DESC);
-- Partial: only rows that can actually expire are worth scanning.
CREATE INDEX IF NOT EXISTS bat_sessions_expiry
    ON bat_sessions (expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS bat_messages (
    tenant_id    TEXT        NOT NULL,
    id           TEXT        NOT NULL,
    session_id   TEXT        NOT NULL,
    role         TEXT        NOT NULL,
    content      TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    tool_calls   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, id),
    -- Deleting a session takes its transcript with it, in one statement and
    -- without an application-side cascade that could be interrupted halfway.
    FOREIGN KEY (tenant_id, session_id)
        REFERENCES bat_sessions (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS bat_messages_transcript
    ON bat_messages (tenant_id, session_id, created_at, id);
"""

#: Optional hardening. Requires a session-scoped `SET app.tenant_id`, so it does
#: not work under pgbouncer in transaction pooling mode without care.
SCHEMA_RLS = """
ALTER TABLE bat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE bat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY bat_sessions_tenant ON bat_sessions
    USING (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY bat_messages_tenant ON bat_messages
    USING (tenant_id = current_setting('app.tenant_id', true));
"""


def _encode_cursor(session: Session) -> str:
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


class PostgresSessionStore:
    """:class:`~bat.ports.session_store.SessionStore` over asyncpg."""

    __slots__ = ("_dsn", "_history_cap", "_max_per_principal", "_pool", "_pool_size", "_ttl")

    def __init__(
        self,
        dsn: str,
        *,
        ttl: timedelta | None = None,
        max_sessions_per_principal: int = 200,
        history_cap: int = 500,
        pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._ttl = ttl
        self._max_per_principal = max_sessions_per_principal
        self._history_cap = history_cap
        self._pool_size = pool_size
        self._pool: Any = None

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Open the pool and apply the schema. Idempotent."""
        if self._pool is not None:
            return
        import asyncpg

        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=1,
                max_size=self._pool_size,
                command_timeout=30.0,
            )
        except Exception as exc:
            raise UpstreamError(f"could not connect to Postgres: {exc}") from exc

        async with self._pool.acquire() as connection:
            await connection.execute(SCHEMA)
        logger.info("postgres session store ready", extra={"pool_size": self._pool_size})

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _acquire(self) -> Any:
        if self._pool is None:
            await self.connect()
        return self._pool.acquire()

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
        session = Session.create(
            tenant_id=tenant_id,
            principal_id=principal_id,
            title=title,
            ttl=ttl or self._ttl,
            metadata=metadata,
        )
        async with await self._acquire() as connection:
            # One statement, so the quota check cannot race a concurrent create:
            # the INSERT ... SELECT only produces a row when the count is under
            # the cap, evaluated inside the same statement.
            inserted = await connection.fetchval(
                """
                INSERT INTO bat_sessions (
                    tenant_id, id, principal_id, title,
                    created_at, updated_at, expires_at, message_count, metadata
                )
                SELECT $1, $2, $3, $4, $5, $6, $7, 0, $8::jsonb
                WHERE (
                    SELECT count(*) FROM bat_sessions
                    WHERE tenant_id = $1 AND principal_id = $3
                      AND (expires_at IS NULL OR expires_at > now())
                ) < $9
                RETURNING id
                """,
                tenant_id,
                session.id,
                principal_id,
                session.title,
                session.created_at,
                session.updated_at,
                session.expires_at,
                json.dumps(session.metadata),
                self._max_per_principal,
            )
        if inserted is None:
            raise ConflictError(
                "session limit reached for this user",
                details={"limit": self._max_per_principal},
            )
        return session

    async def get_session(self, *, tenant_id: str, session_id: str) -> Session:
        async with await self._acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT * FROM bat_sessions
                WHERE tenant_id = $1 AND id = $2
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                tenant_id,
                session_id,
            )
        if row is None:
            raise NotFoundError("session not found", details={"session_id": session_id})
        return _to_session(row)

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

        after_created, after_id = (
            _decode_cursor(cursor) if cursor else (None, None)
        )
        async with await self._acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT * FROM bat_sessions
                WHERE tenant_id = $1
                  AND ($2::text IS NULL OR principal_id = $2)
                  AND (expires_at IS NULL OR expires_at > now())
                  AND (
                        $3::timestamptz IS NULL
                        OR (created_at, id) < ($3::timestamptz, $4::text)
                  )
                ORDER BY created_at DESC, id DESC
                LIMIT $5
                """,
                tenant_id,
                principal_id,
                after_created,
                after_id,
                limit + 1,
            )

        has_more = len(rows) > limit
        items = tuple(_to_session(r) for r in rows[:limit])
        return Page(
            items=items,
            next_cursor=_encode_cursor(items[-1]) if has_more and items else None,
        )

    async def delete_session(self, *, tenant_id: str, session_id: str) -> None:
        async with await self._acquire() as connection:
            # Messages go with it via ON DELETE CASCADE.
            deleted = await connection.fetchval(
                """
                DELETE FROM bat_sessions
                WHERE tenant_id = $1 AND id = $2
                RETURNING id
                """,
                tenant_id,
                session_id,
            )
        if deleted is None:
            raise NotFoundError("session not found", details={"session_id": session_id})

    async def append_message(
        self, *, tenant_id: str, session_id: str, message: Message
    ) -> Message:
        if message.tenant_id != tenant_id or message.session_id != session_id:
            raise ValidationError("message does not belong to the target session")

        async with await self._acquire() as connection:
            async with connection.transaction():
                # Bump first: it both proves the session exists (and is live) and
                # takes the row lock, so a concurrent delete cannot leave an
                # orphaned message behind.
                touched = await connection.fetchval(
                    """
                    UPDATE bat_sessions
                    SET updated_at = now(),
                        message_count = message_count + 1,
                        expires_at = CASE
                            WHEN $3::interval IS NULL THEN expires_at
                            ELSE now() + $3::interval
                        END
                    WHERE tenant_id = $1 AND id = $2
                      AND (expires_at IS NULL OR expires_at > now())
                    RETURNING id
                    """,
                    tenant_id,
                    session_id,
                    self._ttl,
                )
                if touched is None:
                    raise NotFoundError(
                        "session not found", details={"session_id": session_id}
                    )

                await connection.execute(
                    """
                    INSERT INTO bat_messages (
                        tenant_id, id, session_id, role, content,
                        created_at, tool_calls, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb)
                    """,
                    tenant_id,
                    message.id,
                    session_id,
                    str(message.role),
                    message.content,
                    message.created_at,
                    json.dumps([_call_to_json(c) for c in message.tool_calls]),
                    json.dumps(message.metadata),
                )

                # Trim in the same transaction, so the cap is never briefly
                # exceeded in a way another reader could observe.
                await connection.execute(
                    """
                    DELETE FROM bat_messages
                    WHERE tenant_id = $1 AND session_id = $2 AND id IN (
                        SELECT id FROM bat_messages
                        WHERE tenant_id = $1 AND session_id = $2
                        ORDER BY created_at DESC, id DESC
                        OFFSET $3
                    )
                    """,
                    tenant_id,
                    session_id,
                    self._history_cap,
                )
        return message

    async def get_history(
        self, *, tenant_id: str, session_id: str, limit: int = 50
    ) -> tuple[Message, ...]:
        if limit <= 0:
            return ()
        async with await self._acquire() as connection:
            exists = await connection.fetchval(
                """
                SELECT 1 FROM bat_sessions
                WHERE tenant_id = $1 AND id = $2
                  AND (expires_at IS NULL OR expires_at > now())
                """,
                tenant_id,
                session_id,
            )
            if exists is None:
                raise NotFoundError(
                    "session not found", details={"session_id": session_id}
                )
            rows = await connection.fetch(
                """
                SELECT * FROM (
                    SELECT * FROM bat_messages
                    WHERE tenant_id = $1 AND session_id = $2
                    ORDER BY created_at DESC, id DESC
                    LIMIT $3
                ) recent
                ORDER BY created_at ASC, id ASC
                """,
                tenant_id,
                session_id,
                limit,
            )
        return tuple(_to_message(r) for r in rows)

    async def purge_expired(self) -> int:
        async with await self._acquire() as connection:
            deleted = await connection.fetch(
                """
                DELETE FROM bat_sessions
                WHERE expires_at IS NOT NULL AND expires_at <= now()
                RETURNING id
                """
            )
        return len(deleted)


# -- row mapping -----------------------------------------------------------


def _loads(value: Any) -> Any:
    """asyncpg returns jsonb as text unless a codec is registered."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return json.loads(value)
    return value


def _to_session(row: Any) -> Session:
    return Session(
        id=row["id"],
        tenant_id=row["tenant_id"],
        principal_id=row["principal_id"],
        created_at=_as_utc(row["created_at"]),
        updated_at=_as_utc(row["updated_at"]),
        expires_at=_as_utc(row["expires_at"]) if row["expires_at"] else None,
        title=row["title"],
        message_count=row["message_count"],
        metadata=_loads(row["metadata"]) or {},
    )


def _to_message(row: Any) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        tenant_id=row["tenant_id"],
        role=Role(row["role"]),
        content=row["content"],
        created_at=_as_utc(row["created_at"]),
        tool_calls=tuple(
            ToolCall(
                id=c.get("id") or new_id("call"),
                name=c["name"],
                arguments=c.get("arguments") or {},
            )
            for c in (_loads(row["tool_calls"]) or ())
            if c.get("name")
        ),
        metadata=_loads(row["metadata"]) or {},
    )


def _call_to_json(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _as_utc(value: datetime) -> datetime:
    """TIMESTAMPTZ round-trips as aware; normalise anyway for equality checks."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

