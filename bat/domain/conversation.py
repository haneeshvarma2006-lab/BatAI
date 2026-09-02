"""Conversation aggregate: sessions, messages and tool invocations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Self

class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Prefixed, URL-safe, sortable-enough identifier."""
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def create(cls, name: str, arguments: dict[str, Any]) -> Self:
        return cls(id=new_id("call"), name=name, arguments=arguments)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The observation produced by executing a :class:`ToolCall`."""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a session. Immutable once appended."""

    id: str
    session_id: str
    tenant_id: str
    role: Role
    content: str
    created_at: datetime
    tool_calls: tuple[ToolCall, ...] = ()
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        tenant_id: str,
        role: Role,
        content: str,
        tool_calls: tuple[ToolCall, ...] = (),
        tool_result: ToolResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        return cls(
            id=new_id("msg"),
            session_id=session_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            created_at=utcnow(),
            tool_calls=tool_calls,
            tool_result=tool_result,
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class Session:
    """A conversation thread scoped to one tenant and owned by one principal."""

    id: str
    tenant_id: str
    principal_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    title: str | None = None
    message_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        principal_id: str,
        title: str | None = None,
        ttl: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        now = utcnow()
        return cls(
            id=new_id("sess"),
            tenant_id=tenant_id,
            principal_id=principal_id,
            created_at=now,
            updated_at=now,
            expires_at=now + ttl if ttl else None,
            title=title,
            metadata=metadata or {},
        )

    def is_expired(self, *, now: datetime | None = None) -> bool:
        return self.expires_at is not None and (now or utcnow()) >= self.expires_at

    def touch(self, *, message_delta: int = 0, ttl: timedelta | None = None) -> Self:
        """Return a copy with refreshed activity timestamps.

        Sliding expiry: any activity extends the window, so an in-use session is
        never reaped mid-conversation.
        """
        now = utcnow()
        return replace(
            self,
            updated_at=now,
            message_count=self.message_count + message_delta,
            expires_at=now + ttl if ttl else self.expires_at,
        )


@dataclass(frozen=True, slots=True)
class Page[T]:
    """One page of a keyset-paginated listing."""

    items: tuple[T, ...]
    next_cursor: str | None = None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None
