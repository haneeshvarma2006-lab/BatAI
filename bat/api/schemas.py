"""Wire DTOs.

Kept separate from the domain dataclasses on purpose. The domain is free to
change shape; these are the public contract, and the ``from_domain`` mappers are
the single place a change has to be reconciled. Note that ``tenant_id`` is never
echoed back -- the caller cannot influence it, so returning it only invites
clients to start trusting a field they should ignore.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bat.domain.conversation import Message, Role, Session

MAX_INPUT_CHARS = 32_000
TitleField = Annotated[str, Field(min_length=1, max_length=200)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- requests --------------------------------------------------------------


class CreateSessionRequest(_Base):
    title: TitleField | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=32)

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Metadata is caller-controlled and gets persisted, so bound it."""
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError(f"metadata key too long: {key[:32]}...")
            if isinstance(value, str) and len(value) > 1024:
                raise ValueError(f"metadata value for {key!r} exceeds 1024 chars")
        return v


class SendMessageRequest(_Base):
    content: Annotated[str, Field(min_length=1, max_length=MAX_INPUT_CHARS)]
    #: Client-supplied key that makes a retry safe to replay.
    idempotency_key: Annotated[str, Field(max_length=128)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=16)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must not be blank")
        return stripped


# -- responses -------------------------------------------------------------


class SessionResponse(_Base):
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    message_count: int
    metadata: dict[str, Any]

    @classmethod
    def from_domain(cls, session: Session) -> Self:
        return cls(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
            message_count=session.message_count,
            metadata=session.metadata,
        )


class ToolCallResponse(_Base):
    id: str
    name: str
    arguments: dict[str, Any]


class MessageResponse(_Base):
    id: str
    session_id: str
    role: Role
    content: str
    created_at: datetime
    tool_calls: tuple[ToolCallResponse, ...] = ()

    @classmethod
    def from_domain(cls, message: Message) -> Self:
        return cls(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            tool_calls=tuple(
                ToolCallResponse(id=c.id, name=c.name, arguments=c.arguments)
                for c in message.tool_calls
            ),
        )


class SessionListResponse(_Base):
    items: tuple[SessionResponse, ...]
    next_cursor: str | None = None
    has_more: bool = False


class MessageListResponse(_Base):
    items: tuple[MessageResponse, ...]


class TurnResponse(_Base):
    """The result of one non-streaming agent turn."""

    session_id: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    stop_reason: str
    steps: int
    usage: dict[str, int] = Field(default_factory=dict)


class HealthResponse(_Base):
    status: str
    service: str
    version: str
    environment: str


class ReadinessResponse(_Base):
    ready: bool
    checks: dict[str, str]


class WhoAmIResponse(_Base):
    tenant_id: str
    principal_id: str
    display_name: str | None
    scopes: tuple[str, ...]
