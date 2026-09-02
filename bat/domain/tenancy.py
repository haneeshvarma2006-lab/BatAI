"""Tenancy primitives.

Every operation in BAT happens on behalf of exactly one tenant. The
:class:`TenantContext` is created once per request at the authentication
boundary and threaded explicitly through every layer below it. Nothing reads
tenancy from a global or a contextvar: if a function can touch tenant data, the
tenant is in its signature, so a missing isolation check is a visible omission
rather than an invisible one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Self

from bat.domain.errors import AuthorizationError, ValidationError

_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


class Scope(StrEnum):
    """Capability scopes granted to a principal.

    Scopes are additive and checked at the API boundary *and* again at the tool
    boundary, so a compromised route cannot silently widen a principal's reach.
    """

    SESSIONS_READ = "sessions:read"
    SESSIONS_WRITE = "sessions:write"
    AGENT_INVOKE = "agent:invoke"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    TOOLS_EXECUTE = "tools:execute"
    ADMIN = "admin"


#: The scope set handed to an ordinary end user of a tenant.
DEFAULT_SCOPES: frozenset[Scope] = frozenset(
    {
        Scope.SESSIONS_READ,
        Scope.SESSIONS_WRITE,
        Scope.AGENT_INVOKE,
        Scope.MEMORY_READ,
    }
)


def validate_tenant_id(value: str) -> str:
    """Return ``value`` if it is a usable tenant id, else raise.

    Tenant ids end up in vector-store collection names, cache keys and log
    fields, so the character set is deliberately narrow.
    """
    if not _TENANT_ID_RE.fullmatch(value):
        raise ValidationError(
            "tenant_id must be 2-63 chars of [a-z0-9_-] and start alphanumeric",
            details={"tenant_id": value},
        )
    return value


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated actor behind a request."""

    id: str
    tenant_id: str
    scopes: frozenset[Scope]
    display_name: str | None = None
    #: Free-form claims from the credential; never trusted for authorization.
    claims: dict[str, str] = field(default_factory=dict)

    def has(self, scope: Scope) -> bool:
        return scope in self.scopes or Scope.ADMIN in self.scopes


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Per-request tenancy and identity, passed explicitly down the stack."""

    tenant_id: str
    principal: Principal
    request_id: str
    started_at: datetime

    def __post_init__(self) -> None:
        if self.principal.tenant_id != self.tenant_id:
            # Defensive: a principal must never be paired with a foreign tenant.
            raise AuthorizationError(
                "principal does not belong to the requested tenant",
                details={"tenant_id": self.tenant_id},
            )

    @property
    def principal_id(self) -> str:
        return self.principal.id

    def require(self, *scopes: Scope) -> Self:
        """Assert every scope in ``scopes``, returning self for chaining."""
        missing = [s for s in scopes if not self.principal.has(s)]
        if missing:
            raise AuthorizationError(
                "missing required scope",
                details={"required": sorted(str(s) for s in missing)},
            )
        return self

    def owns(self, resource_tenant_id: str) -> bool:
        return resource_tenant_id == self.tenant_id

    def log_fields(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "principal_id": self.principal.id,
            "request_id": self.request_id,
        }
