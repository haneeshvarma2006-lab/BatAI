"""Domain error hierarchy.

Errors carry their own transport-independent ``code`` and a suggested HTTP
status. The API layer translates them into RFC 9457 problem documents; nothing
below the API layer needs to know about HTTP.
"""

from __future__ import annotations

from typing import Any


class BatError(Exception):
    """Base class for every error BAT raises deliberately."""

    code: str = "internal_error"
    status: int = 500
    #: Whether the message is safe to return to an untrusted caller verbatim.
    public: bool = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class AuthenticationError(BatError):
    """No credential was supplied, or it did not resolve to a principal."""

    code = "unauthenticated"
    status = 401
    public = True


class AuthorizationError(BatError):
    """The principal is known but lacks the required scope or tenancy."""

    code = "forbidden"
    status = 403
    public = True


class NotFoundError(BatError):
    """A resource does not exist *within the caller's tenant*.

    Cross-tenant reads must raise this rather than :class:`AuthorizationError`,
    so that a caller cannot probe for the existence of another tenant's data.
    """

    code = "not_found"
    status = 404
    public = True


class ConflictError(BatError):
    """The request conflicts with current resource state."""

    code = "conflict"
    status = 409
    public = True


class ValidationError(BatError):
    """The request was well-formed but semantically invalid."""

    code = "invalid_request"
    status = 422
    public = True


class RateLimitError(BatError):
    """The tenant exceeded its allotted request budget."""

    code = "rate_limited"
    status = 429
    public = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after_seconds = retry_after_seconds


class ToolError(BatError):
    """A tool failed, or was denied by policy.

    Tool failures are *observations* fed back into the agent loop, not fatal
    errors, so this is caught by the runner rather than surfacing as a 5xx.
    """

    code = "tool_error"
    status = 400
    public = True


class ToolDeniedError(ToolError):
    """A tool invocation was rejected by the tenant's capability policy."""

    code = "tool_denied"
    status = 403


class UpstreamError(BatError):
    """A dependency (model server, vector store) failed."""

    code = "upstream_error"
    status = 502


class UpstreamTimeoutError(UpstreamError):
    """A dependency did not respond within its deadline."""

    code = "upstream_timeout"
    status = 504
