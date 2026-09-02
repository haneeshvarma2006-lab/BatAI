"""Dependency injection.

Long-lived collaborators are built once in the lifespan and stashed on
``app.state``; these providers read them back with types attached. Nothing is a
module-level global, so tests construct an isolated app per case and override
any single dependency without touching the others.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from bat.api.security import ApiKeyAuthenticator
from bat.domain.tenancy import Scope, TenantContext
from bat.observability import bind_tenant
from bat.ports.agent import AgentRunner
from bat.ports.session_store import SessionStore
from bat.ratelimit import ConcurrencyLimiter, TokenBucketLimiter
from bat.settings import Settings


def _state(request: Request, name: str) -> Any:
    value = getattr(request.app.state, name, None)
    if value is None:  # pragma: no cover - wiring bug, not a runtime condition
        raise RuntimeError(f"application state {name!r} was never initialised")
    return value


def get_settings_dep(request: Request) -> Settings:
    return _state(request, "settings")


def get_session_store(request: Request) -> SessionStore:
    return _state(request, "session_store")


def get_agent_runner(request: Request) -> AgentRunner:
    return _state(request, "agent_runner")


def get_authenticator(request: Request) -> ApiKeyAuthenticator:
    return _state(request, "authenticator")


def get_run_limiter(request: Request) -> ConcurrencyLimiter:
    return _state(request, "run_limiter")


async def get_context(
    request: Request,
    authenticator: Annotated[ApiKeyAuthenticator, Depends(get_authenticator)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> TenantContext:
    """Authenticate, then apply the tenant's request-rate budget.

    Order matters: rate limiting is keyed by tenant, so it can only run after
    the credential resolves. Unauthenticated floods are the edge proxy's job.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    context = authenticator.authenticate(request, request_id=request_id)
    bind_tenant(context.tenant_id, context.principal_id)

    if settings.rate_limit.enabled:
        limiter: TokenBucketLimiter | None = getattr(
            request.app.state, "rate_limiter", None
        )
        if limiter is not None:
            await limiter.acquire(context.tenant_id)

    request.state.context = context
    return context


CurrentContext = Annotated[TenantContext, Depends(get_context)]
Store = Annotated[SessionStore, Depends(get_session_store)]
Runner = Annotated[AgentRunner, Depends(get_agent_runner)]
Config = Annotated[Settings, Depends(get_settings_dep)]
RunLimiter = Annotated[ConcurrencyLimiter, Depends(get_run_limiter)]


def scoped(*scopes: Scope) -> Any:
    """Build a dependency that asserts scopes on the authenticated context.

    Used as a route dependency so the requirement is visible in the signature
    and in the generated OpenAPI, rather than buried in the handler body.
    """

    async def _dependency(context: CurrentContext) -> TenantContext:
        return context.require(*scopes)

    return _dependency


SessionsRead = Annotated[TenantContext, Depends(scoped(Scope.SESSIONS_READ))]
SessionsWrite = Annotated[TenantContext, Depends(scoped(Scope.SESSIONS_WRITE))]
AgentInvoke = Annotated[TenantContext, Depends(scoped(Scope.AGENT_INVOKE))]


__all__ = [
    "AgentInvoke",
    "Config",
    "CurrentContext",
    "RunLimiter",
    "Runner",
    "SessionsRead",
    "SessionsWrite",
    "Store",
    "get_agent_runner",
    "get_authenticator",
    "get_context",
    "get_session_store",
    "get_settings_dep",
    "scoped",
]
