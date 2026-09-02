"""Application factory and lifespan.

``create_app`` takes settings rather than reading them, so a test builds an
isolated app with its own store, limiter and credentials in one line and no
global teardown. Everything long-lived is constructed once in the lifespan and
attached to ``app.state``; nothing is a module-level singleton.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bat import __version__
from bat.adapters.agent_reference import ReferenceAgentRunner
from bat.adapters.session_store_memory import InMemorySessionStore
from bat.api.errors import install_error_handlers
from bat.api.middleware import BodySizeLimitMiddleware, CorrelationMiddleware
from bat.api.routers import health, messages, sessions
from bat.api.security import ApiKeyAuthenticator
from bat.observability import configure_logging
from bat.ports.session_store import SessionStore
from bat.ratelimit import ConcurrencyLimiter, TokenBucketLimiter
from bat.settings import Settings, get_settings

logger = logging.getLogger("bat.api.app")

_PURGE_INTERVAL_S = 300.0

DESCRIPTION = """
Multi-tenant agent platform.

Authenticate with `Authorization: Bearer <api key>` (or `X-API-Key`). Every
resource is scoped to the tenant that owns the credential; ids from another
tenant return 404, not 403.
"""


def build_session_store(settings: Settings) -> SessionStore:
    """Select the session backend.

    Only the in-memory backend exists today. `Settings._harden` already refuses
    to boot production with it, so this raises rather than silently degrading
    to a backend that loses data.
    """
    if settings.session.backend == "memory":
        return InMemorySessionStore(
            ttl=timedelta(seconds=settings.session.ttl_seconds),
            max_sessions_per_principal=settings.session.max_sessions_per_principal,
        )
    raise NotImplementedError(
        f"session backend {settings.session.backend!r} is configured but its "
        "adapter is not implemented yet"
    )


async def _purge_loop(store: SessionStore, interval_s: float = _PURGE_INTERVAL_S) -> None:
    """Background reaper for expired sessions."""
    while True:
        try:
            await asyncio.sleep(interval_s)
            removed = await store.purge_expired()
            if removed:
                logger.info("purged expired sessions", extra={"removed": removed})
        except asyncio.CancelledError:
            raise
        except Exception:
            # A reaper failure must never take the process down with it.
            logger.exception("session purge failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_output=settings.json_logs)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.session_store = build_session_store(settings)
        app.state.agent_runner = ReferenceAgentRunner()
        app.state.authenticator = ApiKeyAuthenticator(settings.api_keys)
        app.state.run_limiter = ConcurrencyLimiter(
            limit=settings.rate_limit.max_concurrent_runs
        )
        app.state.rate_limiter = (
            TokenBucketLimiter(
                rate_per_second=settings.rate_limit.requests_per_second,
                burst=settings.rate_limit.burst,
            )
            if settings.rate_limit.enabled
            else None
        )

        if not settings.api_keys:
            logger.warning(
                "no API keys configured; every request will be rejected with 401"
            )

        purge_task = asyncio.create_task(_purge_loop(app.state.session_store))
        logger.info(
            "bat api started",
            extra={
                "version": __version__,
                "environment": str(settings.environment),
                "session_backend": settings.session.backend,
                "tenants_configured": len({k.tenant_id for k in settings.api_keys}),
            },
        )
        try:
            yield
        finally:
            purge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await purge_task
            logger.info("bat api stopped")

    app = FastAPI(
        title="BAT Platform API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_url=settings.openapi_url,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
    )

    # Middleware runs outermost-last, so correlation is added last in order to
    # wrap everything else and stamp a request id on every response.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id"],
            expose_headers=["X-Request-Id", "Retry-After"],
            max_age=600,
        )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=1_048_576)
    app.add_middleware(CorrelationMiddleware)

    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(messages.router)

    return app


app = create_app
"""Module-level factory reference.

Run with ``uvicorn bat.api.app:app --factory``. Exposing the factory rather
than a constructed instance keeps import of this module side-effect free.
"""
