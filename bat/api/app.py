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
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bat import __version__
from bat.adapters.agent_reference import ReferenceAgentRunner
from bat.adapters.llama_cpp_client import LlamaCppClient
from bat.adapters.llama_cpp_embedder import HashingEmbedder, LlamaCppEmbedder
from bat.adapters.ratelimit_redis import (
    RedisConcurrencyLimiter,
    RedisTokenBucketLimiter,
    create_redis_client,
)
from bat.adapters.session_store_memory import InMemorySessionStore
from bat.adapters.session_store_postgres import PostgresSessionStore
from bat.adapters.vector_chroma import ChromaVectorStore
from bat.adapters.vector_memory import InMemoryVectorStore
from bat.services.agent.loop import NativeAgentRunner
from bat.services.agent.tools import InMemoryToolRegistry, PolicyToolExecutor
from bat.services.rag.pipeline import LocalMemoryPipeline
from bat.tools.builtin import build_default_tools
from bat.api.errors import install_error_handlers
from bat.api.middleware import BodySizeLimitMiddleware, CorrelationMiddleware
from bat.api.routers import health, memory, messages, sessions
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

    Raises rather than silently degrading: a config asking for Postgres that
    quietly got an in-memory store would lose every session on restart while
    looking healthy.
    """
    ttl = timedelta(seconds=settings.session.ttl_seconds)
    if settings.session.backend == "memory":
        return InMemorySessionStore(
            ttl=ttl,
            max_sessions_per_principal=settings.session.max_sessions_per_principal,
        )
    if settings.session.backend == "postgres":
        if settings.session.dsn is None:  # pragma: no cover - settings validate
            raise RuntimeError("session.dsn is required for the postgres backend")
        return PostgresSessionStore(
            settings.session.dsn.get_secret_value(),
            ttl=ttl,
            max_sessions_per_principal=settings.session.max_sessions_per_principal,
            pool_size=settings.session.pool_size,
        )
    raise NotImplementedError(
        f"session backend {settings.session.backend!r} is configured but its "
        "adapter is not implemented yet"
    )


async def build_limiters(settings: Settings) -> tuple[Any, Any, Any]:
    """Return ``(rate_limiter, run_limiter, redis_client)``.

    The Redis pair verify their Lua at startup. Without that, a Redis without
    scripting fails on every call and `fail_open` swallows it, so the limiter
    silently stops limiting while every health check stays green.
    """
    cfg = settings.rate_limit
    if not cfg.enabled:
        return None, ConcurrencyLimiter(limit=cfg.max_concurrent_runs), None

    if cfg.backend == "memory":
        return (
            TokenBucketLimiter(
                rate_per_second=cfg.requests_per_second, burst=cfg.burst
            ),
            ConcurrencyLimiter(limit=cfg.max_concurrent_runs),
            None,
        )

    if cfg.dsn is None:  # pragma: no cover - settings validate this
        raise RuntimeError("rate_limit.dsn is required for the redis backend")
    client = await create_redis_client(cfg.dsn.get_secret_value())
    rate = RedisTokenBucketLimiter(
        client,
        rate_per_second=cfg.requests_per_second,
        burst=cfg.burst,
        fail_open=cfg.fail_open,
    )
    runs = RedisConcurrencyLimiter(
        client,
        limit=cfg.max_concurrent_runs,
        lease_ttl_s=cfg.lease_ttl_s,
        fail_open=cfg.fail_open,
    )
    await rate.verify()
    await runs.verify()
    logger.info("redis limiters ready", extra={"fail_open": cfg.fail_open})
    return rate, runs, client



def build_embedder(settings: Settings) -> Any:
    """Local .gguf embeddings when configured, else a deterministic stand-in.

    The stand-in is not semantic. It keeps the pipeline runnable without weights
    for development and tests; production is refused without a real model.
    """
    if settings.embedding.is_configured:
        return LlamaCppEmbedder(settings.embedding)
    if settings.environment.is_production:  # pragma: no cover - guarded earlier
        raise RuntimeError("embedding.model_path is required in production")
    logger.warning(
        "no embedding model configured; using non-semantic HashingEmbedder. "
        "Retrieval quality will be poor -- set BAT_EMBEDDING__MODEL_PATH."
    )
    return HashingEmbedder()


def build_vector_store(settings: Settings, embedder: Any) -> Any:
    if settings.vector.mode == "memory":
        return InMemoryVectorStore(embedder)
    return ChromaVectorStore(settings.vector, embedder)


def build_agent_runner(settings: Settings, llm: Any, memory: Any) -> Any:
    """The native loop when weights are configured, else the reference runner.

    Falling back keeps the API serviceable on a machine without a .gguf; the
    production guard in `Settings._harden` stops that fallback ever shipping.
    """
    if not settings.model.is_configured:
        logger.warning(
            "no model weights configured; serving the reference runner. "
            "Set BAT_MODEL__MODEL_PATH to a .gguf for real inference."
        )
        return ReferenceAgentRunner()

    # Everything is registered; the per-tenant policy decides what is
    # reachable. A tool the policy refuses is never advertised to the model.
    registry = InMemoryToolRegistry(
        build_default_tools(memory=memory, vector_settings=settings.vector)
    )
    return NativeAgentRunner(
        llm=llm,
        memory=memory,
        registry=registry,
        executor=PolicyToolExecutor(registry),
        vector_settings=settings.vector,
        model_name=settings.model.name,
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
        app.state.authenticator = ApiKeyAuthenticator(settings.api_keys)

        embedder = build_embedder(settings)
        vector_store = build_vector_store(settings, embedder)
        app.state.embedder = embedder
        app.state.vector_store = vector_store
        app.state.memory = LocalMemoryPipeline(
            store=vector_store, settings=settings.vector
        )
        app.state.llm = (
            LlamaCppClient(settings.model) if settings.model.is_configured else None
        )
        app.state.agent_runner = build_agent_runner(
            settings, app.state.llm, app.state.memory
        )
        rate_limiter, run_limiter, redis_client = await build_limiters(settings)
        app.state.rate_limiter = rate_limiter
        app.state.run_limiter = run_limiter
        app.state.redis = redis_client

        if not settings.api_keys:
            logger.warning(
                "no API keys configured; every request will be rejected with 401"
            )

        connect = getattr(app.state.session_store, "connect", None)
        if connect is not None:
            # Postgres opens its pool and applies the schema here, so a bad DSN
            # fails at boot rather than on a user's first request.
            await connect()

        if app.state.llm is not None and settings.model.preload:
            # Load before serving: the first request would otherwise block for
            # the whole load, and /readyz would lie about being ready.
            try:
                await app.state.llm.load()
            except Exception:
                logger.exception(
                    "model preload failed; the API will start but agent turns "
                    "will fail until the weights load"
                )

        purge_task = asyncio.create_task(_purge_loop(app.state.session_store))
        logger.info(
            "bat api started",
            extra={
                "version": __version__,
                "environment": str(settings.environment),
                "session_backend": settings.session.backend,
                "rate_limit_backend": settings.rate_limit.backend,
                "model": settings.model.name if settings.model.is_configured else None,
                "vector_mode": settings.vector.mode,
                "enabled_tools": sorted(settings.agent.enabled_tools),
                "tenants_configured": len({k.tenant_id for k in settings.api_keys}),
            },
        )
        try:
            yield
        finally:
            purge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await purge_task
            for resource in (
                app.state.llm,
                embedder,
                app.state.session_store,
                redis_client,
            ):
                closer = getattr(resource, "close", None)
                if closer is not None:
                    with contextlib.suppress(Exception):
                        await closer()
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
    app.include_router(memory.router)

    return app


app = create_app
"""Module-level factory reference.

Run with ``uvicorn bat.api.app:app --factory``. Exposing the factory rather
than a constructed instance keeps import of this module side-effect free.
"""
