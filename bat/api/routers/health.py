"""Liveness, readiness and identity endpoints.

Liveness and readiness are deliberately different things: liveness answers "is
this process wedged, should the orchestrator restart it", readiness answers "can
this replica serve traffic right now". Conflating them makes a dependency
outage look like a crash loop.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response

from bat import __version__
from bat.api.deps import Config, CurrentContext, Store
from bat.api.schemas import HealthResponse, ReadinessResponse, WhoAmIResponse

logger = logging.getLogger("bat.api.health")

router = APIRouter(tags=["system"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz(settings: Config) -> HealthResponse:
    """Cheap and dependency-free: it must not fail when a backend is down."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=__version__,
        environment=str(settings.environment),
    )


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(
    store: Store, settings: Config, response: Response
) -> ReadinessResponse:
    """Exercise each dependency; report 503 when any of them is unusable."""
    checks: dict[str, str] = {}

    try:
        await store.purge_expired()
        checks["session_store"] = "ok"
    except Exception as exc:
        logger.warning("session store not ready", exc_info=exc)
        checks["session_store"] = "unavailable"

    # The model server and vector store are checked here once their adapters
    # land; declaring them "not_configured" is honest rather than optimistic.
    checks["vector_store"] = (
        "ok" if settings.vector.mode != "memory" else "ephemeral"
    )
    checks["model_server"] = "not_configured"

    ready = all(v != "unavailable" for v in checks.values())
    if not ready:
        response.status_code = 503
    return ReadinessResponse(ready=ready, checks=checks)


@router.get(
    "/v1/whoami",
    response_model=WhoAmIResponse,
    summary="Resolve the calling credential",
)
async def whoami(context: CurrentContext) -> WhoAmIResponse:
    """Lets a client confirm which tenant and scopes its key actually carries."""
    return WhoAmIResponse(
        tenant_id=context.tenant_id,
        principal_id=context.principal_id,
        display_name=context.principal.display_name,
        scopes=tuple(sorted(str(s) for s in context.principal.scopes)),
    )
