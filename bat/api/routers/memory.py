"""Tenant memory endpoints.

Lets a tenant load its own corpus into the RAG index and inspect what retrieval
would return for a query -- which is the difference between a pipeline that is
"wired up" and one that is actually usable.

Everything is namespaced by ``context.tenant_id``. The caller never names a
namespace, so there is no parameter through which one tenant could address
another's memory.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from bat.api.deps import Config, scoped
from bat.domain.errors import ValidationError
from bat.domain.tenancy import Scope, TenantContext
from bat.ports.retrieval import MemoryKind, RetrievalQuery
from bat.services.rag.chunking import estimate_tokens

logger = logging.getLogger("bat.api.memory")

router = APIRouter(prefix="/v1/memory", tags=["memory"])

MemoryRead = Annotated[TenantContext, Depends(scoped(Scope.MEMORY_READ))]
MemoryWrite = Annotated[TenantContext, Depends(scoped(Scope.MEMORY_WRITE))]

MAX_DOCUMENT_CHARS = 500_000


def get_memory(request: Request) -> Any:
    pipeline = getattr(request.app.state, "memory", None)
    if pipeline is None:  # pragma: no cover - wiring guard
        raise RuntimeError("memory pipeline was never initialised")
    return pipeline


Memory = Annotated[Any, Depends(get_memory)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IngestRequest(_Base):
    text: Annotated[str, Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)]
    kind: MemoryKind = MemoryKind.DOCUMENT
    source: Annotated[str, Field(max_length=512)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=16)


class IngestResponse(_Base):
    chunks: int
    kind: MemoryKind
    source: str | None


class RetrievedChunkResponse(_Base):
    text: str
    kind: MemoryKind
    source: str | None
    score: float


class SearchResponse(_Base):
    items: tuple[RetrievedChunkResponse, ...]
    query: str


class ContextResponse(_Base):
    """The exact context block the agent loop would prepend for this query."""

    context: str
    token_estimate: int


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Chunk, embed and index a document",
)
async def ingest_document(
    body: IngestRequest, context: MemoryWrite, memory: Memory
) -> IngestResponse:
    if not body.text.strip():
        raise ValidationError("text must not be blank")
    written = await memory.ingest(
        tenant_id=context.tenant_id,
        text=body.text,
        kind=body.kind,
        source=body.source,
        metadata=body.metadata,
    )
    return IngestResponse(chunks=written, kind=body.kind, source=body.source)


@router.get("/search", response_model=SearchResponse, summary="Query tenant memory")
async def search_memory(
    context: MemoryRead,
    memory: Memory,
    settings: Config,
    q: Annotated[str, Query(min_length=1, max_length=2000)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
    kind: Annotated[MemoryKind | None, Query()] = None,
) -> SearchResponse:
    results = await memory.retrieve(
        tenant_id=context.tenant_id,
        query=RetrievalQuery(
            text=q,
            top_k=top_k,
            kinds=(kind,) if kind else (),
            min_score=settings.vector.min_score,
        ),
    )
    return SearchResponse(
        query=q,
        items=tuple(
            RetrievedChunkResponse(
                text=r.chunk.text,
                kind=r.chunk.kind,
                source=r.chunk.source,
                score=round(r.score, 4),
            )
            for r in results
        ),
    )


@router.get(
    "/context",
    response_model=ContextResponse,
    summary="Preview the context block the agent would use",
)
async def preview_context(
    context: MemoryRead,
    memory: Memory,
    settings: Config,
    q: Annotated[str, Query(min_length=1, max_length=2000)],
) -> ContextResponse:
    """Makes retrieval debuggable: see exactly what the model will be shown."""
    block = await memory.build_context(
        tenant_id=context.tenant_id,
        query=q,
        token_budget=settings.vector.context_token_budget,
    )
    return ContextResponse(
        context=block, token_estimate=estimate_tokens(block) if block else 0
    )


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Erase all of this tenant's memory",
)
async def forget(context: MemoryWrite, memory: Memory) -> None:
    """Hard delete, for erasure requests. Scoped to the caller's own tenant."""
    await memory.forget_tenant(tenant_id=context.tenant_id)
    logger.info("tenant memory erased", extra={"tenant_id": context.tenant_id})


__all__ = ["router"]
