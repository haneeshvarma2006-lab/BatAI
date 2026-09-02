"""RAG memory port.

The pipeline is modelled as three replaceable stages — chunk, embed, retrieve —
so ChromaDB is an implementation detail behind :class:`VectorStore` rather than
a dependency of the agent loop.

Tenant isolation is structural: ``namespace`` is derived from the tenant id by
:func:`tenant_namespace` and every read and write is filtered by it. An
implementation that ignores the namespace is a data-leak bug, not a style
choice, so implementations should assert on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class MemoryKind(StrEnum):
    """Which memory bank a chunk belongs to.

    Kept as metadata on a single per-tenant collection rather than as separate
    collections, so one query can span banks and still be tenant-filtered.
    """

    KNOWLEDGE = "knowledge"
    USER_PROFILE = "user_profile"
    DOCUMENT = "document"
    CONVERSATION = "conversation"


def tenant_namespace(tenant_id: str) -> str:
    """Collection/namespace name for a tenant. Single source of truth."""
    return f"bat_t_{tenant_id}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text plus its provenance."""

    id: str
    text: str
    kind: MemoryKind
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by a search, with its relevance score."""

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A scoped retrieval request."""

    text: str
    top_k: int = 5
    kinds: tuple[MemoryKind, ...] = ()
    #: Minimum similarity; results below this are dropped rather than padded.
    min_score: float = 0.0
    filters: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into overlapping, embeddable units."""

    def split(self, text: str, *, source: str | None = None) -> Sequence[str]: ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors."""

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Tenant-namespaced vector persistence."""

    async def upsert(self, *, namespace: str, chunks: Sequence[Chunk]) -> int: ...

    async def search(
        self, *, namespace: str, query: RetrievalQuery
    ) -> Sequence[RetrievedChunk]: ...

    async def delete(self, *, namespace: str, chunk_ids: Sequence[str]) -> int: ...

    async def drop_namespace(self, *, namespace: str) -> None:
        """Hard-delete a tenant's memory. Required for GDPR erasure."""
        ...


@runtime_checkable
class MemoryPipeline(Protocol):
    """The tenant-facing façade over chunk + embed + store.

    This is what the agent loop depends on; it never sees a vector.
    """

    async def ingest(
        self,
        *,
        tenant_id: str,
        text: str,
        kind: MemoryKind,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed and store ``text``. Returns the chunk count."""
        ...

    async def retrieve(
        self, *, tenant_id: str, query: RetrievalQuery
    ) -> Sequence[RetrievedChunk]: ...

    async def build_context(
        self, *, tenant_id: str, query: str, token_budget: int
    ) -> str:
        """Retrieve and render a prompt-ready context block within a budget."""
        ...

    async def forget_tenant(self, *, tenant_id: str) -> None: ...
