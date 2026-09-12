"""In-process vector store.

Exact brute-force cosine search over per-namespace lists. Linear in corpus size,
so it is a development and test backend rather than a production one -- but it
is a *correct* one, which makes it the reference the Chroma adapter is checked
against.

Namespaces are separate dicts, so a query can only ever see one tenant's
vectors. That is the same structural-isolation argument as the session store:
forgetting to filter yields an empty result, not another tenant's data.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass

from bat.ports.retrieval import Chunk, RetrievalQuery, RetrievedChunk


@dataclass(slots=True)
class _Entry:
    chunk: Chunk
    vector: tuple[float, ...]


class InMemoryVectorStore:
    """:class:`~bat.ports.retrieval.VectorStore` with exact cosine search."""

    __slots__ = ("_embed", "_lock", "_namespaces")

    def __init__(self, embedder) -> None:  # noqa: ANN001 - Embedder protocol
        self._embed = embedder
        self._namespaces: dict[str, dict[str, _Entry]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, *, namespace: str, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = await self._embed.embed([c.text for c in chunks])
        async with self._lock:
            space = self._namespaces.setdefault(namespace, {})
            for chunk, vector in zip(chunks, vectors, strict=True):
                space[chunk.id] = _Entry(chunk=chunk, vector=tuple(vector))
        return len(chunks)

    async def search(
        self, *, namespace: str, query: RetrievalQuery
    ) -> Sequence[RetrievedChunk]:
        async with self._lock:
            entries = list(self._namespaces.get(namespace, {}).values())
        if not entries:
            return []

        if query.kinds:
            wanted = set(query.kinds)
            entries = [e for e in entries if e.chunk.kind in wanted]
        for key, value in query.filters.items():
            entries = [e for e in entries if e.chunk.metadata.get(key) == value]
        if not entries:
            return []

        probe = (await self._embed.embed([query.text]))[0]
        scored = [
            RetrievedChunk(chunk=entry.chunk, score=_cosine(probe, entry.vector))
            for entry in entries
        ]
        scored = [s for s in scored if s.score >= query.min_score]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: query.top_k]

    async def delete(self, *, namespace: str, chunk_ids: Sequence[str]) -> int:
        async with self._lock:
            space = self._namespaces.get(namespace)
            if not space:
                return 0
            return sum(1 for cid in chunk_ids if space.pop(cid, None) is not None)

    async def drop_namespace(self, *, namespace: str) -> None:
        async with self._lock:
            self._namespaces.pop(namespace, None)

    async def count(self, *, namespace: str) -> int:
        async with self._lock:
            return len(self._namespaces.get(namespace, {}))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [0, 1].

    Negative similarity is treated as zero relevance: a chunk pointing away from
    the query is not evidence, and letting it go negative would make `min_score`
    thresholds behave unintuitively.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))
