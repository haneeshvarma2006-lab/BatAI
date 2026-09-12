"""ChromaDB vector store.

Two things this adapter does differently from the legacy `memory/chroma_cloud.py`:

* **One collection per tenant**, named by :func:`tenant_namespace`, instead of
  three globally shared collections. Tenant isolation is then a property of
  which collection is opened, not of remembering a metadata filter.
* **Embeddings are supplied by us**, never by Chroma's default embedding
  function. Chroma's default silently downloads an ONNX model on first use;
  here the vectors come from the local `.gguf` embedder so the pipeline has no
  hidden network dependency and one consistent vector space.

Chroma's client is synchronous, so every call is pushed to a worker thread.
`PersistentClient` also holds a file lock and is single-process only -- with
more than one uvicorn worker, run Chroma in server mode (`vector.mode="http"`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from bat.domain.errors import UpstreamError
from bat.ports.retrieval import Chunk, MemoryKind, RetrievalQuery, RetrievedChunk
from bat.settings import VectorSettings

logger = logging.getLogger("bat.rag.chroma")


def _import_chroma() -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise UpstreamError(
            "chromadb is not installed; install it with 'pip install chromadb'"
        ) from exc
    return chromadb


class ChromaVectorStore:
    """:class:`~bat.ports.retrieval.VectorStore` over ChromaDB."""

    __slots__ = ("_client", "_collections", "_embed", "_lock", "_settings")

    def __init__(self, settings: VectorSettings, embedder) -> None:  # noqa: ANN001
        self._settings = settings
        self._embed = embedder
        self._client: Any | None = None
        self._collections: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = await asyncio.to_thread(self._build_client)

    def _build_client(self) -> Any:
        chromadb = _import_chroma()
        cfg = self._settings
        if cfg.mode == "http":
            return chromadb.HttpClient(host=cfg.host, port=cfg.port, ssl=cfg.ssl)
        if cfg.mode == "persistent":
            if cfg.path is None:  # pragma: no cover - settings validate this
                raise UpstreamError("vector.path is required for persistent mode")
            cfg.path.mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=str(cfg.path))
        return chromadb.EphemeralClient()

    async def _collection(self, namespace: str) -> Any:
        async with self._lock:
            cached = self._collections.get(namespace)
            if cached is not None:
                return cached
            if self._client is None:
                self._client = await asyncio.to_thread(self._build_client)
            collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=namespace,
                # Cosine, to match the normalised vectors the embedder emits.
                # Passed by keyword on purpose: the second *positional*
                # parameter is `configuration`, not `metadata`, so a dict
                # given positionally is rejected with a confusing
                # "'dict' object has no attribute 'serialize_to_json'".
                metadata={"hnsw:space": "cosine"},
                # Chroma otherwise installs a default embedding function that
                # downloads an ONNX model on first use. Every vector here comes
                # from the local .gguf embedder, so that would be a hidden
                # network dependency and a second, inconsistent vector space.
                embedding_function=None,
            )
            self._collections[namespace] = collection
            return collection

    # -- VectorStore protocol ---------------------------------------------

    async def upsert(self, *, namespace: str, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        collection = await self._collection(namespace)
        vectors = await self._embed.embed([c.text for c in chunks])

        await asyncio.to_thread(
            collection.upsert,
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[list(v) for v in vectors],
            metadatas=[_to_metadata(c) for c in chunks],
        )
        return len(chunks)

    async def search(
        self, *, namespace: str, query: RetrievalQuery
    ) -> Sequence[RetrievedChunk]:
        collection = await self._collection(namespace)
        probe = (await self._embed.embed([query.text]))[0]

        where = _build_where(query)
        try:
            raw = await asyncio.to_thread(
                collection.query,
                query_embeddings=[list(probe)],
                n_results=max(1, query.top_k),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise UpstreamError(f"vector search failed: {exc}") from exc

        return _to_results(raw, min_score=query.min_score)

    async def delete(self, *, namespace: str, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        collection = await self._collection(namespace)
        await asyncio.to_thread(collection.delete, ids=list(chunk_ids))
        return len(chunk_ids)

    async def drop_namespace(self, *, namespace: str) -> None:
        """Hard-delete a tenant's memory. Required for erasure requests."""
        if self._client is None:
            await self.connect()
        async with self._lock:
            self._collections.pop(namespace, None)
        try:
            await asyncio.to_thread(self._client.delete_collection, namespace)
        except Exception as exc:
            # Deleting an absent collection is success, not failure.
            logger.info(
                "drop_namespace no-op", extra={"namespace": namespace, "reason": str(exc)}
            )

    async def count(self, *, namespace: str) -> int:
        collection = await self._collection(namespace)
        return int(await asyncio.to_thread(collection.count))


# -- mapping helpers -------------------------------------------------------


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Chroma metadata values must be scalars, so everything else is dropped."""
    metadata: dict[str, Any] = {"kind": str(chunk.kind)}
    if chunk.source:
        metadata["source"] = chunk.source
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def _build_where(query: RetrievalQuery) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = []
    if query.kinds:
        clauses.append({"kind": {"$in": [str(k) for k in query.kinds]}})
    for key, value in query.filters.items():
        clauses.append({key: value})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _to_results(raw: dict[str, Any], *, min_score: float) -> list[RetrievedChunk]:
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else 1.0
        # Chroma reports cosine *distance*; the ports speak similarity.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        if score < min_score:
            continue
        results.append(
            RetrievedChunk(
                chunk=Chunk(
                    id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    kind=_parse_kind(metadata.pop("kind", None)),
                    source=metadata.pop("source", None),
                    metadata=metadata,
                ),
                score=score,
            )
        )
    return results


def _parse_kind(value: Any) -> MemoryKind:
    try:
        return MemoryKind(value)
    except (ValueError, TypeError):
        return MemoryKind.KNOWLEDGE
