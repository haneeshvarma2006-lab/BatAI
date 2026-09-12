"""Local embeddings via ``llama-cpp-python``.

A second ``.gguf`` loaded with ``embedding=True``, on its own single-thread
executor. It is deliberately *not* the chat model's instance: embedding a
document while a generation is in flight would contend for the same serialised
`Llama` object and stall user-facing turns behind a bulk ingest.

With this in place the RAG pipeline has no external dependency at all --
chunking, embedding, storage and retrieval are all in-process.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bat.adapters.llama_cpp_client import ModelUnavailableError, _import_llama
from bat.domain.errors import UpstreamError
from bat.settings import EmbeddingSettings

logger = logging.getLogger("bat.rag.embedder")


class LlamaCppEmbedder:
    """:class:`~bat.ports.retrieval.Embedder` backed by a local ``.gguf``."""

    __slots__ = ("_dimensions", "_executor", "_llama", "_lock", "_settings")

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._llama: Any | None = None
        self._dimensions = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llama-embed"
        )

    @property
    def dimensions(self) -> int:
        """Vector width. Zero until the model has been loaded."""
        return self._dimensions

    @property
    def is_loaded(self) -> bool:
        return self._llama is not None

    async def load(self) -> None:
        if self._llama is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_blocking)

    def _load_blocking(self) -> Any:
        with self._lock:
            if self._llama is not None:
                return self._llama

            cfg = self._settings
            if cfg.model_path is None:
                raise ModelUnavailableError(
                    "embedding.model_path is not configured; set "
                    "BAT_EMBEDDING__MODEL_PATH to an embedding .gguf"
                )
            path = Path(cfg.model_path)
            if not path.is_file():
                raise ModelUnavailableError(
                    f"embedding weights not found at {path}",
                    details={"path": str(path)},
                )

            llama_cpp = _import_llama()
            started = time.perf_counter()
            try:
                self._llama = llama_cpp.Llama(
                    model_path=str(path),
                    embedding=True,
                    n_ctx=cfg.n_ctx,
                    n_gpu_layers=cfg.n_gpu_layers,
                    n_threads=cfg.n_threads,
                    n_batch=cfg.n_batch,
                    verbose=cfg.verbose,
                )
            except Exception as exc:
                raise ModelUnavailableError(
                    f"failed to load embedding model: {exc}"
                ) from exc

            # Probe once so `dimensions` is known before the first real query;
            # the vector store needs the width up front to create a collection.
            probe = _coerce_vectors(self._llama.create_embedding("dimension probe"))
            self._dimensions = len(probe[0]) if probe else 0
            logger.info(
                "embedding model loaded",
                extra={
                    "path": str(path),
                    "dimensions": self._dimensions,
                    "load_seconds": round(time.perf_counter() - started, 2),
                },
            )
            return self._llama

    async def close(self) -> None:
        llama, self._llama = self._llama, None
        if llama is not None:
            with contextlib.suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, getattr(llama, "close", lambda: None)
                )
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed in batches so a large ingest yields between chunks."""
        if not texts:
            return []
        if self._llama is None:
            await self.load()

        loop = asyncio.get_running_loop()
        vectors: list[list[float]] = []
        size = self._settings.batch_size
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            try:
                raw = await loop.run_in_executor(
                    self._executor, self._embed_blocking, batch
                )
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise UpstreamError(f"embedding failed: {exc}") from exc
            vectors.extend(raw)
        return vectors

    def _embed_blocking(self, batch: list[str]) -> list[list[float]]:
        llama = self._llama
        if llama is None:  # pragma: no cover - load raises first
            raise ModelUnavailableError("embedding model failed to load")
        vectors = _coerce_vectors(llama.create_embedding(batch))
        if self._settings.normalize:
            vectors = [_l2_normalize(v) for v in vectors]
        return vectors


def _coerce_vectors(response: Any) -> list[list[float]]:
    """Normalise llama.cpp's embedding response into a list of vectors.

    The shape varies with build and input type: a single text may return one
    embedding, a batch returns several, and some builds return per-token
    matrices that need mean-pooling into one vector per input.
    """
    if response is None:
        return []
    data = response.get("data") if isinstance(response, dict) else response
    if data is None:
        return []

    vectors: list[list[float]] = []
    for entry in data:
        raw = entry.get("embedding") if isinstance(entry, dict) else entry
        if raw is None:
            continue
        if raw and isinstance(raw[0], (list, tuple)):
            # Per-token matrix: mean-pool to a single sentence vector.
            vectors.append(_mean_pool([list(map(float, row)) for row in raw]))
        else:
            vectors.append([float(x) for x in raw])
    return vectors


def _mean_pool(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    width = len(rows[0])
    return [sum(row[i] for row in rows) / len(rows) for i in range(width)]


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """Unit-length vectors make cosine similarity a plain dot product."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


class HashingEmbedder:
    """Deterministic dependency-free embedder for tests and local dev.

    Not semantic -- it hashes character n-grams into a fixed-width vector. It
    exists so the RAG pipeline, the vector stores and the agent loop can be
    exercised end to end on a machine with no ``.gguf`` present. Never select it
    when a real embedding model is configured.
    """

    __slots__ = ("_dimensions",)

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            for gram in (token, *(token[i : i + 3] for i in range(len(token) - 2))):
                vector[_stable_bucket(gram, self._dimensions)] += 1.0
        return _l2_normalize(vector)


def _stable_bucket(text: str, buckets: int) -> int:
    """Process-independent bucket index.

    Python's built-in `hash()` is salted per process, so vectors written today
    would not match vectors computed after a restart -- fatal against a
    persistent store.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets
