"""Native in-process inference via ``llama-cpp-python``.

Replaces the Ollama HTTP client. The weights are loaded into this process from a
``.gguf`` file and generation runs against them directly -- no model server, no
network hop.

Three properties of llama.cpp drive this whole design:

1. **The ``Llama`` object is not thread-safe.** Two concurrent calls against one
   instance corrupt its KV cache. Every call is therefore funnelled through a
   single-worker executor, which serialises them by construction rather than by
   remembering to take a lock.
2. **Calls are blocking C++.** Running one on the event loop stalls every other
   request in the worker, so nothing touches ``Llama`` outside that executor.
3. **Loading is slow and memory-heavy.** Weights load once, at startup, and are
   shared for the process lifetime.

The consequence worth being explicit about: the model is a *serialised*
resource. One instance serves one generation at a time, so under concurrent
load requests queue. ``max_queue_depth`` bounds that queue and callers beyond it
are rejected with a retryable error rather than left to burn their deadline in
a queue they cannot see.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bat.domain.conversation import ToolCall
from bat.domain.errors import RateLimitError, UpstreamError, UpstreamTimeoutError
from bat.ports.llm import ChatMessage, Completion, ToolSpec
from bat.settings import ModelSettings

logger = logging.getLogger("bat.llm.llama_cpp")

#: Sentinel pushed onto the stream queue to mark end-of-generation.
_DONE = object()


class ModelUnavailableError(UpstreamError):
    """The weights are not configured, not present, or failed to load."""

    code = "model_unavailable"
    status = 503
    public = True


def _register_cuda_dll_paths() -> None:
    """Make the pip-installed CUDA runtime discoverable on Windows.

    The CUDA build of llama-cpp-python links against ``cublas64_12.dll`` and
    friends. Those normally come from a CUDA Toolkit install; on a machine
    without one the ``nvidia-*-cu12`` wheels supply them, but pip drops them
    under ``site-packages/nvidia/*/bin``, which is not on Windows' DLL search
    path. Without this, the import fails with a bare "Could not find module
    llama.dll" -- which names the wrong file entirely.

    Both mechanisms are applied on purpose. ``os.add_dll_directory`` is the
    modern one, but llama-cpp-python loads its library with
    ``winmode=RTLD_GLOBAL``, and that path does not consult those directories --
    so on its own it silently does nothing. Prepending to ``PATH`` is what
    actually resolves the dependency.

    Silent no-op elsewhere: on Linux the wheels carry their own rpath, and the
    CPU build needs none of this.
    """
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return  # No pip-installed CUDA runtime; a toolkit install may still work.

    directories = [
        str(binary_dir)
        for root in getattr(nvidia, "__path__", [])
        for binary_dir in Path(root).glob("*/bin")
        if binary_dir.is_dir()
    ]
    if not directories:
        return

    for directory in directories:
        with contextlib.suppress(OSError):
            os.add_dll_directory(directory)

    current = os.environ.get("PATH", "")
    missing = [d for d in directories if d not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current])


def _import_llama() -> Any:
    """Import ``llama_cpp`` lazily, with an actionable message if it is absent.

    Kept out of module scope so the package imports (and the test suite runs)
    on machines without the compiled extension installed.
    """
    _register_cuda_dll_paths()
    try:
        import llama_cpp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ModelUnavailableError(
            "llama-cpp-python is not installed. On Windows, PyPI has no wheel; "
            "install from the maintainer's index, e.g. "
            "'pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cpu' (or .../cu124 "
            "for GPU, which also needs 'pip install nvidia-cublas-cu12 "
            "nvidia-cuda-runtime-cu12')."
        ) from exc
    except (OSError, RuntimeError) as exc:  # pragma: no cover - env-dependent
        # A CUDA build whose runtime DLLs are missing surfaces here, and the
        # default message blames llama.dll rather than the real cause.
        raise ModelUnavailableError(
            f"llama-cpp-python is installed but its native library will not "
            f"load ({exc}). For a CUDA build, install the runtime with "
            "'pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12', or "
            "reinstall the CPU build."
        ) from exc
    return llama_cpp


class LlamaCppClient:
    """:class:`~bat.ports.llm.LLMClient` backed by a local ``.gguf`` model."""

    __slots__ = (
        "_executor",
        "_llama",
        "_load_error",
        "_load_lock",
        "_settings",
        "_slot",
        "_waiters",
        "_waiters_lock",
    )

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._llama: Any | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        # One worker == one generation at a time == thread-safety by design.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llama-infer"
        )
        self._slot = asyncio.Semaphore(1)
        self._waiters = 0
        self._waiters_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._llama is not None

    @property
    def model_name(self) -> str:
        return self._settings.name

    async def load(self) -> None:
        """Load the weights on the inference thread. Idempotent."""
        if self._llama is not None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_blocking)

    def _load_blocking(self) -> Any:
        """Runs only on the inference thread, so the lock is uncontended."""
        with self._load_lock:
            if self._llama is not None:
                return self._llama

            cfg = self._settings
            if cfg.model_path is None:
                raise ModelUnavailableError(
                    "model.model_path is not configured; set BAT_MODEL__MODEL_PATH "
                    "to a .gguf file"
                )
            path = Path(cfg.model_path)
            if not path.is_file():
                raise ModelUnavailableError(
                    f"model weights not found at {path}", details={"path": str(path)}
                )

            llama_cpp = _import_llama()
            started = time.perf_counter()
            logger.info(
                "loading model weights",
                extra={"path": str(path), "n_ctx": cfg.n_ctx,
                       "n_gpu_layers": cfg.n_gpu_layers},
            )
            try:
                self._llama = llama_cpp.Llama(
                    model_path=str(path),
                    n_ctx=cfg.n_ctx,
                    n_gpu_layers=cfg.n_gpu_layers,
                    n_threads=cfg.n_threads,
                    n_batch=cfg.n_batch,
                    chat_format=cfg.chat_format,
                    seed=cfg.seed if cfg.seed is not None else -1,
                    use_mmap=cfg.use_mmap,
                    use_mlock=cfg.use_mlock,
                    verbose=cfg.verbose,
                )
            except ModelUnavailableError:
                raise
            except Exception as exc:
                self._load_error = str(exc)
                raise ModelUnavailableError(
                    f"failed to load model weights: {exc}"
                ) from exc

            logger.info(
                "model loaded",
                extra={
                    "path": str(path),
                    "load_seconds": round(time.perf_counter() - started, 2),
                },
            )
            return self._llama

    async def close(self) -> None:
        """Release the weights and stop the inference thread."""
        llama, self._llama = self._llama, None
        if llama is not None:
            with contextlib.suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, getattr(llama, "close", lambda: None)
                )
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- admission ---------------------------------------------------------

    def _enter_queue(self) -> None:
        with self._waiters_lock:
            if self._waiters >= self._settings.max_queue_depth:
                raise RateLimitError(
                    "inference queue is full; the model serves one generation at "
                    "a time",
                    retry_after_seconds=2.0,
                    details={"max_queue_depth": self._settings.max_queue_depth},
                )
            self._waiters += 1

    def _leave_queue(self) -> None:
        with self._waiters_lock:
            self._waiters = max(0, self._waiters - 1)

    @property
    def queue_depth(self) -> int:
        return self._waiters

    # -- LLMClient protocol ------------------------------------------------

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> Completion:
        payload = _to_wire(messages)
        kwargs = self._generation_kwargs(temperature, max_tokens)
        if tools:
            kwargs["tools"] = [_tool_to_wire(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        # Only a turn that advertises tools uses the tool-calling template; see
        # ModelSettings.tool_chat_format for why one template cannot do both.
        chat_format = (
            self._settings.tool_chat_format if tools else self._settings.chat_format
        )

        raw = await self._run(
            lambda llama: _with_chat_format(
                llama,
                chat_format,
                lambda: llama.create_chat_completion(messages=payload, **kwargs),
            ),
            timeout_s=timeout_s or self._settings.request_timeout_s,
        )
        return _parse_completion(raw)

    async def stream(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas.

        llama.cpp produces tokens from a blocking generator on the inference
        thread; each delta is handed to the event loop through a queue, so the
        loop is never blocked and back-pressure is natural.
        """
        payload = _to_wire(messages)
        kwargs = self._generation_kwargs(temperature, max_tokens)
        budget = timeout_s or self._settings.request_timeout_s

        loop = asyncio.get_running_loop()
        # Unbounded on purpose. A bounded queue would have the producer block in
        # `put` while the consumer sits in its `finally` waiting for the
        # producer -- a deadlock. Generation is already bounded by max_tokens,
        # so the queue cannot grow without limit.
        queue: asyncio.Queue[Any] = asyncio.Queue()
        # Checked between tokens so a client disconnect actually stops the work
        # rather than generating into a queue nobody is reading.
        cancel = threading.Event()

        base_format = self._settings.chat_format

        def produce(llama: Any) -> None:
            try:
                # Pinned explicitly rather than relying on whatever the last
                # call left behind: streaming never advertises tools, so it must
                # use the model's own template.
                stream = _with_chat_format(
                    llama,
                    base_format,
                    lambda: llama.create_chat_completion(
                        messages=payload, stream=True, **kwargs
                    ),
                )
                for chunk in stream:
                    if cancel.is_set():
                        break
                    delta = _extract_delta(chunk)
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
            except BaseException as exc:  # noqa: BLE001 - forwarded to consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        self._enter_queue()
        try:
            async with self._slot:
                llama = await self._ensure_loaded()
                future = loop.run_in_executor(self._executor, produce, llama)
                # Absolute deadline: a per-`get` timeout would restart the clock
                # on every token, so a slow-but-steady generation could run
                # forever without ever tripping it.
                expires_at = loop.time() + budget
                try:
                    while True:
                        remaining = expires_at - loop.time()
                        if remaining <= 0:
                            raise UpstreamTimeoutError(
                                f"generation exceeded {budget}s"
                            )
                        item = await asyncio.wait_for(queue.get(), timeout=remaining)
                        if item is _DONE:
                            break
                        if isinstance(item, BaseException):
                            raise UpstreamError(f"inference failed: {item}") from item
                        yield item
                except TimeoutError as exc:
                    raise UpstreamTimeoutError(
                        f"generation exceeded {budget}s"
                    ) from exc
                finally:
                    # Signal the worker, then let it unwind. It exits within one
                    # token because it checks `cancel` each iteration, and it
                    # never blocks on the queue.
                    cancel.set()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await asyncio.shield(future)
        finally:
            self._leave_queue()

    # -- internals ---------------------------------------------------------

    def _generation_kwargs(
        self, temperature: float | None, max_tokens: int | None
    ) -> dict[str, Any]:
        cfg = self._settings
        return {
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
            "top_p": cfg.top_p,
            "repeat_penalty": cfg.repeat_penalty,
        }

    async def _ensure_loaded(self) -> Any:
        if self._llama is None:
            await self.load()
        if self._llama is None:  # pragma: no cover - load raises before this
            raise ModelUnavailableError("model failed to load")
        return self._llama

    async def _run(self, fn: Any, *, timeout_s: float) -> Any:
        """Serialise one blocking call onto the inference thread."""
        self._enter_queue()
        try:
            async with self._slot:
                llama = await self._ensure_loaded()
                loop = asyncio.get_running_loop()
                try:
                    return await asyncio.wait_for(
                        loop.run_in_executor(self._executor, fn, llama),
                        timeout=timeout_s,
                    )
                except TimeoutError as exc:
                    # The C++ call cannot be interrupted; it finishes on the
                    # worker and its result is discarded. The slot is held until
                    # then, which is why the queue is bounded.
                    raise UpstreamTimeoutError(
                        f"generation exceeded {timeout_s}s"
                    ) from exc
        finally:
            self._leave_queue()


# -- wire helpers ----------------------------------------------------------


def _with_chat_format(llama: Any, chat_format: str | None, call: Any) -> Any:
    """Run ``call`` with ``llama.chat_format`` temporarily set.

    llama-cpp-python resolves the chat handler from ``self.chat_format`` at call
    time, so swapping the attribute is enough and no second model instance is
    needed. Safe despite being shared mutable state: every call to this client
    is serialised onto one worker thread, so no two swaps can overlap. The
    original value is always restored.
    """
    if chat_format is None:
        return call()
    previous = getattr(llama, "chat_format", None)
    if previous == chat_format:
        return call()
    llama.chat_format = chat_format
    try:
        return call()
    finally:
        llama.chat_format = previous


def _to_wire(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Map domain messages onto llama.cpp's OpenAI-shaped chat format."""
    wire: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        wire.append(entry)
    return wire


def _tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _parse_completion(raw: dict[str, Any]) -> Completion:
    """Read one llama.cpp chat completion, tolerating partial responses."""
    choices = raw.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}

    calls: list[ToolCall] = []
    for entry in message.get("tool_calls") or ():
        function = entry.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=entry.get("id") or f"call_{len(calls)}",
                name=name,
                arguments=_parse_arguments(function.get("arguments")),
            )
        )

    return Completion(
        content=message.get("content") or "",
        tool_calls=tuple(calls),
        finish_reason=choice.get("finish_reason") or "stop",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string* and are model-generated.

    Malformed JSON is a normal occurrence with small quantised models, so it is
    reported as data for the loop to feed back rather than raised.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"__unparsed__": str(raw)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _extract_delta(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or ()
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""
