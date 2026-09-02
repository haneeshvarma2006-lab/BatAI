"""Test doubles.

:class:`FakeLlama` mimics the parts of ``llama_cpp.Llama`` the adapters touch,
so :class:`~bat.adapters.llama_cpp_client.LlamaCppClient` is exercised as
written -- threading, admission, streaming, cancellation and all -- on a machine
with neither the compiled extension nor any weights.

It deliberately keeps the two behaviours that make the real thing awkward:
calls are **blocking**, and concurrent calls against one instance are illegal.
`concurrent_calls` records any overlap so a test can assert the client really
serialises access rather than merely appearing to.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class FakeLlama:
    """A blocking, non-thread-safe stand-in for ``llama_cpp.Llama``."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.embedding = kwargs.get("embedding", False)
        self.calls: list[dict[str, Any]] = []
        self.concurrent_calls = 0
        self.reply = "Hello from the fake model."
        self.tool_calls: list[dict[str, Any]] = []
        self.delay_s = 0.0
        self._active = 0
        self._guard = threading.Lock()

    # -- overlap detection -------------------------------------------------

    def _enter(self) -> None:
        with self._guard:
            self._active += 1
            if self._active > 1:
                self.concurrent_calls += 1

    def _leave(self) -> None:
        with self._guard:
            self._active -= 1

    # -- llama_cpp surface -------------------------------------------------

    def create_chat_completion(
        self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any
    ) -> Any:
        self._enter()
        try:
            self.calls.append({"messages": messages, "stream": stream, **kwargs})
            if self.delay_s:
                time.sleep(self.delay_s)
            if stream:
                return self._stream()
            return self._buffered()
        finally:
            self._leave()

    def _buffered(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.reply}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
            message["content"] = ""
        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if self.tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }

    def _stream(self) -> Any:
        # A generator, exactly as llama.cpp returns, so the client's
        # thread-to-loop bridge is exercised rather than bypassed.
        def gen():
            for word in self.reply.split(" "):
                if self.delay_s:
                    time.sleep(self.delay_s)
                yield {"choices": [{"delta": {"content": word + " "}}]}

        return gen()

    def create_embedding(self, text: Any) -> dict[str, Any]:
        self._enter()
        try:
            texts = [text] if isinstance(text, str) else list(text)
            return {
                "data": [
                    {"embedding": [float(len(t) % 7), 1.0, 0.5, 0.25]} for t in texts
                ]
            }
        finally:
            self._leave()

    def close(self) -> None:
        return None


class FakeLlamaModule:
    """Stands in for the ``llama_cpp`` module in ``sys.modules``."""

    def __init__(self) -> None:
        self.instances: list[FakeLlama] = []

    def Llama(self, **kwargs: Any) -> FakeLlama:  # noqa: N802 - mirrors the real name
        instance = FakeLlama(**kwargs)
        self.instances.append(instance)
        return instance


class ScriptedLLM:
    """An :class:`~bat.ports.llm.LLMClient` that replays canned completions."""

    def __init__(self, completions: list[Any], stream_text: str = "streamed answer") -> None:
        self._completions = list(completions)
        self._stream_text = stream_text
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, messages, tools=(), **kwargs: Any) -> Any:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._completions:
            raise AssertionError("ScriptedLLM ran out of scripted completions")
        return self._completions.pop(0)

    async def stream(self, *, messages, tools=(), **kwargs: Any):
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        for word in self._stream_text.split(" "):
            yield word + " "

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1]["messages"][0].content if self.calls else ""
