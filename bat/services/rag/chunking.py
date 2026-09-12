"""Document chunking.

The legacy pipeline used ``text[i:i+1000]`` with no overlap, which splits
mid-word and mid-sentence: a fact straddling a boundary becomes unretrievable
because neither half carries the whole statement.

This splitter instead descends a hierarchy of separators -- paragraphs, then
lines, then sentences, then words -- and only falls back to a hard character cut
when a single word exceeds the window. Consecutive chunks overlap, so a fact cut
by one boundary still appears whole in a neighbour.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Tried in order, coarsest first, so chunks break at the most natural seam.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Collapse runaway whitespace without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    """Splits text into overlapping windows on natural boundaries."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split(self, text: str, *, source: str | None = None) -> Sequence[str]:
        cleaned = normalize(text)
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]
        return self._merge(self._atomize(cleaned, self.separators))

    def _atomize(self, text: str, separators: Sequence[str]) -> list[str]:
        """Break text into pieces no larger than ``chunk_size``.

        Recurses through finer separators; a piece still too long after the last
        separator is cut by character, which is the only option left for an
        unbroken run such as a base64 blob.
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        head, *rest = separators
        pieces: list[str] = []
        for part in text.split(head):
            if not part:
                continue
            # Put the separator back so sentences keep their punctuation.
            candidate = part + head if head.strip() else part
            if len(candidate) <= self.chunk_size:
                pieces.append(candidate)
            else:
                pieces.extend(self._atomize(candidate, rest))
        return pieces

    def _merge(self, pieces: Sequence[str]) -> list[str]:
        """Greedily pack pieces up to ``chunk_size``, carrying an overlap."""
        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for piece in pieces:
            if length + len(piece) > self.chunk_size and current:
                chunks.append("".join(current).strip())
                carry, carried = [], 0
                # Walk backwards so the overlap is the *tail* of the chunk just
                # emitted -- that is the part a following chunk continues from.
                for previous in reversed(current):
                    if carried + len(previous) > self.chunk_overlap:
                        break
                    carry.insert(0, previous)
                    carried += len(previous)
                if not carry and self.chunk_overlap:
                    # No whole piece fits the overlap budget -- happens whenever
                    # sentences are longer than the overlap. Carry a character
                    # tail instead, so the guarantee that neighbours share
                    # context actually holds rather than silently lapsing.
                    tail = current[-1][-self.chunk_overlap :]
                    carry, carried = [tail], len(tail)
                if carried + len(piece) > self.chunk_size:
                    # Carrying the overlap would push the *next* chunk past the
                    # limit. The size bound is the hard guarantee -- callers
                    # size it to a context window -- so the overlap yields.
                    carry, carried = [], 0
                current, length = carry, carried
            current.append(piece)
            length += len(piece)

        if current:
            tail = "".join(current).strip()
            if tail:
                chunks.append(tail)
        return [c for c in chunks if c]


def estimate_tokens(text: str) -> int:
    """Rough token count for budgeting.

    Deliberately an estimate: calling the real tokenizer would mean holding the
    serialised model slot just to measure a string. It runs ~4 chars/token and
    errs high, so a budget computed with it does not overflow the context
    window. Swap in `Llama.tokenize` if exactness ever matters more than
    contention.
    """
    return max(1, (len(text) + 3) // 4)
