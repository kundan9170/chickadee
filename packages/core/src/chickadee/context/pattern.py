"""The model's regex for `grep_in`, compiled and run within bounds.

The pattern is the model's. Compiling it is bounded by length and nesting, which the search's
timeout does not cover, and the search by `GREP_TIMEOUT_S`, since a pattern that backtracks
catastrophically would otherwise hold the session's lock for as long as it runs.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Final, Protocol, cast

import regex as _regex

__all__ = [
    "GREP_NESTING_MAX",
    "GREP_PATTERN_MAX",
    "GREP_TIMEOUT_S",
    "CompiledPattern",
    "PatternTooComplex",
    "compile_pattern",
    "search_lines",
]


GREP_PATTERN_MAX: Final = 1000


GREP_NESTING_MAX: Final = 50
"""Bounds on compiling the model's pattern, which the search timeout does not cover: both
compilers recurse on nested groups, and 500 of them overflows the stack."""


GREP_TIMEOUT_S: Final = 2.0
"""The whole search's budget. The pattern is the model's, and one that backtracks
catastrophically (`^(a+)+$` on a long near-match) would otherwise hold the session's lock
for as long as it runs."""


class CompiledPattern(Protocol):
    """The part of a `regex` pattern the search uses: `re`'s syntax, and a timeout `re`
    does not have."""

    @property
    def pattern(self) -> str: ...

    def search(self, string: str, *, timeout: float | None = None) -> object | None: ...


def compile_pattern(pattern: str) -> CompiledPattern:
    """`re` decides what the pattern means and says what is wrong with it — the language the
    tool promises is Python's own; `regex` only runs it, under a timeout.

    Compiling is bounded too: the timeout covers matching only, and a compiler recurses on
    nesting, so a pattern past `GREP_PATTERN_MAX` characters or `GREP_NESTING_MAX` levels of
    groups is refused before either compiler sees it, and a resource failure inside one is
    reported as the pattern's fault rather than escaping the tool call."""
    if len(pattern) > GREP_PATTERN_MAX:
        raise PatternTooComplex(f"longer than {GREP_PATTERN_MAX} characters")
    if _nesting(pattern) > GREP_NESTING_MAX:
        raise PatternTooComplex(f"groups nested deeper than {GREP_NESTING_MAX}")
    try:
        re.compile(pattern)
        return cast("CompiledPattern", _regex.compile(pattern))
    except (RecursionError, OverflowError, MemoryError) as exc:
        raise PatternTooComplex(f"too complex to compile ({type(exc).__name__})") from None


class PatternTooComplex(Exception):
    pass


def _nesting(pattern: str) -> int:
    """The deepest group nesting, skipping escaped parentheses and those inside a character
    class. Approximate on purpose: it only needs to bound what a compiler recurses on."""
    depth = deepest = 0
    escaped = in_class = False
    for ch in pattern:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
            deepest = max(deepest, depth)
        elif ch == ")":
            depth = max(depth - 1, 0)
    return deepest


def search_lines(regex: CompiledPattern, lines: Sequence[str]) -> list[int] | None:
    """The matching line numbers, or None when the search ran past `GREP_TIMEOUT_S`."""
    deadline = time.monotonic() + GREP_TIMEOUT_S
    hits: list[int] = []
    try:
        for n, text in enumerate(lines, start=1):
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            if regex.search(text, timeout=left) is not None:
                hits.append(n)
    except TimeoutError:
        return None
    return hits
