"""Markdown fences in a model's reply: the one scanner, for the formats that read fences and
for the JSON rungs' fenced object (`adapters/base.parse_structured`).

Moved from `adapters/base` unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Fence", "fence_opener", "fence_path", "scan_fences"]

_OPEN = re.compile(r"(```|~~~)([^\n]*)\n")
_PATH_EQ = re.compile(r"(?:^|[\s;])path=(\S+)")


@dataclass(frozen=True)
class Fence:
    mark: str
    info: str
    body: str
    closed: bool

    @property
    def line_count(self) -> int:
        if not self.body:
            return 0
        return len(self.body.splitlines())


def fence_opener(line: str) -> str | None:
    """The info string of a line that opens a fence, or None for any other line."""
    m = _OPEN.match(line + "\n")
    return m.group(2).strip() if m is not None else None


def scan_fences(text: str) -> tuple[tuple[Fence, ...], bool]:
    """Left-to-right fence spans. An unclosed last opener is truncated, not a body."""
    fences: list[Fence] = []
    pos = 0
    truncated = False
    while True:
        match = _OPEN.search(text, pos)
        if match is None:
            break
        mark = match.group(1)
        info = match.group(2).strip()
        start = match.end()
        close_at = _closing(text, start, mark)
        if close_at is None:
            truncated = True
            fences.append(Fence(mark=mark, info=info, body=text[start:], closed=False))
            break
        fences.append(Fence(mark=mark, info=info, body=text[start:close_at], closed=True))
        pos = close_at + len(mark)
    return tuple(fences), truncated


def _closing(text: str, start: int, mark: str) -> int | None:
    """Index of the body end (the newline before the closing mark), or None."""
    needle = "\n" + mark
    at = text.find(needle, start - 1 if start > 0 else 0)
    # Empty body: opener's newline is immediately followed by the mark on the next line,
    # which `find` from `start` already covers via `needle` starting with `\n`.
    if at == -1:
        return None
    if at < start - 1:
        return None
    # The close must begin a line. `needle` is `\n``` so `at` is the newline before it.
    # Body excludes that newline.
    if at < start:
        # opener ended at `start`; the `\n` of the close *is* the opener's trailing
        # newline reused — empty body, close starts at `start`.
        if text.startswith(mark, start):
            return start
        return None
    return at


def fence_path(info: str, paths: list[str]) -> str | None:
    """The path a fence's info string names. `path=` is read even when it names none of
    `paths`, so a wrong path fails closed rather than falling to the default target."""
    if not info:
        return None
    marked = _PATH_EQ.search(" " + info)
    if marked:
        return marked.group(1)
    for token in info.split():
        if token in paths:
            return token
    return info if info in paths else None
