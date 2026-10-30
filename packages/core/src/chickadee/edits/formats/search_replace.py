"""`search_replace`: Aider's SEARCH/REPLACE blocks.

    app.py
    ```python
    <<<<<<< SEARCH
    def go() -> int:
    =======
    def go() -> str:
    >>>>>>> REPLACE
    ```

A block's path is the last one named before it: a fence's info string (`path=` or a target),
or a line that is a path alone, before the fence or inside it (Aider's `diff-fenced`); `./` and
a heading's `#`s are dropped. A bare line is a path when it names one of the packet's paths, or
is a path with a directory (`src/x.py`): `README.md` or `v1.2` in prose is not. With one
writable target an unnamed block is that target's. Each block is one exact-text hunk of whole
lines, matched as lines, in order; several on a path are its hunks. A SEARCH with no
lines writes the whole file: a create, and then its path's only block.

A block divides at its `=======` line, so a block holding more than one is refused: where its
SEARCH ends would be a guess, and a wrong guess that is unique would apply.
"""

from __future__ import annotations

import re

from chickadee.edits.formats.base import (
    Create,
    FileEdits,
    Format,
    Targets,
    collected,
    hunk,
    whole_lines,
)
from chickadee.edits.formats.fences import fence_opener, fence_path
from chickadee.edits.hunk import Hunk
from chickadee.edits.lines import split_lines
from chickadee.schema.adapter import ParseError

__all__ = ["DIVIDER", "FORMAT", "MARKERS", "REPLACE", "SEARCH", "parse"]

SEARCH = "<<<<<<< SEARCH"
DIVIDER = "======="
REPLACE = ">>>>>>> REPLACE"
MARKERS = (SEARCH, REPLACE)
"""The two lines no file's own text is mistaken for: `=======` alone is a heading underline in
reStructuredText, and is a marker only between these."""

_PATH_LINE = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+")
"""A line that is a path with a directory alone: one token, no space."""


def parse(text: str, targets: Targets) -> FileEdits | ParseError:
    lines = split_lines(text)
    default = targets.writable[0] if len(targets.writable) == 1 else None
    path: str | None = None
    found: dict[str, list[Hunk] | Create] = {}
    n = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != SEARCH:
            named = _named(line, targets)
            if named is not None:
                path = named
            i += 1
            continue
        divider = _next(lines, i + 1, DIVIDER)
        end = _next(lines, divider + 1, REPLACE) if divider is not None else None
        if divider is None or end is None:
            return ParseError(
                kind="bad_format",
                hint=f"block {n}: {SEARCH} needs a {DIVIDER} line and a {REPLACE} line",
            )
        target = path or default
        if target is None:
            return ParseError(
                kind="bad_format",
                hint=f"block {n}: name its TARGETS path on the line before its fence",
            )
        dividers = sum(ln.strip() == DIVIDER for ln in lines[i + 1 : end])
        if dividers > 1:
            return ParseError(
                kind="bad_format",
                hint=f"block {n}: it holds {dividers} {DIVIDER} lines, so where its SEARCH "
                "ends is ambiguous; anchor on lines without one (an underline may be any "
                "length at least its title's)",
            )
        searched, written = lines[i + 1 : divider], lines[divider + 1 : end]
        held = found.get(target)
        if isinstance(held, Create) or (held is not None and not searched):
            return ParseError(
                kind="bad_format",
                hint=f"{target}: an empty SEARCH writes the whole file, as its only block",
            )
        if not searched:
            found[target] = Create(whole_lines(written))
        else:
            found[target] = [*(held if isinstance(held, list) else []), hunk(searched, written)]
        n += 1
        i = end + 1
    return collected(found)


def _next(lines: tuple[str, ...], start: int, marker: str) -> int | None:
    """The next line that is `marker`, before another block opens."""
    for j in range(start, len(lines)):
        stripped = lines[j].strip()
        if stripped == marker:
            return j
        if stripped == SEARCH:
            return None
    return None


def _named(line: str, targets: Targets) -> str | None:
    """The path `line` names: a fence's info string, or a line that is a path alone."""
    info = fence_opener(line)
    if info is not None:
        return fence_path(info, sorted(targets.named))
    bare = line.strip().lstrip("#").strip().strip("`*").strip().removesuffix(":")
    bare = bare.removeprefix("./")
    if bare in targets.named or _PATH_LINE.fullmatch(bare):
        return bare
    return None


FORMAT = Format(
    name="search_replace",
    parse=parse,
    template="formats/search_replace.md",
    markers=(SEARCH,),
    create="one block with an empty SEARCH",
    unit="block",
    each="name each TARGETS path before its blocks",
    no_block="emit SEARCH/REPLACE blocks, or one TOOL request",
)
