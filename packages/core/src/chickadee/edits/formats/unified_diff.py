"""`unified_diff`: a unified diff, fenced or bare (promoted).

The `@@` numbers are ignored: a hunk's text is its anchor. Each hunk's old text is its context
and `-` lines, its new text its context and `+` lines, one exact-text hunk, so a pure insertion
needs a context line (an empty old is `AMBIGUOUS`). `--- /dev/null` creates the file from the
`+` lines; `+++ /dev/null`, a delete, and a rename are refused. A hunk is
whole lines, so it matches only at a line's end. A `--- `/`+++ ` pair is a file header only
when an `@@` line follows it: inside a hunk the same two lines are a changed `-- ` comment. A
diff never reaches the Applier as a diff: `ApplyMode.DIFF` stays unused.

`new_sides` is the one reading of the `@@` numbers, for a diff git wrote rather than a model:
where each hunk's lines now are (`context/changes.py`, `read(at="@changed")`).
"""

from __future__ import annotations

import re
from typing import Final

from chickadee.edits.formats.base import (
    Create,
    FileEdits,
    Format,
    Targets,
    collected,
    diff_hunk,
)
from chickadee.edits.hunk import Hunk
from chickadee.edits.lines import split_lines
from chickadee.schema.adapter import ParseError

__all__ = ["FORMAT", "HUNK", "NULL", "new_sides", "parse"]

HUNK = "@@"
"""A hunk's header line starts so."""
NULL = "/dev/null"
_OLD, _NEW = "--- ", "+++ "
_IN_HUNK = (" ", "+", "-", "\\")
_NEW_SIDE: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
"""A hunk header's new side: its first line, and how many lines it holds when not one."""


def parse(text: str, targets: Targets) -> FileEdits | ParseError:
    lines = split_lines(text)
    default = targets.writable[0] if len(targets.writable) == 1 else None
    path: str | None = None
    creating = False
    found: dict[str, list[Hunk] | Create] = {}
    n = 0
    i = 0
    while i < len(lines):
        if _header_at(lines, i):
            old_path = _path(lines[i][4:], targets)
            new_path = _path(lines[i + 1][4:], targets)
            if new_path == NULL:
                return ParseError(
                    kind="bad_format",
                    hint=f"{old_path}: a diff may not delete a file; deletes are the planner's",
                )
            if old_path not in (NULL, new_path):
                return ParseError(
                    kind="bad_format",
                    hint=f"{old_path}: a diff may not move a file to {new_path}; moves are the "
                    "planner's",
                )
            path, creating = new_path, old_path == NULL
            i += 2
            continue
        if not lines[i].startswith(HUNK):
            i += 1
            continue
        end = _hunk_end(lines, i + 1)
        body = lines[i + 1 : end]
        target = path or default
        if target is None:
            return ParseError(
                kind="bad_format",
                hint=f"hunk {n}: name its file with --- and +++ lines before its @@ line",
            )
        held = found.get(target)
        if isinstance(held, Create) or (held is not None and creating):
            return ParseError(
                kind="bad_format",
                hint=f"{target}: a diff from {NULL} creates the file in one hunk",
            )
        hunk = diff_hunk(body)
        if creating:
            found[target] = Create(hunk.new if hunk is not None else "")
        elif hunk is not None:
            if hunk.old == hunk.new:
                return ParseError(
                    kind="bad_format", hint=f"hunk {n}: it changes nothing; mark lines - or +"
                )
            found[target] = [*(held if isinstance(held, list) else []), hunk]
        n += 1
        i = end
    return collected(found)


def new_sides(text: str) -> tuple[tuple[int, int], ...]:
    """Each hunk header's new side, `(first line, line count)`, in order. A count of 0 is a
    hunk that only deletes: its first line is the one the deletion follows, 0 at the head."""
    found: list[tuple[int, int]] = []
    for line in split_lines(text):
        match = _NEW_SIDE.match(line)
        if match is not None:
            count = match.group(2)
            found.append((int(match.group(1)), 1 if count is None else int(count)))
    return tuple(found)


def _header_at(lines: tuple[str, ...], i: int) -> bool:
    """A file header: `--- `, `+++ ` and an `@@` line, in that order."""
    return (
        i + 2 < len(lines)
        and lines[i].startswith(_OLD)
        and lines[i + 1].startswith(_NEW)
        and lines[i + 2].startswith(HUNK)
    )


def _path(header: str, targets: Targets) -> str:
    """A header's path: before a tab's timestamp, without git's `a/` or `b/` unless the
    path a target names starts so."""
    path = header.split("\t", 1)[0].strip()
    if path in targets.named:
        return path
    return path[2:] if path.startswith(("a/", "b/")) else path


def _hunk_end(lines: tuple[str, ...], start: int) -> int:
    """The line after a hunk's last: one that is not a hunk line, or the next header."""
    for j in range(start, len(lines)):
        line = lines[j]
        if line.startswith(HUNK) or _header_at(lines, j):
            return j
        if line and not line.startswith(_IN_HUNK):
            return j
    return len(lines)


FORMAT = Format(
    name="unified_diff",
    parse=parse,
    template="formats/unified_diff.md",
    markers=(_OLD, HUNK),
    create=f"a diff from {NULL}",
    unit="diff",
    each="emit a diff for each TARGETS path",
    no_block="emit a unified diff, or one TOOL request",
)
