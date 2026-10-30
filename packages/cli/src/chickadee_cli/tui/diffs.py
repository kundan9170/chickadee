"""The session's committed diff, split by file and hunk, for the diff screen
(the mockup): `HandoffSummary.diff` is git's unified diff of the session's
commits from its base, as text; this reads it, and nothing else.

**Selected from the wire, never invented.** Every line is the diff's own, made safe to draw by
`render/diff.shown_line` (the coder's bytes: an escape sequence in them would move the cursor
on the screen the person decides from), and each line's numbers are counted from its hunk's
`@@ -a,b +c,d @@` header, as git means them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from chickadee_cli.render.diff import shown_line

__all__ = ["DiffHunk", "DiffLine", "FileHunks", "split_diff"]

type LineKind = Literal["context", "added", "removed", "note"]

_FILE: Final = re.compile(r"^diff --git a/(.*) b/(.*)$")
_HUNK: Final = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class DiffLine:
    kind: LineKind
    text: str
    """The line without its `+`, `-` or space, drawn safe (`shown_line`)."""
    old: int | None = None
    """Its number before the change; None for an added line or a note."""
    new: int | None = None
    """Its number after; None for a removed line or a note."""


@dataclass(frozen=True)
class DiffHunk:
    header: str
    """The `@@ … @@` line, with the enclosing function git names after it, if any."""
    lines: tuple[DiffLine, ...]


@dataclass(frozen=True)
class FileHunks:
    path: str
    hunks: tuple[DiffHunk, ...]
    notes: tuple[str, ...] = ()
    """What git says of the file beyond its lines: a new or deleted file, a rename, a binary."""

    @property
    def added(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "added")

    @property
    def removed(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "removed")


def split_diff(text: str) -> dict[str, FileHunks]:
    """`text`, a unified diff, by the path each file has after the change (before, for one the
    change deleted), in the diff's order."""
    files: dict[str, FileHunks] = {}
    path: str | None = None
    notes: list[str] = []
    hunks: list[DiffHunk] = []
    header: str | None = None
    lines: list[DiffLine] = []
    old = new = 0

    def close_hunk() -> None:
        nonlocal header, lines
        if header is not None:
            hunks.append(DiffHunk(header, tuple(lines)))
        header, lines = None, []

    def close_file() -> None:
        nonlocal notes, hunks
        close_hunk()
        if path is not None:
            files[path] = FileHunks(path, tuple(hunks), tuple(notes))
        notes, hunks = [], []

    for raw in text.split("\n"):
        started = _FILE.match(raw)
        if started:
            close_file()
            path = started.group(2)
            continue
        if path is None:
            continue
        at = _HUNK.match(raw)
        if at:
            close_hunk()
            header = shown_line(raw)
            old, new = int(at.group(1)), int(at.group(2))
            continue
        if header is None:
            if raw.startswith(("new file", "deleted file", "rename ", "Binary files")):
                notes.append(shown_line(raw))
            continue
        if raw.startswith("+"):
            lines.append(DiffLine("added", shown_line(raw[1:]), new=new))
            new += 1
        elif raw.startswith("-"):
            lines.append(DiffLine("removed", shown_line(raw[1:]), old=old))
            old += 1
        elif raw.startswith(" "):
            lines.append(DiffLine("context", shown_line(raw[1:]), old=old, new=new))
            old += 1
            new += 1
        elif raw.startswith("\\"):
            lines.append(DiffLine("note", shown_line(raw[1:].strip())))
    close_file()
    return files
