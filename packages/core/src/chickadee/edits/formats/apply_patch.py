"""`apply_patch`: Codex's V4A patch envelope, bare.

    *** Begin Patch
    *** Update File: app.py
    @@ def go
    -def go() -> int:
    +def go() -> str:
    *** Add File: new.py
    +print("hi")
    *** End Patch

An `*** Update File` section's hunks are split by its `@@` lines, and the scope each `@@` names
is kept beside its hunk (`Hunks.scopes`; consecutive `@@` lines nest), and the worker's draft
honours it. `*** Add File` creates. `*** Delete File`
removes a file, and `*** Move to`, the line after an `*** Update File` header, moves it, edited
by the section's hunks if it has any: the box's edit tool
performs both when undo could bring the file back.
"""

from __future__ import annotations

import re

from chickadee.edits.formats.base import (
    Create,
    Delete,
    FileEdits,
    Format,
    Hunks,
    Move,
    Targets,
    collected,
    diff_hunk,
)
from chickadee.edits.lines import split_lines
from chickadee.schema.adapter import ParseError

__all__ = ["ADD", "BEGIN", "DELETE", "END", "FORMAT", "MOVE", "UPDATE", "parse"]

BEGIN = "*** Begin Patch"
END = "*** End Patch"
UPDATE = "*** Update File:"
ADD = "*** Add File:"
DELETE = "*** Delete File:"
MOVE = "*** Move to:"
END_OF_FILE = "*** End of File"
_SCOPE = "@@"
_DIFF_HEADER = re.compile(r"^-\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
"""What follows `@@` in a unified diff's hunk header, which a V4A patch may carry: a range."""
_IN_HUNK = (" ", "+", "-", _SCOPE)
"""How an update section's lines start: a hunk line, or the `@@` that splits hunks."""


def parse(text: str, targets: Targets) -> FileEdits | ParseError:
    _ = targets  # every section names its path
    lines = split_lines(text)
    # Markers at a line's start only: an indented one is a context line quoting V4A.
    begin = next((i for i, ln in enumerate(lines) if ln.rstrip() == BEGIN), None)
    if begin is None:
        return FileEdits()
    end = next((i for i in range(begin + 1, len(lines)) if lines[i].rstrip() == END), None)
    if end is None:
        return ParseError(kind="bad_format", hint=f"close the patch with {END}")
    sections = _sections(lines[begin + 1 : end])
    if isinstance(sections, ParseError):
        return sections
    found: dict[str, Hunks | Create | Delete | Move] = {}
    for header, path, body in sections:
        if path in found:
            return ParseError(kind="bad_format", hint=f"{path}: one section per file")
        edit = _section(header, path, body)
        if isinstance(edit, ParseError):
            return edit
        found[path] = edit
    return collected(found)


def _section(
    header: str, path: str, body: list[str]
) -> Hunks | Create | Delete | Move | ParseError:
    """One file section's edit: a delete, a create, or an update's hunks, moved when its first
    line is a `*** Move to`."""
    if header == DELETE:
        if any(body):
            return ParseError(
                kind="bad_format", hint=f"{path}: a {DELETE} section holds no lines"
            )
        return Delete()
    if header == ADD:
        if any(ln and not ln.startswith("+") for ln in body):
            return ParseError(kind="bad_format", hint=f"{path}: each {ADD} line starts with +")
        added = diff_hunk(body)
        return Create(added.new if added is not None else "")
    moved_to: str | None = None
    if body and body[0].rstrip().startswith(MOVE):
        moved_to = body[0].rstrip()[len(MOVE) :].strip()
        body = body[1:]
        if not moved_to:
            return ParseError(kind="bad_format", hint=f"{path}: {MOVE} names a path")
    if any(ln and not ln.startswith(_IN_HUNK) for ln in body):
        return ParseError(
            kind="bad_format", hint=f"{path}: each hunk line starts with a space, - or +"
        )
    kept = [(scope, h) for scope, c in _chunks(body) if (h := diff_hunk(c)) is not None]
    hunks = [h for _, h in kept]
    if moved_to is not None and not hunks and not any(body):
        return Move(moved_to)
    if not hunks:
        return ParseError(kind="bad_format", hint=f"{path}: an {UPDATE} section needs a hunk")
    if any(h.old == h.new for h in hunks):
        return ParseError(
            kind="bad_format", hint=f"{path}: a hunk changes nothing; mark lines - or +"
        )
    edited = Hunks(tuple(hunks), tuple(scope for scope, _ in kept))
    return edited if moved_to is None else Move(moved_to, edited)


def _sections(lines: tuple[str, ...]) -> list[tuple[str, str, list[str]]] | ParseError:
    """Each file section: its header, its path and its lines, `*** End of File` dropped."""
    out: list[tuple[str, str, list[str]]] = []
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith(MOVE):
            # Only as an update section's first line: anywhere else it moves nothing.
            if not out or out[-1][0] != UPDATE or out[-1][2]:
                return ParseError(
                    kind="bad_format", hint=f"{MOVE} follows its {UPDATE} line directly"
                )
            out[-1][2].append(line)
            continue
        header = next((h for h in (UPDATE, ADD, DELETE) if stripped.startswith(h)), None)
        if header is not None:
            out.append((header, stripped[len(header) :].strip(), []))
        elif stripped == END_OF_FILE:
            continue
        elif out:
            out[-1][2].append(line)
        elif stripped:
            return ParseError(
                kind="bad_format", hint=f"open a file with {UPDATE} or {ADD} before its lines"
            )
    return out


def _chunks(body: list[str]) -> list[tuple[tuple[str, ...], list[str]]]:
    """An update section's hunks, split at its `@@` lines, each with the scope its `@@` lines
    name: consecutive ones nest (`@@ class Foo`, then `@@ def go`), a bare `@@` names none."""
    chunks: list[tuple[list[str], list[str]]] = [([], [])]
    for line in body:
        if line.startswith(_SCOPE):
            named = line[len(_SCOPE) :].strip()
            if _DIFF_HEADER.match(named):
                named = (
                    ""  # a unified diff's `@@ -1,2 +1,2 @@` numbers lines, it names no scope
                )
            if chunks[-1][1]:
                chunks.append(([], []))
            if named:
                chunks[-1][0].append(named)
        else:
            chunks[-1][1].append(line)
    return [(tuple(scope), lines) for scope, lines in chunks]


FORMAT = Format(
    name="apply_patch",
    parse=parse,
    template="formats/apply_patch.md",
    markers=(BEGIN,),
    create=f"an {ADD} section",
    unit="section",
    each=f"give each TARGETS path its {UPDATE} or {ADD} section",
    no_block=f"emit one {BEGIN} … {END} envelope, or one TOOL request",
)
