"""`whole_file`: each target's complete new content in a fence, the default.

Today's parse, moved from `adapters/base` with its order kept, so the default's
answers do not move: a truncated fence, then no fence, then a fence naming a delete target,
then AD-7's pick among fences, then an empty block, then a path written inside the block
instead of on its fence line, then which target each fence is.
"""

from __future__ import annotations

from chickadee.edits.formats import search_replace
from chickadee.edits.formats.base import FileEdits, Format, Targets, Whole
from chickadee.edits.formats.fences import Fence, fence_path, scan_fences
from chickadee.schema.adapter import ParseError

__all__ = ["FORMAT", "parse"]

_TINY_LINES = 3


def parse(text: str, targets: Targets) -> FileEdits | ParseError:
    fences, truncated = scan_fences(text)
    if truncated:
        return ParseError(
            kind="truncated",
            hint="the last fence was not closed; copy the file inside ``` … ```",
        )
    closed = tuple(f for f in fences if f.closed)
    if not closed:
        return FileEdits()
    deleted = _naming_a_delete(closed, targets)
    if deleted is not None:
        # Before the pick: a fence for a file the Applier removes is never an example to
        # drop, and the packet rules refuse it with D108's reason.
        return deleted
    chosen = _pick(closed, n_targets=len(targets.writable))
    if chosen is None:
        return ParseError(
            kind="multiple_blocks",
            hint="emit one fence per TARGETS path, in that order; drop extra examples",
        )
    if any(not f.body.strip() for f in chosen):
        return ParseError(kind="bad_format", hint="empty block")
    writable = list(targets.writable)
    inside = next((p for f in chosen if (p := _path_inside(f, writable)) is not None), None)
    if inside is not None:
        return ParseError(
            kind="bad_format",
            hint=(
                f"put path={inside} on the opening fence line (```python path={inside}), "
                "not inside the block: the block is the whole file, its first line included"
            ),
        )
    if not writable:
        # Nothing to write: each fence under the path it names ("" for none), so the packet
        # rules answer that the packet has no TARGETS path, as they did before the move.
        return FileEdits(tuple((fence_path(f.info, []) or "", Whole(f.body)) for f in chosen))
    if len(chosen) == 1 and len(writable) == 1:
        named = fence_path(chosen[0].info, writable)
        if named is not None and named != writable[0]:
            return FileEdits(((named, Whole(chosen[0].body)),))
        return _checked(((writable[0], chosen[0]),))
    assigned: list[tuple[str, Fence]] = []
    for fence in chosen:
        path = fence_path(fence.info, writable)
        if path is None or path not in writable:
            return ParseError(
                kind="bad_format",
                hint="each fence info string must name a TARGETS path (path=src/foo.py)",
            )
        if any(path == p for p, _ in assigned):
            return ParseError(
                kind="bad_format",
                hint=f"duplicate fence for {path}; one block per target",
            )
        assigned.append((path, fence))
    return _checked(tuple(assigned))


def _path_inside(fence: Fence, writable: list[str]) -> str | None:
    """The target a fence that names no path writes as its first line, bare or as
    `path=<target>`: taken whole, that line would be written into the file."""
    if fence_path(fence.info, writable) is not None:
        return None
    first = next((line.strip() for line in fence.body.splitlines() if line.strip()), "")
    named = first.removeprefix("path=").strip()
    return named if named in writable else None


def _checked(assigned: tuple[tuple[str, Fence], ...]) -> FileEdits | ParseError:
    """Each fence as its target's new content; a fence holding SEARCH/REPLACE markers is
    refused rather than written into the file, markers and all."""
    for _, fence in assigned:
        if any(mark in fence.body for mark in search_replace.MARKERS):
            return ParseError(
                kind="bad_format",
                hint="emit the complete file; SEARCH/REPLACE hunks are not applied yet",
            )
    return FileEdits(tuple((path, Whole(fence.body)) for path, fence in assigned))


def _pick(closed: tuple[Fence, ...], n_targets: int) -> tuple[Fence, ...] | None:
    """A large patch plus tiny usage examples → the large one.

    Several large blocks on a one-file packet are competing patches. Several large
    blocks on a multi-file packet are the ordered set — mapping checks names later.
    """
    if len(closed) == 1:
        return closed
    large = [f for f in closed if f.line_count > _TINY_LINES]
    tiny = [f for f in closed if f.line_count <= _TINY_LINES]
    if n_targets <= 1:
        if len(large) == 1:
            return (large[0],)
        if not large and tiny:
            # all tiny: still one-file, pick the longest rather than fail on a 3-line file
            return (max(closed, key=lambda f: len(f.body)),)
        return None
    # Multi-file: every closed fence is a candidate. AD-7 is the one-file example-drop
    # rule; a 3-line real target must still be in the ordered set.
    return closed


def _naming_a_delete(fences: tuple[Fence, ...], targets: Targets) -> FileEdits | None:
    """The first fence naming a delete target, as that path's edit."""
    deletes = list(targets.deletes)
    if not deletes:
        return None
    for fence in fences:
        path = fence_path(fence.info, deletes)
        if path in deletes:
            return FileEdits(((path, Whole(fence.body)),))
    return None


FORMAT = Format(
    name="whole_file",
    parse=parse,
    template="formats/whole_file.md",
    markers=(),
    create="the whole file in a fence",
    unit="fence",
    each="emit one fence per TARGETS path, in that order, naming each path",
    no_block="wrap the complete file in a fenced block, or emit one TOOL request",
)
