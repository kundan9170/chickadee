"""Anchors resolved on the worker's draft: a text anchor, what `edit` and
`apply_patch` share; a line anchor, what `replace_lines` and
`insert_lines` share; and a symbol's name, what `replace_symbol` and
`insert_after_symbol` share. A tool is only a resolver: it finds the region its anchor
names, with the one matcher (`edits/match.find`), the one line run (`edits/match.line_run`) or
the one symbol finder (`edits/symbols`), and
writes it through the call's `apply`, the draft's one write (guard row E7). No tool matches or
writes another way."""

from __future__ import annotations

from collections.abc import Sequence

from chickadee.edits.draft import DraftRefused
from chickadee.edits.hunk import MatchReason
from chickadee.edits.lines import AnchorKind
from chickadee.edits.match import MatchFailure, find, line_run
from chickadee.tools.context import DraftCall
from chickadee.tools.tool import Answer, TextAnswer

__all__ = ["answered", "replace_lines", "replace_symbol", "replace_text"]


def replace_text(
    call: DraftCall,
    path: str,
    old: str,
    new: str,
    *,
    lines: bool,
    replace_all: bool = False,
    scopes: Sequence[str] = (),
    label: str = "",
) -> DraftRefused | None:
    """`old` replaced by `new` in `path` as the call holds it now, or why not. `label` names
    the edit within its call (`edit 1: `), so a refusal says which."""
    text = call.text(path)
    if isinstance(text, DraftRefused):
        return text
    found = find(
        text,
        old,
        new,
        lines=lines,
        replace_all=replace_all,
        scopes=scopes,
        crlf=call.crlf(path),
    )
    if isinstance(found, MatchFailure):
        failure = "NO_MATCH" if found.reason is MatchReason.NO_MATCH else "AMBIGUOUS"
        return DraftRefused(failure, f"{path}: {label}{found.detail}", do=found.do)
    # From the last: each region is of the text before any of them was written.
    for one in reversed(found):
        refused = call.apply(path, one.region, one.content)
        if refused is not None:
            return refused
    return None


def replace_lines(
    call: DraftCall,
    path: str,
    start: tuple[int | str, AnchorKind],
    end: int | str | None,
    content: str,
) -> DraftRefused | None:
    """The lines from `start` to `end` of `path` replaced by `content`'s lines, or, with `end`
    None, `content`'s lines inserted after the line `start` names (an `after` anchor, `0` the
    file's start); or why not. Each anchor is resolved against the text as it now reads: a
    moved line is `STALE_ANCHOR`, never written."""
    text = call.text(path)
    if isinstance(text, DraftRefused):
        return text
    first = call.line(path, *start)
    if isinstance(first, DraftRefused):
        return first
    if end is None:
        found = line_run(text, first + 1, first, content)
        return call.apply(path, found.region, found.content)
    last = call.line(path, end, "line")
    if isinstance(last, DraftRefused):
        return last
    if last < first:
        return DraftRefused(
            "BAD_FORMAT",
            f"{path}: the range ends at line {last}, before its start at line {first}",
            do="to add lines without replacing any, insert them; otherwise end at or after "
            "the start",
        )
    found = line_run(text, first, last, content)
    return call.apply(path, found.region, found.content)


def replace_symbol(
    call: DraftCall, path: str, name: str, content: str, *, after: bool
) -> DraftRefused | None:
    """The declaration `name` names in `path`, decorators included, replaced by `content`'s
    lines; with `after`, `content`'s lines inserted after it instead. Resolved on the text as
    it now reads, so a declaration an earlier edit moved is found where it is."""
    text = call.text(path)
    if isinstance(text, DraftRefused):
        return text
    found = call.symbol(path, name)
    if isinstance(found, DraftRefused):
        return found
    first, last = (found.last + 1, found.last) if after else (found.first, found.last)
    run = line_run(text, first, last, content)
    return call.apply(path, run.region, run.content)


def answered(call: DraftCall) -> Answer:
    """How every edit tool's call ends once its edits are written: the call's answer, or its
    refusal when the syntax check undoes it."""
    said = call.answer()
    return said if isinstance(said, DraftRefused) else TextAnswer(said, "bytes")
