"""What a re-read is answered by, and when a file's read cap is raised.

**A re-read of lines the model already holds** is answered by what it can still use, decided
from the content hash, the conversation and this memory, never from the call's words:

| The earlier view | Answer |
|---|---|
| the file changed since it | the new lines, marked changed: other bytes, so nothing is held |
| out of the conversation | the lines (nothing is held) |
| older than `Held.turns` turns or `Held.tokens` tokens | the lines again, counted |
| recent | a pointer and the next unread span, counted |
| the same span asked right after the pointer | the lines, `Held.insist` times, counted |
| past that | the pointer again, until the role's repetition bound ends it |

"Counted" is the role's `Repetition` rule's: it keys a read on the span it asks of the bytes it
reads (`Reader.repeat_key`), so one span asked again in other words is one span asked again.

**A file's cap is raised only when the model clearly needs it in one view.** Each read of a file
is classed against what its earlier reads showed: a first read, a continuation (it starts right
after what was shown), a read within what is held, a restart past the cap (it starts at or
before an earlier view's start and asks for more than one read shows), or `max_bytes` above the
cap. The last two are strikes; at `Raise.strikes` the file's cap is raised for the rest of the
conversation to the span asked, within `Raise.ceiling` and `Raise.share` percent of the context
still free, or the model is told how to read it instead.

Nothing here calls a model, reads a file or writes a line: the reader asks it, under its home's
lock, and says what it decided in the result it returns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from chickadee.context.homes import HeldView
from chickadee.context.results import size_text
from chickadee.context.spans import Span, coalesce, minus
from chickadee.context.terms import Held, Raise
from chickadee.edits.lines import parse_row, split_lines
from chickadee.limits import CHARS_PER_TOKEN

__all__ = [
    "HEADROOM",
    "HELD_HEAD",
    "POINTER_HEAD",
    "RAISED",
    "FileReads",
    "ReadMemory",
    "cover",
    "is_pointer",
    "next_unread",
    "pointer_note",
    "raise_to",
    "reread",
    "strikes",
    "too_big",
]


RAISED: Final = "[limit raised for this file: you asked {n} times for more than one read shows]"


@dataclass
class FileReads:
    """What one conversation's reads of one file showed, and what they earned."""

    shown: list[tuple[Span, ...]] = field(default_factory=lambda: list[tuple[Span, ...]]())
    strikes: int = 0
    raised: tuple[int, int] | None = None
    """The cap this file reads with since its raise: bytes and lines."""


@dataclass
class _Pointed:
    handle_id: str
    spans: tuple[Span, ...]
    insisted: int = 0


class ReadMemory:
    """One reader's memory of a conversation's reads, forgotten when
    the conversation changes: a raise and its strikes reset at the next task. Guarded by the
    reader's home lock, as everything the reader decides is."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._files: dict[str, FileReads] = {}
        self._pointed: _Pointed | None = None

    def at(self, conversation: str | None) -> None:
        """The conversation the next read lands in; a different one starts from nothing."""
        if conversation != self._key:
            self._key, self._files, self._pointed = conversation, {}, None

    def file(self, path: str) -> FileReads:
        return self._files.setdefault(path, FileReads())

    def insists(self, handle_id: str, view: HeldView, times: int) -> bool:
        """Whether a read of a recent `view` is shown rather than pointed at: it comes right
        after the pointer to the same view, and fewer than `times` showings were given since.
        Otherwise this read is the pointer, and remembered as the last one."""
        p = self._pointed
        if p is not None and (p.handle_id, p.spans) == (handle_id, view.spans):
            if p.insisted < times:
                p.insisted += 1
                return True
            return False
        self._pointed = _Pointed(handle_id, view.spans)
        return False

    def moved_on(self) -> None:
        """Any read but the one after a pointer: it is no longer 'right after'."""
        self._pointed = None


def reread(
    views: Sequence[HeldView], wanted: Sequence[Span], held: Held
) -> tuple[Literal["show", "again", "recent"], HeldView | None]:
    """Whether a read of `wanted` is of lines held, and how old they are. A span not wholly
    held is read as any read (`show`). The newest view it falls within decides; a span held
    only by several views together is as old as the oldest of them, and pointed at as all
    of them, so no line of it is taken as fresh when it is not. An old view's lines are
    shown `again`; a `recent` one is pointed at, or shown if the model insists
    (`ReadMemory.insists`)."""
    view = cover(views, wanted)
    if view is None:
        return "show", None
    if view.turns_ago > held.turns or view.tokens_ago > held.tokens:
        return "again", view
    return "recent", view


def cover(views: Sequence[HeldView], wanted: Sequence[Span]) -> HeldView | None:
    """The held view `wanted` falls within: the newest that holds it alone, else every view
    it overlaps taken together when they hold it between them, else None. What the pointer
    names, and what a repetition keys a held read on (`Reader.repeat_key`)."""
    within = [v for v in views if not minus(wanted, v.spans)]
    if within:
        return within[-1]
    parts = [v for v in views if _overlaps(v.spans, wanted)]
    if not parts or minus(wanted, coalesce(s for v in parts for s in v.spans)):
        return None
    return _together(parts)


def _overlaps(a: Sequence[Span], b: Sequence[Span]) -> bool:
    return any(x.start <= y.end and y.start <= x.end for x in a for y in b)


def _together(parts: Sequence[HeldView]) -> HeldView:
    """Several views as one: their lines, named by the newest call, as old as the oldest."""
    return HeldView(
        coalesce(s for v in parts for s in v.spans),
        parts[-1].named,
        parts[-1].calls_ago,
        max(v.turns_ago for v in parts),
        max(v.tokens_ago for v in parts),
    )


def next_unread(held: Sequence[Span], after: Span, total: int, width: int) -> Span | None:
    """The first lines past `after` not held, as wide as the read asked; None at the end."""
    start = after.end + 1
    for s in coalesce(held):
        if s.start <= start <= s.end:
            start = s.end + 1
    if start > total:
        return None
    return Span(start, min(total, start + max(width, 1) - 1))


def pointer_note(handle_id: str, view: HeldView, nxt: Span | None, total: int) -> str:
    """The pointer, not the lines (decision 10): where they already are, and what is next."""
    ago = "just now" if view.calls_ago <= 1 else f"{view.calls_ago} calls ago"
    spans = ", ".join(str(s) for s in view.spans)
    rest = (
        f"the next unread lines are {nxt}"
        if nxt is not None
        else f"you hold the file to its end, line {total:,}"
    )
    return f"{POINTER_HEAD}{spans} ({handle_id}) from {view.named}, {ago}; {rest}]"


POINTER_HEAD: Final = "[you already hold lines "
"""How a pointer begins (`pointer_note`): a read answered with where its lines are, not them."""

HELD_HEAD: Final = "[you already hold view "
"""How a read says which of its lines are already held (`views._held_note`): on an answer that
re-sends none of them, and on one that shows the lines beside them."""


def is_pointer(answer: str) -> bool:
    """Whether a `read`'s answer showed no lines because they are held: the pointer, or the held
    note on an answer that shows none. The progress watchdog counts such a read as yielding
    nothing."""
    if POINTER_HEAD in answer:
        return True
    return HELD_HEAD in answer and not any(
        parse_row(line) is not None for line in split_lines(answer)
    )


def strikes(
    reads: FileReads, wanted: Sequence[Span], *, fits: bool, held: Sequence[Span]
) -> bool:
    """Whether this read is a restart past the cap (decision 11): not the file's first read,
    not a continuation, not within what is held, but starting at or before an earlier view's
    start and asking for more than one read shows."""
    earlier = [s for shown in reads.shown for s in shown]
    if not earlier or not wanted or fits:
        return False
    start = wanted[0].start
    if start == reads.shown[-1][-1].end + 1:
        return False
    if held and not minus(wanted, held):
        return False
    return any(start <= s.start for s in earlier)


HEADROOM: Final = 1024
"""Bytes a raised view leaves under its ceiling for its header, notes and diagnostics: the
ceiling is the role's `result_max`, where its pipeline clips a result, and a raised read that
the clip then cut would be the cut it was raised to end."""


def raise_to(need: int, raise_: Raise, room: int | None) -> int | None:
    """The bytes a raised cap may reach for a span of `need` bytes, or None when it would pass
    the ceiling or `raise_.share` percent of the context still free."""
    limit = raise_.ceiling - HEADROOM
    if room is not None:
        limit = min(limit, int(room * CHARS_PER_TOKEN * raise_.share / 100))
    return need if need <= limit else None


def too_big(file_bytes: int, cap: int) -> str:
    """Why a raise was not given, and what to do instead (decision 11)."""
    return (
        f"[this file is {size_text(file_bytes)}; one read shows at most {size_text(cap)}; "
        'read it in parts with "lines", or by symbol with "at"]'
    )
