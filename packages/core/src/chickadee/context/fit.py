"""A view's lines, numbered and cut to the inline caps.

Every read is bounded once, by the tool that produced it: an inline byte cap and a line cap,
which `max_bytes` may only lower, and a `Truncation` naming where to continue whenever either
cut. A gap of at most `MERGE_GAP` lines inside a view is shown rather than marked.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from chickadee.context.handle import (
    Handle,
)
from chickadee.context.outline import (
    Outline,
    fit_rows,
)
from chickadee.context.results import (
    InvalidArgument,
    Truncation,
    size_text,
)
from chickadee.context.spans import (
    Span,
    coalesce,
    line_count,
    minus,
    numbers,
    runs,
)
from chickadee.context.terms import Caps
from chickadee.edits.lines import elided, number_lines, prefix_bytes

__all__ = [
    "MERGE_GAP",
    "Fit",
    "bridge",
    "cap_of",
    "cost_of",
    "cut_room",
    "fit_cut",
    "fit_lines",
    "fitted",
    "header_line",
    "no_room",
    "preview_text",
    "render_lines",
    "shown_cut",
    "truncation_of",
]


MERGE_GAP: Final = 5
"""A gap of at most this many lines inside a view is shown, not marked: three lines of code
are cheaper than a marker plus a false discontinuity."""


def header_line(h: Handle) -> str:
    """The line above a preview: what the handle names and its shape."""
    s = h.shape
    if s.is_image and s.width is not None and s.height is not None:
        what = f"{s.kind} · {s.width}x{s.height} · {size_text(s.bytes)}"
    elif s.is_pdf and s.pages is not None:
        noun = "page" if s.pages == 1 else "pages"
        what = f"{s.kind} · {s.pages:,} {noun} · {size_text(s.bytes)}"
    elif s.lines is None:
        what = f"{s.kind} · {size_text(s.bytes)} · not text"
    else:
        what = (
            f"{s.lines:,} line{'' if s.lines == 1 else 's'} · {size_text(s.bytes)} · {s.kind}"
        )
        if s.symbols is not None:
            what += f" · {s.symbols:,} symbols"
    return f"{h.label} — {what} [{h.id}]"


def preview_text(outline: Outline, ops: Sequence[str]) -> str:
    rows = list(outline.rows)
    if outline.note:
        rows.append(f"[{outline.note}]")
    rows.append("ops: " + (" · ".join(ops) if ops else "none"))
    return "\n".join(rows)


def fitted(outline: Outline, budget: int, what: str) -> Outline:
    """An outline's rows up to `budget` bytes, the rest counted. A preview names no entries:
    `at=` resolves against the whole outline, never a cut one."""
    return Outline(tuple(fit_rows(outline.rows, budget, what=what)), note=outline.note)


def fit_cut(
    lines: Sequence[str],
    n: int,
    caps: Caps,
    render: Callable[[Mapping[int, int]], str],
    *,
    credit: int = 0,
) -> tuple[dict[int, int], str] | None:
    """The room line `n` is shown cut with, and the content rendered with it: sized from the
    rendered form, then shrunk by whatever the render still overshoots — gap and edge
    markers included — so what this read newly sends is within the cap. `credit` is the
    bytes of an earlier view the render re-sends in place of the result that carried them.
    None when the cap leaves no room for any of the line."""
    room = cut_room(caps, len(lines), lines[n - 1])
    for _ in range(3):
        if room <= 0:
            return None
        content = render({n: room})
        over = len(content.encode("utf-8")) - credit - caps.inline_bytes
        if over <= 0:
            return {n: room}, content
        room -= over
    return None


def cost_of(lines: Sequence[str], spans: Iterable[Span]) -> int:
    """The rendered bytes of whole lines, costed as `fit_lines` costs them."""
    prefix = prefix_bytes(len(lines))
    return sum(prefix + len(lines[n - 1].encode("utf-8")) + 1 for n in numbers(spans))


def bridge(lines: Sequence[str], spans: Iterable[Span], room: int) -> tuple[Span, ...]:
    """`spans` in order, a gap of at most `MERGE_GAP` lines between two of them filled with
    the file's own lines — but only while those lines fit in
    `room`. A bridged line is sent whole and unmarked, so an uncosted bridge across one long
    line would carry it past every cap; a gap that does not fit stays a marked gap."""
    out: list[Span] = []
    for s in coalesce(spans):
        if out and s.start - out[-1].end - 1 <= MERGE_GAP:
            gap = Span(out[-1].end + 1, s.start - 1)
            cost = cost_of(lines, (gap,))
            if cost <= room:
                room -= cost
                out[-1] = Span(out[-1].start, s.end)
                continue
        out.append(s)
    return tuple(out)


def no_room(lines: Sequence[str], n: int, caps: Caps) -> InvalidArgument:
    """A cap too small for any of the one line asked for. Refused, not answered empty: an
    empty answer with a continuation would move past a line that was never shown."""
    size = len(lines[n - 1].encode("utf-8"))
    return InvalidArgument(
        "max_bytes",
        f"{caps.inline_bytes:,} bytes leaves no room for any of line {n} ({size:,} bytes)",
        do="send the read without max_bytes, or grep_in inside the line",
    )


def cut_room(caps: Caps, total: int, line: str) -> int:
    """Content bytes a line over the cap alone may show, so that the rendered row — its
    number, the bar, the kept head and the `…[+N bytes]` marker — stays within the cap."""
    size = len(line.encode("utf-8"))
    marker = len(_cut_mark(size).encode())
    return caps.inline_bytes - prefix_bytes(total) - marker


def _cut(text: str, room: int) -> str:
    raw = text.encode("utf-8")
    head = raw[: max(room, 0)].decode("utf-8", errors="ignore")
    return head + _cut_mark(len(raw) - len(head.encode("utf-8")))


def _cut_mark(left: int) -> str:
    return f"…[+{left:,} bytes]"


_CUT_MARK_RE: Final = re.compile(r"…\[\+[\d,]+ bytes\]\Z")


def shown_cut(text: str) -> bool:
    """Whether a rendered line's text is a line shown cut: its head and the marker for the
    rest (`_cut`), never the line. A line that itself ends in such a marker reads as cut too,
    which only ever costs a citation of it its content check."""
    return _CUT_MARK_RE.search(text) is not None


def render_lines(
    lines: Sequence[str],
    spans: Sequence[Span],
    *,
    cut: Mapping[int, int] | None = None,
    edges: Span | None = None,
) -> str:
    """`N│content` per line, gaps between spans marked. `cut` maps a line to the bytes
    it is shown with; `edges` also marks what the spans leave out at either end of it."""
    source: Sequence[str] = lines
    if cut:
        patched = list(lines)
        for n, room in cut.items():
            patched[n - 1] = _cut(lines[n - 1], room)
        source = patched
    rows = number_lines(source, spans)
    if edges is not None and spans:
        if spans[0].start > edges.start:
            rows.insert(0, elided(edges.start, spans[0].start - 1))
        if spans[-1].end < edges.end:
            rows.append(elided(spans[-1].end + 1, edges.end))
    return "\n".join(rows)


@dataclass(frozen=True)
class Fit:
    kept: tuple[Span, ...]
    cut: int | None = None
    """A first line shown cut, because alone it is over the byte cap."""
    reason: Literal["byte_cap", "line_cap"] | None = None


def fit_lines(lines: Sequence[str], need: Sequence[Span], caps: Caps) -> Fit:
    """The longest prefix of `need` under both caps, costed as rendered. A first line that is
    alone over the byte cap is still shown, cut and marked, so a read always progresses."""
    prefix = prefix_bytes(len(lines))
    kept: list[int] = []
    used = 0
    for n in numbers(need):
        if len(kept) >= caps.max_lines:
            return Fit(runs(kept), reason="line_cap")
        cost = prefix + len(lines[n - 1].encode("utf-8")) + 1
        if used + cost > caps.inline_bytes:
            if kept:
                return Fit(runs(kept), reason="byte_cap")
            return Fit((Span(n, n),), cut=n, reason="byte_cap")
        kept.append(n)
        used += cost
    return Fit(runs(kept))


def truncation_of(
    wanted: Sequence[Span],
    shown: Sequence[Span],
    handle: Handle,
    total: int,
    reason: Literal["byte_cap", "line_cap"] | None,
    caps: Caps,
) -> Truncation | None:
    """D78's value: absent when the request was met, which is how a caller tells *the file
    ended* from *the output was cut*."""
    missing = minus(wanted, shown)
    if not missing:
        return None
    # What was asked for and sent, span by span: the lines between two of them were asked
    # for and were NOT sent, and one range across the gap would claim them.
    why = reason or "byte_cap"
    return Truncation(
        minus(wanted, missing), line_count(missing), total, why, handle, cap_of(caps, why)
    )


def cap_of(caps: Caps, reason: Literal["byte_cap", "line_cap"]) -> int:
    """The cap that cut, in its own unit, for the cut note to name."""
    return caps.inline_bytes if reason == "byte_cap" else caps.max_lines
