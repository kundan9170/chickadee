"""Line spans, and the arithmetic every reading module does on them.

`Span` is 1-indexed and inclusive. `SpanWire` is its form in a session-log line. The
rest is what a view's lines are built from: the Reader merges, subtracts and counts spans
(`coalesce`, `minus`, `line_count`), and an outline turns line numbers back into them (`runs`).
Imports nothing from `chickadee.context`, so every module can.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import PlainSerializer, PlainValidator, TypeAdapter
from pydantic.types import StrictInt

__all__ = [
    "Span",
    "SpanWire",
    "coalesce",
    "holds",
    "line_count",
    "minus",
    "numbers",
    "runs",
    "spans_text",
]


@dataclass(frozen=True, order=True)
class Span:
    """1-indexed and inclusive, in lines unless said otherwise."""

    start: int
    end: int
    unit: Literal["line", "byte"] = "line"

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"a span starts at 1 or later, not {self.start}")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes its start {self.start}")

    @staticmethod
    def parse(text: str) -> Span:
        """The `read` tool's `lines` form, `start-end` — the one its schema admits."""
        head, sep, tail = text.partition("-")
        if not sep or not head.isdigit() or not tail.isdigit():
            raise ValueError(f"{text!r} is not a span of the form start-end")
        return Span(int(head), int(tail))

    def __len__(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"


_PAIR: Final = TypeAdapter(tuple[StrictInt, StrictInt])


def _span(value: object) -> Span:
    """A span as a line holds it, `[start, end]`: two integers, never bools or floats."""
    if isinstance(value, Span):
        return value
    start, end = _PAIR.validate_python(value)
    return Span(start, end)


def _pair(span: Span) -> list[int]:
    return [span.start, span.end]


SpanWire = Annotated[Span, PlainValidator(_span), PlainSerializer(_pair, return_type=list[int])]
"""`Span` as a session-log line holds it: `[start, end]`, a line span. The one place
its wire form is declared; `context.lines` and `Handle` use it."""


# -- arithmetic -------------------------------------------------------------------------


def coalesce(spans: Iterable[Span], gap: int = 0) -> tuple[Span, ...]:
    """Spans in line order, merged where they overlap, touch, or leave a gap of at most `gap`
    lines — a bridged gap is then shown rather than marked."""
    out: list[Span] = []
    for s in sorted(spans):
        if out and s.start <= out[-1].end + 1 + gap:
            if s.end > out[-1].end:
                out[-1] = Span(out[-1].start, s.end)
        else:
            out.append(s)
    return tuple(out)


def minus(spans: Iterable[Span], taken: Iterable[Span]) -> tuple[Span, ...]:
    """The lines of `spans` that `taken` does not hold."""
    holes = coalesce(taken)
    out: list[Span] = []
    for s in coalesce(spans):
        start = s.start
        for hole in holes:
            if hole.end < start or hole.start > s.end:
                continue
            if hole.start > start:
                out.append(Span(start, hole.start - 1))
            start = hole.end + 1
            if start > s.end:
                break
        if start <= s.end:
            out.append(Span(start, s.end))
    return tuple(out)


def line_count(spans: Iterable[Span]) -> int:
    return sum(len(s) for s in spans)


def spans_text(spans: Iterable[Span]) -> str:
    return ", ".join(str(s) for s in spans)


def numbers(spans: Iterable[Span]) -> Iterator[int]:
    for s in spans:
        yield from range(s.start, s.end + 1)


def holds(spans: Iterable[Span], line: int) -> bool:
    return any(s.start <= line <= s.end for s in spans)


def runs(numbers: Iterable[int]) -> tuple[Span, ...]:
    """Ascending line numbers as the spans they form."""
    spans: list[Span] = []
    for n in numbers:
        if spans and n == spans[-1].end + 1:
            spans[-1] = Span(spans[-1].start, n)
        else:
            spans.append(Span(n, n))
    return tuple(spans)
