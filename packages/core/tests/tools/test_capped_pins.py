"""Pins on `tools/catalogue.capped`, the one place a tool's text answer is cut to its
bound: its answers as they were before it split lines with `edits.lines` rather than
`str.splitlines`, the two that moved marked with their old answers.

A line-bounded answer is a list of search hits, one per line; a byte-bounded one is a file's
or a process's text, cut at a whole line. Its handle's rest is read back numbered by
`split_lines`, so the `Truncation`'s counts must be lines as that numbers them.
"""

from __future__ import annotations

from chickadee.context.results import Truncation
from chickadee.context.spans import Span
from chickadee.tools.catalogue import capped

from .tools_helpers import make_handle

HANDLE = make_handle("findings", "x", b"x")


def _cut(text: str, limit: int, unit: str) -> tuple[str, tuple[object, ...] | None]:
    assert unit in ("bytes", "lines")
    content, t = capped(text, HANDLE, limit=limit, unit=unit)  # pyright: ignore[reportArgumentType]
    return content, _counts(t)


def _counts(t: Truncation | None) -> tuple[object, ...] | None:
    if t is None:
        return None
    assert t.continue_with == HANDLE
    return (t.kept, t.dropped, t.total, t.reason)


HITS = "a.py:1:x\nb.py:2:y\nc.py:3:z\n"


def test_hits_within_the_bound_are_untouched() -> None:
    assert _cut(HITS, 3, "lines") == (HITS, None)


def test_hits_over_the_bound_keep_the_first() -> None:
    assert _cut(HITS, 2, "lines") == (
        "a.py:1:x\nb.py:2:y\n",
        ((Span(1, 2),), 1, 3, "result_cap"),
    )


def test_a_zero_line_bound_still_shows_one() -> None:
    assert _cut(HITS, 0, "lines") == ("a.py:1:x\n", ((Span(1, 1),), 2, 3, "result_cap"))


def test_text_within_the_byte_bound_is_untouched() -> None:
    assert _cut("ab\ncd", 5, "bytes") == ("ab\ncd", None)


def test_text_over_the_byte_bound_keeps_whole_lines() -> None:
    assert _cut("ab\ncd\nef", 6, "bytes") == ("ab\ncd\n", ((Span(1, 2),), 1, 3, "byte_cap"))


def test_one_line_over_the_byte_bound_is_cut_and_marked() -> None:
    assert _cut("abcdef\ngh\n", 3, "bytes") == (
        "abc\n[line 1 is cut at 3 bytes]",
        ((Span(1, 1),), 1, 2, "byte_cap"),
    )


def test_crlf_lines_keep_their_endings() -> None:
    assert _cut("ab\r\ncd\r\nef\r\n", 8, "bytes") == (
        "ab\r\ncd\r\n",
        ((Span(1, 2),), 1, 3, "byte_cap"),
    )


def test_a_hit_holding_a_form_feed_is_one_hit() -> None:
    """Moved by T-3.137. Before: `splitlines` split the hit at the form feed, so three
    hits counted four and the cut kept half of the second: `("a.py:1:x\\nb.py:2:y\\x0c",
    ((Span(1, 2),), 2, 4, "result_cap"))`."""
    hits = "a.py:1:x\nb.py:2:y\x0cz\nc.py:3:w\n"
    assert _cut(hits, 2, "lines") == (
        "a.py:1:x\nb.py:2:y\x0cz\n",
        ((Span(1, 2),), 1, 3, "result_cap"),
    )


def test_a_lone_cr_does_not_end_a_line() -> None:
    """Moved by T-3.137. Before: a progress bar's lone `\\r` ended a line, so the same
    text was counted as lines its handle's reader does not number: `((Span(1, 3),), 1, 4,
    "byte_cap")`, and `continue_with` pointed past the file's last line."""
    assert _cut("10%\r20%\r30%\ndone\n", 13, "bytes") == (
        "10%\r20%\r30%\n",
        ((Span(1, 1),), 1, 2, "byte_cap"),
    )
