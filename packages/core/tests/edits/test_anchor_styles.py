"""Line anchor styles: the one tag, each style's rows, its argument schema,
and an anchor resolved to a line or to `Stale` with the lines there now."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from chickadee.context.spans import Span
from chickadee.edits.lines import (
    ANCHOR_STYLES,
    HASH,
    NUMBER,
    AnchorKind,
    AnchorStyle,
    Stale,
    hit,
    parse_row,
    tag,
)
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.validate import InvalidArguments, validate

LINES = ("def go() -> int:", "    x = 1", "    return x")


def _spec(style: AnchorStyle, kind: AnchorKind) -> ToolSpec:
    return dataclasses.replace(
        SURFACE[0],
        schema={
            "type": "object",
            "properties": {"at": style.argument_schema(kind)},
            "required": ["at"],
            "additionalProperties": False,
        },
    )


def _valid(style: AnchorStyle, kind: AnchorKind, value: object) -> bool:
    return not isinstance(validate(_spec(style, kind), {"at": value}), InvalidArguments)


# -- the tag ---------------------------------------------------------------------------------


def test_a_tag_is_the_first_three_hex_digits_of_the_lines_sha256() -> None:
    assert tag("x = 1") == hashlib.sha256(b"x = 1").hexdigest()[:3]
    assert len(tag("")) == 3


def test_a_tag_leaves_the_line_ending_out() -> None:
    assert tag("x = 1\n") == tag("x = 1\r\n") == tag("x = 1")
    assert tag("x = 1 ") != tag("x = 1")


# -- rows ------------------------------------------------------------------------------------


def test_the_styles_are_named_and_number_is_todays_rows() -> None:
    assert ANCHOR_STYLES == {"number": NUMBER, "hash": HASH}
    assert NUMBER.rows(LINES) == ["1│def go() -> int:", "2│    x = 1", "3│    return x"]


def test_hash_rows_carry_each_lines_tag_and_parse_back() -> None:
    rows = HASH.rows(LINES)
    assert rows[1] == f"2:{tag(LINES[1])}│    x = 1"
    assert [parse_row(r) for r in rows] == [(n, t) for n, t in enumerate(LINES, start=1)]


def test_rows_of_spans_mark_the_gap() -> None:
    rows = HASH.rows(LINES, (Span(1, 1), Span(3, 3)))
    assert rows == [f"1:{tag(LINES[0])}│{LINES[0]}", "[elided: line 2]", HASH.rows(LINES)[2]]


def test_an_anchor_and_a_hit_in_each_style() -> None:
    assert NUMBER.anchor(2, LINES[1]) == "2"
    assert HASH.anchor(2, LINES[1]) == f"2:{tag(LINES[1])}"
    assert hit("app.py", HASH.anchor(2, LINES[1]), LINES[1]) == (
        f"app.py:2:{tag(LINES[1])}:    x = 1"
    )
    assert NUMBER.example(12) == "12" and HASH.example(12) == "12:a3f"


# -- the schema: a mismatched anchor fails it, never reinterpreted --------------------------


@pytest.mark.parametrize(
    ("style", "kind", "value", "valid"),
    [
        (NUMBER, "line", 12, True),
        (NUMBER, "line", 0, False),
        (NUMBER, "line", "12:a3f", False),  # a tag under `number`
        (NUMBER, "line", "12", False),
        (NUMBER, "after", 0, True),
        (NUMBER, "after", -1, False),
        (HASH, "line", "12:a3f", True),
        (HASH, "line", 12, False),  # a bare number under `hash`
        (HASH, "line", "12", False),
        (HASH, "line", "12:A3F", False),
        (HASH, "line", "12:a3f9", False),
        (HASH, "line", 0, False),
        (HASH, "after", 0, True),
        (HASH, "after", "0", True),  # as a `TOOL` line carries it
        (HASH, "after", "12:a3f", True),
        (HASH, "after", 12, False),
        (HASH, "after", "12", False),
    ],
)
def test_an_anchor_fits_its_styles_schema_only(
    style: AnchorStyle, kind: AnchorKind, value: object, valid: bool
) -> None:
    assert _valid(style, kind, value) is valid


def test_a_refused_hash_anchor_names_the_shapes_it_may_take() -> None:
    refused = validate(_spec(HASH, "after"), {"at": 12})
    assert isinstance(refused, InvalidArguments)
    problem = refused.violations[0].problem
    assert "0" in problem and "string matching" in problem and "got 12" in problem


# -- resolve: the line, or Stale with the lines there now ----------------------------------


def test_an_anchor_resolves_to_its_line() -> None:
    assert NUMBER.resolve(2, LINES, "line") == 2
    assert HASH.resolve(HASH.anchor(3, LINES[2]), LINES, "line") == 3
    assert NUMBER.resolve(0, LINES, "after") == 0
    assert HASH.resolve("0", LINES, "after") == 0
    assert HASH.resolve(0, LINES, "after") == 0


def test_a_moved_line_is_stale_with_the_lines_there_now() -> None:
    anchor = HASH.anchor(2, LINES[1])
    moved = (LINES[0], "    # a comment", *LINES[1:])
    stale = HASH.resolve(anchor, moved, "line")
    assert isinstance(stale, Stale)
    assert f"line 2 no longer reads as {anchor}" in stale.reason
    assert f"2:{tag(moved[1])}│    # a comment" in stale.reason
    assert f"4:{tag(moved[3])}│    return x" in stale.reason


def test_a_line_past_the_end_is_stale_in_either_style() -> None:
    stale = NUMBER.resolve(9, LINES, "line")
    assert isinstance(stale, Stale)
    assert "the file has 3 lines" in stale.reason and "3│    return x" in stale.reason
    empty = HASH.resolve("1:abc", (), "line")
    assert isinstance(empty, Stale) and "the file is empty" in empty.reason


def test_a_number_anchor_never_checks_a_tag() -> None:
    assert NUMBER.resolve(2, ("anything",) * 3, "line") == 2
