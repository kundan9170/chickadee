"""`edits/lines`: one splitter, one newline policy, one line renderer."""

from __future__ import annotations

from chickadee.context.spans import Span
from chickadee.edits.lines import (
    BAR,
    BAR_BYTES,
    ROW_CHARS_MAX,
    clip,
    decode,
    encode,
    number_lines,
    parse_row,
    prefix_bytes,
    row,
    split_lines,
)


def test_split_lines_splits_only_on_newline() -> None:
    assert split_lines("") == ()
    assert split_lines("a\nb") == ("a", "b")
    assert split_lines("a\r\nb\r\n") == ("a", "b")
    assert split_lines("a\x0cb\x0bc\N{LINE SEPARATOR}d\n") == (
        "a\x0cb\x0bc\N{LINE SEPARATOR}d",
    )


def test_split_lines_keeping_ends_gives_the_same_lines_joined_back_exactly() -> None:
    for text in (
        "",
        "a",
        "a\n",
        "\n",
        "a\nb",
        "a\r\nb\r\n",
        "\n\nx",
        "1\r2\x0c3\n4\N{LINE SEPARATOR}",
    ):
        kept = split_lines(text, keepends=True)
        assert "".join(kept) == text
        assert len(kept) == len(split_lines(text))
    assert split_lines("a\r\nb\rc\n\n", keepends=True) == ("a\r\n", "b\rc\n", "\n")


def test_a_uniformly_crlf_text_decodes_to_lf_and_encodes_back() -> None:
    decoded = decode("a\r\nb\r\nc")
    assert (decoded.text, decoded.crlf) == ("a\nb\nc", True)
    assert encode("a\nB\nc\n", decoded) == "a\r\nB\r\nc\r\n"


def test_a_carriage_return_before_a_crlf_survives_the_round_trip() -> None:
    """Review: `encode` once collapsed `\r\n` to `\n` first, and a line ending
    `\r\r\n` came back as `\r\n`, a byte lost."""
    for text in ("a\r\r\nb\r\n", "x\r\n", "a\rb\r\n"):
        decoded = decode(text)
        assert decoded.crlf, repr(text)
        assert encode(decoded.text, decoded) == text, repr(text)


def test_any_other_text_is_kept_as_it_is() -> None:
    for text in ("a\nb\n", "a\r\nb\n", "a\rb", "no newline", ""):
        decoded = decode(text)
        assert (decoded.text, decoded.crlf) == (text, False), repr(text)
        assert encode(text, decoded) == text


def test_a_lone_carriage_return_inside_a_crlf_file_is_kept() -> None:
    decoded = decode("a\rb\r\nc\r\n")
    assert decoded.text == "a\rb\nc\n"
    assert encode(decoded.text, decoded) == "a\rb\r\nc\r\n"


def test_a_row_is_built_measured_and_read_back_in_one_place() -> None:
    assert row(7, 3, "x = 1") == f"  7{BAR}x = 1"
    assert prefix_bytes(120) == 3 + BAR_BYTES
    assert parse_row(row(7, 3, "x = 1")) == (7, "x = 1")
    assert parse_row("[elided: lines 3-4]") is None


def test_number_lines_round_trips_through_parse_row() -> None:
    lines = ("a", "b", "c")
    rows = number_lines(lines, [Span(1, 1), Span(3, 3)])
    parsed = [parse_row(r) for r in rows]
    assert parsed == [(1, "a"), None, (3, "c")]


def test_clip_cuts_a_long_line_and_counts_the_rest() -> None:
    assert clip("short") == "short"
    assert clip("x" * (ROW_CHARS_MAX + 5)) == "x" * ROW_CHARS_MAX + "…[+5 chars]"
    assert clip("abcdef", 3) == "abc…[+3 chars]"
