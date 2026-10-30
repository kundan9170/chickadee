"""#559: the worker wrote its target's path as the fence's first body line. Taken whole, the
path became the file's line 1, lint failed, and the identical answer went to rescue. It is
refused before anything is written, with the fix in the hint."""

from __future__ import annotations

import pytest

from chickadee.edits.formats.base import FileEdits, Targets, Whole
from chickadee.edits.formats.whole_file import parse
from chickadee.schema.adapter import ParseError

BODY = "class SegmentTree:\n    pass\n"
ONE = Targets(writable=("segment_tree.py",))


@pytest.mark.parametrize(
    "first", ["segment_tree.py", "path=segment_tree.py", "  segment_tree.py "]
)
def test_a_path_on_the_first_body_line_is_refused_with_the_fix(first: str) -> None:
    got = parse(f"~~~python\n{first}\n{BODY}~~~\n", ONE)
    assert isinstance(got, ParseError) and got.kind == "bad_format"
    assert "put path=segment_tree.py on the opening fence line" in got.hint


def test_the_same_path_on_the_fence_line_is_taken() -> None:
    got = parse(f"~~~python path=segment_tree.py\n{BODY}~~~\n", ONE)
    assert got == FileEdits((("segment_tree.py", Whole(BODY.rstrip("\n"))),))


def test_a_first_line_that_is_not_a_target_is_content() -> None:
    body = "other.py\n" + BODY
    got = parse(f"~~~python\n{body}~~~\n", ONE)
    assert got == FileEdits((("segment_tree.py", Whole(body.rstrip("\n"))),))


def test_with_several_targets_each_fence_is_held_to_it() -> None:
    two = Targets(writable=("a.py", "b.py"))
    got = parse("~~~python path=a.py\nx = 1\n~~~\n~~~python\nb.py\ny = 2\n~~~\n", two)
    assert isinstance(got, ParseError)
    assert "put path=b.py on the opening fence line" in got.hint
