"""`edits/match.resolve`: the one text matcher. The Applier's
own behaviour through it is `tests/applier/test_apply_search_replace.py`'s; this holds what
the matcher answers on its own, which the worker's draft will read too."""

from __future__ import annotations

from typing import cast

from chickadee.edits.hunk import Hunk, MatchReason
from chickadee.edits.match import (
    NEAREST_LINES,
    MatchFailure,
    Region,
    Replacement,
    find,
    resolve,
    splice,
)


def _failed(reason: MatchReason, detail: str) -> MatchFailure:
    """The failure expected, by its cause: `do` is not part of a failure's equality."""
    return MatchFailure(reason, detail, do="")


def _h(*pairs: tuple[str, str]) -> tuple[Hunk, ...]:
    return tuple(Hunk(old=old, new=new) for old, new in pairs)


def test_hunks_apply_in_order_each_on_the_result_before_it() -> None:
    assert resolve("a\n", _h(("a", "b"), ("b\n", "c\n"))) == "c\n"


def test_a_failure_names_the_hunk_and_never_the_path() -> None:
    assert resolve("a\na\n", _h(("x", "y"))) == _failed(
        MatchReason.NO_MATCH, "hunk 0: old not found"
    )
    assert resolve("a\nb\n", _h(("a\n", "x\n"), ("b", "y"), ("a", "q"))) == _failed(
        MatchReason.NO_MATCH, "hunk 2: old not found"
    )
    assert resolve("a\na\n", _h(("a", "y"))) == _failed(
        MatchReason.AMBIGUOUS, "hunk 0: old occurs 2 times"
    )
    assert resolve("a\n", _h(("", "y"))) == _failed(
        MatchReason.AMBIGUOUS, "hunk 0: empty old matches everywhere"
    )


def test_a_uniformly_crlf_text_is_matched_as_lf_either_way_and_kept_crlf() -> None:
    crlf = "def go():\r\n    return 1\r\n"
    lf = _h(("def go():\n    return 1", "def go():\n    return 2"))
    own = _h(("def go():\r\n    return 1", "def go():\r\n    return 2"))
    assert resolve(crlf, lf) == resolve(crlf, own) == "def go():\r\n    return 2\r\n"


def test_whitespace_and_indentation_are_never_tolerated() -> None:
    assert isinstance(resolve("    x = 1\n", _h(("  x = 1", "y"))), str)
    assert resolve("    x = 1\n", _h(("x  = 1", "y"))) == _failed(
        MatchReason.NO_MATCH, "hunk 0: old not found; nearest lines:\n1│    x = 1"
    )


SOURCE = (
    "def fetch_order(order_id: str, retries: int = 3) -> dict:\n"
    "    resp = None\n"
    "    for attempt in range(retries):\n"
    "        resp = get(order_id)\n"
)


def test_a_no_match_names_the_nearest_line_as_the_file_holds_it() -> None:
    """A stray space in a copied anchor finds its line, shown with its numbers."""
    bad = _h(("def fetch_order(order_id: str, retries: int = 3) -> dict :", "x"))
    assert resolve(SOURCE, bad) == _failed(
        MatchReason.NO_MATCH,
        "hunk 0: old not found; nearest lines:\n"
        "1│def fetch_order(order_id: str, retries: int = 3) -> dict:",
    )


def test_a_wrong_indentation_still_finds_its_line() -> None:
    bad = _h(("  for attempt in range(retries):\n  resp = get(order_id)", "x"))
    failure = resolve(SOURCE, bad)
    assert isinstance(failure, MatchFailure)
    assert failure.detail.endswith("nearest lines:\n3│    for attempt in range(retries):")


def test_the_nearest_lines_are_best_first_and_at_most_three() -> None:
    text = "value = 10\nvalue = 11\nvalue = 12\nvalue = 13\nvalue = 1\n"
    failure = resolve(text, _h(("value = 1;", "y")))
    assert isinstance(failure, MatchFailure)
    rows = failure.detail.split("\n")[1:]
    assert len(rows) == NEAREST_LINES
    assert rows == ["5│value = 1", "1│value = 10", "2│value = 11"]


def test_a_line_below_the_ratio_is_not_named() -> None:
    """A starting guess (`NEAREST_RATIO`, 0.6): an anchor with a trailing comment scores 0.56
    against its line, and is not named. E1 measures whether the guess is right."""
    assert resolve("x = 1\n", _h(("x = 1  # note", "y"))) == _failed(
        MatchReason.NO_MATCH, "hunk 0: old not found"
    )


def test_nothing_alike_names_no_line() -> None:
    assert resolve(SOURCE, _h(("class Unrelated(Base):", "x"))) == _failed(
        MatchReason.NO_MATCH, "hunk 0: old not found"
    )


def test_the_numbers_are_the_text_the_hunk_saw() -> None:
    """Hunk 1 sees the text after hunk 0, which added a line above."""
    failure = resolve("a = 1\nb = 2\n", _h(("a = 1", "z = 0\na = 1"), ("b = 3", "y")))
    assert isinstance(failure, MatchFailure)
    assert failure.detail == "hunk 1: old not found; nearest lines:\n3│b = 2\n1│z = 0\n2│a = 1"


def test_a_crlf_file_s_nearest_lines_are_shown_decoded() -> None:
    failure = resolve("a = 1\r\nb = 2\r\n", _h(("b = 3", "y")))
    assert isinstance(failure, MatchFailure)
    assert failure.detail == "hunk 0: old not found; nearest lines:\n2│b = 2\n1│a = 1"


def test_the_key_is_the_anchor_s_most_distinctive_line() -> None:
    """Review: keyed on the first line, an anchor opening with `}` named three
    unrelated `}` lines. The line with the most letters and digits names where the text is."""
    text = "def a():\n    }\n}\ndef b():\n    }\n    return compute(value)\n"
    failure = resolve(text, _h(("    }\n    return compute(value)  # changed", "x")))
    assert failure == _failed(
        MatchReason.NO_MATCH,
        "hunk 0: old not found; nearest lines:\n6│    return compute(value)",
    )


def test_a_long_nearest_line_is_clipped_like_any_row() -> None:
    """Review: a minified line went into the retry digest whole."""
    text = "x = " + "a" * 5000 + "\n"
    failure = resolve(text, _h(("x = " + "a" * 4990 + "b", "y")))
    assert isinstance(failure, MatchFailure)
    assert failure.detail.endswith("…[+4,764 chars]")
    assert len(failure.detail) < 300


# -- whole lines ------------------------------------------------------------------------


def _lines(old: str, new: str) -> Hunk:
    return Hunk(old=old, new=new, lines=True)


def test_a_lines_hunk_never_matches_part_of_a_line() -> None:
    """As text, `x = 1` edits `max = 1` and `return 1` edits `return 10`."""
    assert resolve("max = 1\nx = 1\n", [_lines("x = 1\n", "x = 2\n")]) == "max = 1\nx = 2\n"
    assert resolve("max = 1\n", [_lines("x = 1\n", "x = 2\n")]) == _failed(
        MatchReason.NO_MATCH, "hunk 0: old not found; nearest lines:\n1│max = 1"
    )
    assert isinstance(resolve("    return 10\n", [_lines("    return 1\n", "")]), MatchFailure)
    # Text matching, Zeus's, is unchanged.
    assert resolve("max = 1\n", [Hunk(old="x = 1", new="x = 2")]) == "max = 2\n"


def test_the_last_line_matches_without_a_final_newline_and_the_state_is_kept() -> None:
    assert (
        resolve("a = 1\nb = 2", [_lines("b = 2\n", "b = 3\nc = 4\n")]) == "a = 1\nb = 3\nc = 4"
    )
    assert resolve("a = 1\nb = 2", [_lines("b = 2\n", "")]) == "a = 1"
    assert resolve("a = 1\nb = 2\n", [_lines("b = 2\n", "b = 3\n")]) == "a = 1\nb = 3\n"


def test_removed_lines_leave_nothing_and_endings_follow_the_run() -> None:
    assert resolve("a\nb\nc\n", [_lines("b\n", "")]) == "a\nc\n"
    assert resolve("a\r\nb\r\n", [_lines("a\n", "A\nA2\n")]) == "A\r\nA2\r\nb\r\n"
    # A mixed file is matched as it is; the lines around keep their own endings.
    assert resolve("a\r\nb\nc\n", [_lines("a\n", "A\n")]) == "A\r\nb\nc\n"


def test_lines_hunks_count_whole_runs_and_an_empty_old_is_ambiguous() -> None:
    assert resolve("x\nx\n", [_lines("x\n", "y\n")]) == _failed(
        MatchReason.AMBIGUOUS, "hunk 0: old occurs 2 times"
    )
    assert resolve("x\n", [_lines("", "y\n")]) == _failed(
        MatchReason.AMBIGUOUS, "hunk 0: empty old matches everywhere"
    )


# -- `find`: the region form, the draft's modes ----------------------


def test_find_answers_the_span_and_its_replacement_which_resolve_writes() -> None:
    text = "a = 1\nb = 2\n"
    found = find(text, "b = 2", "b = 3", lines=False)
    assert found == (Replacement(Region(6, 11), "b = 3"),)
    (only,) = cast(tuple[Replacement, ...], found)
    assert splice(text, only.region, only.content) == "a = 1\nb = 3\n"


def test_find_fails_as_resolve_does_without_naming_a_hunk() -> None:
    assert find("a\na\n", "a", "b", lines=False) == _failed(
        MatchReason.AMBIGUOUS, "old occurs 2 times"
    )
    assert find("a\n", "", "b", lines=False) == _failed(
        MatchReason.AMBIGUOUS, "empty old matches everywhere"
    )
    missing = find("a\n", "zzz", "b", lines=False)
    assert isinstance(missing, MatchFailure) and missing.reason is MatchReason.NO_MATCH


def test_replace_all_takes_every_occurrence_non_overlapping() -> None:
    text = "x = a + a\ny = aa\n"
    found = find(text, "a", "b", lines=False, replace_all=True)
    assert isinstance(found, tuple) and len(found) == 4
    out = text
    for one in reversed(found):
        out = splice(out, one.region, one.content)
    assert out == "x = b + b\ny = bb\n"


def test_lines_keep_the_runs_ending_and_the_files_final_newline() -> None:
    text = "a\nb\nc"
    (only,) = cast(tuple[Replacement, ...], find(text, "c\n", "d\n", lines=True))
    assert splice(text, only.region, only.content) == "a\nb\nd"
    (gone,) = cast(tuple[Replacement, ...], find(text, "c\n", "", lines=True))
    assert splice(text, gone.region, gone.content) == "a\nb"


def test_replace_all_removing_adjacent_runs_to_the_end_keeps_no_final_newline() -> None:
    text = "a\nx\nx"
    found = find(text, "x\n", "", lines=True, replace_all=True)
    assert isinstance(found, tuple)
    out = text
    for one in reversed(found):
        out = splice(out, one.region, one.content)
    assert out == "a"


SCOPED = (
    "class A:\n"
    "    def go(self):\n"
    "        return 1\n"
    "\n"
    "class B:\n"
    "    def go(self):\n"
    "        return 1\n"
)


def test_a_scope_narrows_the_match_to_its_block() -> None:
    assert isinstance(find(SCOPED, "return 1", "return 2", lines=False), MatchFailure)
    (only,) = cast(
        tuple[Replacement, ...],
        find(SCOPED, "return 1", "return 2", lines=False, scopes=("class B:",)),
    )
    assert splice(SCOPED, only.region, only.content).endswith("        return 2\n")
    assert splice(SCOPED, only.region, only.content).count("return 1") == 1


def test_scopes_nest_and_a_missing_or_repeated_one_fails_naming_it() -> None:
    nested = find(
        SCOPED, "return 1", "return 2", lines=False, scopes=("class A:", "def go(self):")
    )
    assert isinstance(nested, tuple) and len(nested) == 1
    assert find(SCOPED, "return 1", "x", lines=False, scopes=("class C:",)) == _failed(
        MatchReason.NO_MATCH, "scope 'class C:' not found"
    )
    assert find(SCOPED, "return 1", "x", lines=False, scopes=("def go(self):",)) == _failed(
        MatchReason.AMBIGUOUS, "scope 'def go(self):' occurs 2 times"
    )


def test_a_brace_blocks_closing_line_is_inside_its_scope() -> None:
    text = "func a() {\n  x()\n}\nfunc b() {\n  x()\n}\n"
    found = find(text, "  x()\n}\n", "  y()\n}\n", lines=True, scopes=("func b() {",))
    assert isinstance(found, tuple) and len(found) == 1
    assert splice(text, found[0].region, found[0].content).endswith("func b() {\n  y()\n}\n")


# -- the review's pins ---------------------------------------------------------------


def test_two_runs_reaching_an_unterminated_end_are_still_ambiguous() -> None:
    """The occurrences are counted before any replacement joins them: the Applier never
    replaces both of two `}` lines because the second ends the file."""
    assert resolve("x\n}\n}", (Hunk(old="}", new="} // end", lines=True),)) == _failed(
        MatchReason.AMBIGUOUS, "hunk 0: old occurs 2 times"
    )
    assert isinstance(find("x\n}\n}", "}", "y", lines=True), MatchFailure)


def test_a_lines_replacement_ending_in_a_lone_cr_keeps_it() -> None:
    assert resolve("a", (Hunk(old="a", new="\r\r", lines=True),)) == "\r"


def test_find_reads_a_crlf_anchor_in_a_crlf_files_decoded_text() -> None:
    """A lines anchor drops a CR before its newline anyway; literal text needs `crlf`."""
    text = "a = 1\nb = 2\n"
    found = find(text, "a = 1\r\n", "a = 3\r\n", lines=False, crlf=True)
    assert found == (Replacement(Region(0, 6), "a = 3\n"),)
    assert isinstance(find(text, "a = 1\r\n", "x", lines=False), MatchFailure)


def test_a_scope_goes_on_through_a_multi_line_signature() -> None:
    text = "def go(\n    a,\n    b,\n) -> int:\n    return a\n\n\ndef stop():\n    return a\n"
    found = find(text, "    return a\n", "    return b\n", lines=True, scopes=("def go(",))
    assert found == (Replacement(Region(32, 45), "    return b\n"),)


def test_an_old_that_matches_only_with_its_escapes_written_back_is_told_so() -> None:
    """`print("a\\n")` written in a Python string sends a newline where the
    file has a backslash and an n. The hint names it; the match stays exact."""
    text = 'def f():\n    print("a\\n")\n'
    failure = find(text, 'print("a\n")', 'print("b\\n")', lines=False)
    assert isinstance(failure, MatchFailure) and failure.reason is MatchReason.NO_MATCH
    assert (
        "occurs once with some of its newlines or tabs written as backslash" in failure.detail
    )
    assert 'r"""..."""' in failure.do
    assert isinstance(find(text, 'print("a\\n")', 'print("b\\n")', lines=False), tuple)


def test_an_old_that_matches_nowhere_gets_no_escape_hint() -> None:
    failure = find("x = 1\n", "y = 2\n", "z\n", lines=False)
    assert isinstance(failure, MatchFailure) and "escapes" not in failure.detail


def test_one_escaped_newline_among_real_ones_is_told_so() -> None:
    """The usual case: `old` spans real lines, and one holds `\\n` inside a string."""
    text = 'def f():\n    print("a\\n")\n    x = 1\n'
    failure = find(text, 'print("a\n")\n    x = 1', "y", lines=False)
    assert isinstance(failure, MatchFailure)
    assert "written as backslash escapes" in failure.detail
