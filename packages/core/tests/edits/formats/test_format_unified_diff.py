"""`unified_diff`: each `@@` hunk to one exact-text hunk, its numbers ignored;
`--- /dev/null` creates; a delete refused."""

from __future__ import annotations

from formats_helpers import FILE, C, E, applied, create, edits, hunks, parse, targets

from chickadee.edits.formats.base import FileEdits
from chickadee.schema.adapter import ParseError

DIFF = (
    "--- a/app.py\n+++ b/app.py\n@@ -99,3 +99,3 @@\n"
    " def go() -> int:\n-    return 1\n+    return 2\n"
)


def test_a_fenced_diff_is_one_hunk_of_context_and_changes_and_applies() -> None:
    result = parse("unified_diff", f"Here:\n```diff\n{DIFF}```\nDone.")
    assert result == edits(
        hunks(
            "app.py",
            ("def go() -> int:\n    return 1\n", "def go() -> int:\n    return 2\n"),
        )
    )
    assert applied(result) == FILE.replace("return 1", "return 2")


def test_a_bare_diff_parses_the_same_and_its_numbers_are_ignored() -> None:
    assert parse("unified_diff", DIFF) == parse("unified_diff", f"```diff\n{DIFF}```")


def test_hunks_and_files_in_order_and_blank_context_kept_inside_only() -> None:
    given = targets(("a.py", E), ("b.py", E))
    reply = (
        "diff --git a/b.py b/b.py\nindex 1..2 100644\n--- b.py\t2026-01-01\n+++ b.py\n"
        "@@ -1 +1 @@\n-b\n+B\n"
        "--- a/a.py\n+++ b/a.py\n@@ @@\n x\n\n-y\n+Y\n\\ No newline at end of file\n\n"
        "@@ @@\n z\n-w\n+W\n"
    )
    # The packet rules ordered the set by its targets (a.py, b.py); the
    # parser keeps the reply's order.
    assert parse("unified_diff", reply, given) == edits(
        hunks("b.py", ("b\n", "B\n")),
        hunks("a.py", ("x\n\ny\n", "x\n\nY\n"), ("z\nw\n", "z\nW\n")),
    )


def test_headerless_hunks_are_the_one_targets() -> None:
    assert parse("unified_diff", "@@ -1 +1 @@\n-    return 1\n+    return 2\n") == edits(
        hunks("app.py", ("    return 1\n", "    return 2\n"))
    )


def test_a_diff_from_dev_null_creates_a_file() -> None:
    reply = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+X = 1\n+Y = 2\n"
    assert parse("unified_diff", reply, targets(("new.py", C))) == edits(
        create("new.py", "X = 1\nY = 2\n")
    )


def test_a_pure_insertion_without_context_is_an_empty_old_the_matcher_refuses() -> None:
    assert parse("unified_diff", "@@ -1,0 +2 @@\n+x = 1\n") == edits(
        hunks("app.py", ("", "x = 1\n"))
    )


def test_refusals() -> None:
    def hint(reply: str) -> str:
        result = parse("unified_diff", reply, targets(("a.py", E), ("b.py", E)))
        assert isinstance(result, ParseError) and result.kind == "bad_format", result
        return result.hint

    assert hint("--- a/a.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n") == (
        "a.py: a diff may not delete a file; deletes are the planner's"
    )
    assert hint("@@ -1 +1 @@\n-x\n+y\n") == (
        "hunk 0: name its file with --- and +++ lines before its @@ line"
    )


def test_no_diff_is_nothing_written() -> None:
    """The one-shot adapter read an empty parse as a tool request
    (`ToolRequest(symbol)`) or as `no_block`; the parser's own answer is that the reply wrote
    nothing, and that adapter retired."""
    assert parse("unified_diff", "TOOL symbol symbol=go") == FileEdits()
    assert parse("unified_diff", "no idea") == FileEdits()


def test_a_changed_sql_comment_is_a_hunk_line_not_a_header() -> None:
    """Review: `--- old`/`+++ new` inside a hunk read as a new file's header, and the
    change was silently lost. A header is followed by an `@@` line."""
    reply = (
        "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n SELECT 1;\n--- old note\n+++ new note\n"
    )
    assert applied(parse("unified_diff", reply), "SELECT 1;\n-- old note\n") == (
        "SELECT 1;\n++ new note\n"
    )


def test_a_rename_and_a_hunk_that_changes_nothing_are_refused() -> None:
    for reply, hint in (
        (
            "--- a/old.py\n+++ b/app.py\n@@ -1 +1 @@\n-x\n+y\n",
            "old.py: a diff may not move a file to app.py; moves are the planner's",
        ),
        ("@@ -1 +1 @@\n x\n", "hunk 0: it changes nothing; mark lines - or +"),
    ):
        assert parse("unified_diff", reply) == ParseError(kind="bad_format", hint=hint)


def test_a_target_under_b_keeps_its_prefix_and_a_create_keeps_its_blank_lines() -> None:
    given = targets(("b/app.py", E))
    assert parse("unified_diff", "--- b/app.py\n+++ b/app.py\n@@ @@\n-x\n+y\n", given) == (
        edits(hunks("b/app.py", ("x\n", "y\n")))
    )
    reply = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,3 @@\n+x = 1\n\n+y = 2\n"
    assert parse("unified_diff", reply, targets(("new.py", C))) == edits(
        create("new.py", "x = 1\n\ny = 2\n")
    )


def test_no_newline_at_end_of_file_takes_the_newline_off_its_line() -> None:
    reply = "@@ @@\n x\n-y\n\\ No newline at end of file\n+Y\n\\ No newline at end of file\n"
    assert applied(parse("unified_diff", reply), "x\ny") == "x\nY"
