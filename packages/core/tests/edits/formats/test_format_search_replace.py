"""`search_replace`: Aider's blocks to whole-line hunks, an empty SEARCH to a
create, and the refusals of a block its parser cannot read."""

from __future__ import annotations

from formats_helpers import FILE, C, E, Mode, applied, create, edits, hunks, parse, targets

from chickadee.edits.formats.base import FileEdits, Hunks
from chickadee.edits.hunk import MatchReason
from chickadee.edits.match import MatchFailure, resolve
from chickadee.schema.adapter import ParseError

BLOCK = "<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE"


def test_a_block_after_its_path_is_one_hunk_that_applies() -> None:
    result = parse("search_replace", f"app.py\n```python\n{BLOCK}\n```\n")
    assert result == edits(hunks("app.py", ("    return 1\n", "    return 2\n")))
    assert applied(result) == FILE.replace("return 1", "return 2")


def test_the_path_may_be_inside_the_fence_or_its_info_string_or_unnamed() -> None:
    expected = edits(hunks("app.py", ("    return 1\n", "    return 2\n")))
    for reply in (
        f"```python\napp.py\n{BLOCK}\n```",
        f"```python path=app.py\n{BLOCK}\n```",
        f"**`app.py`**\n```\n{BLOCK}\n```",
        BLOCK,  # bare and unnamed: the one target's
    ):
        assert parse("search_replace", reply) == expected, reply


def test_prose_that_looks_like_a_name_does_not_take_the_block() -> None:
    """Review: a bare `README.md`, `v1.2` or `os.path` line used to be read as a path
    and refuse the reply; only a target's path, or a path with a directory, names one, and
    `./` and a heading's `#`s are dropped."""
    expected = edits(hunks("app.py", ("    return 1\n", "    return 2\n")))
    for line in ("See README.md:", "version1.2", "os.path", "./app.py", "### app.py"):
        assert parse("search_replace", f"{line}\n```\n{BLOCK}\n```") == expected, line


def test_a_block_is_whole_lines_so_it_neither_matches_inside_a_line_nor_leaves_one() -> None:
    """Review: `return 1` matched inside `return 10`, and an empty REPLACE left the
    removed lines' newlines behind."""
    inside = parse("search_replace", BLOCK)
    assert isinstance(inside, FileEdits)
    ((_, found),) = inside.edits
    assert isinstance(found, Hunks)
    assert resolve("def go() -> int:\n    return 10\n", found.hunks) == MatchFailure(
        MatchReason.NO_MATCH, "hunk 0: old not found; nearest lines:\n2│    return 10", do=""
    )
    removed = "<<<<<<< SEARCH\n\n\ndef stop() -> None:\n    pass\n=======\n>>>>>>> REPLACE"
    assert applied(parse("search_replace", removed)) == "def go() -> int:\n    return 1\n"


def test_blocks_on_one_path_are_its_hunks_in_order() -> None:
    second = "<<<<<<< SEARCH\n    pass\n=======\n    return None\n>>>>>>> REPLACE"
    result = parse("search_replace", f"app.py\n```\n{BLOCK}\n{second}\n```")
    assert result == edits(
        hunks(
            "app.py", ("    return 1\n", "    return 2\n"), ("    pass\n", "    return None\n")
        )
    )


def test_blocks_on_several_paths_are_each_paths_in_the_replys_order() -> None:
    """The packet rules ordered an atomic set by its targets (a.py, b.py); the
    parser keeps the reply's order, and nothing reorders it since they retired."""
    given = targets(("a.py", E), ("b.py", E))
    reply = (
        "b.py\n```\n<<<<<<< SEARCH\nb\n=======\nB\n>>>>>>> REPLACE\n```\n"
        "a.py\n```\n<<<<<<< SEARCH\na\n=======\nA\n>>>>>>> REPLACE\n```\n"
    )
    assert parse("search_replace", reply, given) == edits(
        hunks("b.py", ("b\n", "B\n")), hunks("a.py", ("a\n", "A\n"))
    )


def test_an_empty_search_creates_a_file() -> None:
    reply = "new.py\n```\n<<<<<<< SEARCH\n=======\nX = 1\n>>>>>>> REPLACE\n```"
    assert parse("search_replace", reply, targets(("new.py", C))) == edits(
        create("new.py", "X = 1\n")
    )


def test_a_block_the_parser_cannot_read_is_refused() -> None:
    def refused(reply: str, *named: tuple[str, Mode]) -> str:
        result = parse("search_replace", reply, targets(*named) if named else None)
        assert isinstance(result, ParseError) and result.kind == "bad_format", result
        return result.hint

    create = "<<<<<<< SEARCH\n=======\nX = 1\n>>>>>>> REPLACE"
    assert refused(BLOCK, ("a.py", E), ("b.py", E)) == (
        "block 0: name its TARGETS path on the line before its fence"
    )
    assert refused("<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n") == (
        "block 0: <<<<<<< SEARCH needs a ======= line and a >>>>>>> REPLACE line"
    )
    assert refused(f"new.py\n{create}\n{create}", ("new.py", C)) == (
        "new.py: an empty SEARCH writes the whole file, as its only block"
    )


def test_no_block_is_nothing_written() -> None:
    """The one-shot adapter read an empty parse as a tool request
    (`ToolRequest(read)`) or as `no_block`; the parser's own answer is that the reply wrote
    nothing, and that adapter retired."""
    assert parse("search_replace", "TOOL read path=app.py") == FileEdits()
    assert parse("search_replace", "I am not sure.") == FileEdits()


def test_a_block_with_two_dividers_is_refused_rather_than_split_by_guess() -> None:
    """The first `=======` divided it, and a unique wrong SEARCH applied."""
    reply = (
        "<<<<<<< SEARCH\nUsage\n=======\nold text\n=======\nUsage\n=======\nnew text\n"
        ">>>>>>> REPLACE"
    )
    assert parse("search_replace", reply) == ParseError(
        kind="bad_format",
        hint="block 0: it holds 3 ======= lines, so where its SEARCH ends is ambiguous; anchor "
        "on lines without one (an underline may be any length at least its title's)",
    )


def test_the_last_line_of_a_file_without_a_final_newline_can_be_edited() -> None:
    reply = "<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE"
    assert applied(parse("search_replace", reply), "a = 1\nb = 2") == "a = 1\nb = 3"
