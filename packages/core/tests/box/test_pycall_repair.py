"""Reading a worker's Python block that does not parse (`chickadee/pycall_repair.py`): each
cleanup, each kind of repair, the cases that must stay errors, and
the bound."""

from __future__ import annotations

import time

import pytest

from chickadee import pycall_repair
from chickadee.pycall_repair import Broken, Clean, Repaired, read_block

PARAMS = {
    "read": frozenset({"path", "start_line", "end_line", "symbol", "depth", "handle"}),
    "grep": frozenset({"pattern", "paths"}),
    "edit": frozenset({"path", "old", "new", "replace_all"}),
}
TWELVE = "\n".join(f'print(grep(pattern="x{n}", paths=["src/"]))' for n in range(11))


def test_a_block_that_parses_is_left_as_it_is() -> None:
    assert read_block('read(path="a.py")', PARAMS) == Clean('read(path="a.py")')


@pytest.mark.parametrize(
    ("written", "code", "cleaned"),
    [
        ("read(path=\u201ca.py\u201d)", 'read(path="a.py")', ("curly quotes",)),
        (
            "x = \u201cit\u2019s\u201d\nread(path=x)",
            'x = "it\u2019s"\nread(path=x)',
            ("curly quotes",),
        ),
        (
            '>>> x = read(path="a")\n>>> print(x)',
            'x = read(path="a")\nprint(x)',
            (">>> prompts",),
        ),
    ],
)
def test_a_cleanup_has_one_reading(written: str, code: str, cleaned: tuple[str, ...]) -> None:
    assert read_block(written, PARAMS) == Clean(code, cleaned)


def test_a_curly_quote_inside_a_string_is_the_strings_text() -> None:
    assert read_block('read(path="a\u201cb\u201d.py")', PARAMS) == Clean(
        'read(path="a\u201cb\u201d.py")'
    )


@pytest.mark.parametrize(
    ("written", "code", "what"),
    [
        (
            'read(path="a.py", start_line=1, end_line=40',
            'read(path="a.py", start_line=1, end_line=40)',
            "added `)` at line 1, column 44",
        ),
        (
            'read(path="a.py" start_line=1, end_line=5)',
            'read(path="a.py", start_line=1, end_line=5)',
            "added `,` at line 1, column 17",
        ),
        (
            'for p in ["a.py", "b.py"]\n    print(read(path=p))',
            'for p in ["a.py", "b.py"]:\n    print(read(path=p))',
            "added `:` at line 1, column 26",
        ),
        (
            # A `(` dropped in the first line is closed there, not at the block's end.
            'grep(pattern="foo", paths=["src/"]\nprint("x")',
            'grep(pattern="foo", paths=["src/"])\nprint("x")',
            "added `)` at line 1, column 35",
        ),
        (
            'edit(path="a",\n     old="x",\n     new="y"',
            'edit(path="a",\n     old="x",\n     new="y")',
            "added `)` at line 3, column 13",
        ),
    ],
)
def test_the_only_smallest_repair(written: str, code: str, what: str) -> None:
    read = read_block(written, PARAMS)
    assert isinstance(read, Repaired)
    assert (read.code, read.what) == (code, (what,))
    assert "SyntaxError" in read.error


def test_a_string_that_runs_across_lines_is_made_triple_quoted() -> None:
    written = (
        'edit(path="a.py", old="def f():\n    return 1\n", new="def f():\n    return 2\n")'
    )
    read = read_block(written, PARAMS)
    assert isinstance(read, Repaired)
    assert read.code == (
        'edit(path="a.py", old="""def f():\n    return 1\n""", '
        'new="""def f():\n    return 2\n""")'
    )


def test_a_string_holding_its_own_quote_is_no_repair() -> None:
    assert isinstance(read_block('edit(path="a", old="print("a")\n", new="x")', PARAMS), Broken)


def test_two_smallest_repairs_are_shown_as_the_error() -> None:
    """`(1 + (2))` and `(1) + (2)` are both one insertion each way: nothing says which."""
    read = read_block("x = (1 + (2", PARAMS)
    assert isinstance(read, Broken)
    assert (
        read.error
        == "line 1:\n    x = (1 + (2\n             ^\nSyntaxError: '(' was never closed"
    )


def test_a_repair_passing_a_keyword_the_function_does_not_take_is_none() -> None:
    """`print(grep(pattern="y"), paths=["a"])` parses, and passes print a `paths`."""
    read = read_block(TWELVE + '\nprint(grep(pattern="y", paths=["a"])', PARAMS)
    assert isinstance(read, Repaired)
    assert read.code.endswith('print(grep(pattern="y", paths=["a"]))')


def test_a_block_past_the_bound_is_shown_its_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pycall_repair, "BUDGET", 3)
    assert isinstance(read_block('read(path="a.py", start_line=1, end_line=40', PARAMS), Broken)


def test_an_unrepairable_block_ends_inside_the_bound() -> None:
    started = time.perf_counter()
    read = read_block(TWELVE + '\nprint(grep(pattern="y" paths=["a"] [', PARAMS)
    assert isinstance(read, Broken)
    assert time.perf_counter() - started < 5, "the bound holds the search to well under 5 s"


def test_the_error_names_the_line_and_the_column() -> None:
    read = read_block('read(path="a.py"))', PARAMS)
    assert isinstance(read, Broken)
    assert read.error.splitlines() == [
        "line 1:",
        '    read(path="a.py"))',
        "                     ^",
        "SyntaxError: unmatched ')'",
    ]


def test_two_closers_are_tried_in_both_orders() -> None:
    """`])` is a different repair from `)]`, and the one a call cut short needs."""
    read = read_block('grep("x", paths=["a"', PARAMS)
    assert isinstance(read, Repaired) and read.code == 'grep("x", paths=["a"])'
    # `foo([1, 2])` and `foo([1], 2)` both parse: two smallest repairs, so none.
    assert isinstance(read_block("x = foo([1, 2", PARAMS), Broken)
