"""D172's pins on line splitting and numbering, recorded before `T-3.127` moved them
to `edits/lines.py` and gave the worker's views one splitter and one newline policy. Every
builder of a `N│` prefix `T-3.127` moved is pinned byte for byte here: `number_lines`,
`head_tail`'s preview lines (`_preview_line`), `fit.render_lines` with its cut and edges, and
`fit.cost_of`'s arithmetic. The prefix's reader (`context/citations`) is held by
`test_citations.py`.

Two answers moved in `T-3.127`, each listed in its Log: `search` numbered lines with
`str.splitlines`, which also splits on `\\x0c`, so a hit after one was a line later than the
file says; and `read` showed a mixed-ending file through universal newlines, as `\\n` that no
anchor could match. Both followed `edits/lines`; since the packet's views retired, the box's
edit port is pinned holding a file by the same policy.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.box.edits import WorktreeEdits
from chickadee.box.syntax import SyntaxChecker
from chickadee.context.fit import cost_of, render_lines
from chickadee.context.outline import head_tail
from chickadee.context.spans import Span
from chickadee.edits.lines import elided, number_lines, split_lines
from chickadee.secrets.denylist import Denylist


def test_number_lines_prefixes_each_line_and_marks_a_gap() -> None:
    lines = split_lines("a\nb\nc\nd\ne\nf\n")
    assert number_lines(lines, [Span(1, 2), Span(5, 6)]) == [
        "1│a",
        "2│b",
        "[elided: lines 3-4]",
        "5│e",
        "6│f",
    ]


LINES = [f"line {i}" for i in range(1, 13)]
LINES[1] = "x" * 250


def test_head_tail_numbers_to_the_file_s_width_and_cuts_a_long_line() -> None:
    outline = head_tail(LINES, 2, 2, 10_000)
    assert outline.rows == (
        " 1│line 1",
        f" 2│{'x' * 240}…[+10 chars]",
        "[elided: lines 3-10]",
        "11│line 11",
        "12│line 12",
    )
    assert outline.shown == (Span(1, 1), Span(11, 12))


def test_render_lines_cuts_by_bytes_and_marks_the_edges() -> None:
    assert render_lines(LINES, [Span(1, 2), Span(11, 12)], cut={2: 20}) == (
        f" 1│line 1\n 2│{'x' * 20}…[+230 bytes]\n[elided: lines 3-10]\n11│line 11\n12│line 12"
    )
    # The width is the shown spans' last line's, not the file's.
    assert render_lines(LINES, [Span(3, 4)], edges=Span(1, 12)) == (
        "[elided: lines 1-2]\n3│line 3\n4│line 4\n[elided: lines 5-12]"
    )


def test_cost_of_counts_each_line_s_prefix() -> None:
    assert (cost_of(LINES, [Span(1, 2)]), cost_of(LINES, [Span(3, 4)])) == (268, 24)


def test_the_elision_marker_reads_these_bytes() -> None:
    assert (elided(3, 3), elided(3, 4)) == ("[elided: line 3]", "[elided: lines 3-4]")


def test_split_lines_does_not_split_on_a_form_feed() -> None:
    assert split_lines("x\x0cy\nz\n") == ("x\x0cy", "z")


def _drafted(root: Path, path: str) -> str:
    """`path` as the worker's edit port holds it to anchor in (`box/edits.WorktreeEdits`)."""
    port = WorktreeEdits(root, Denylist(), SyntaxChecker(root, ()), untracked_max_bytes=0)
    with port.edit() as call:
        text = call.text(path)
    assert isinstance(text, str), text
    return text


def test_the_worker_edits_follow_the_one_newline_policy(tmp_path: Path) -> None:
    """Moved by `T-3.127`. Before: `read` showed universal-newline text, `…name\nx = 1…`.
    This file has one `\r\n` and one `\n`, mixed endings, so it is held as it is; and a form
    feed does not split a line. (The packet's `read` and `search` views retired with the packet
    loop, D196, `T-3.157`: the box's edit port is where an anchor is matched now.)"""
    (tmp_path / "app.py").write_bytes(
        b"def build_widget(name): return name\r\nx = 1\x0cy = 2\n"
    )
    text = _drafted(tmp_path, "app.py")
    assert text == "def build_widget(name): return name\r\nx = 1\x0cy = 2\n"
    assert split_lines(text)[1] == "x = 1\x0cy = 2"


def test_the_worker_holds_a_uniformly_crlf_file_as_lf(tmp_path: Path) -> None:
    """Unchanged by `T-3.127`: what universal newlines showed for a uniform file, the newline
    policy shows too, so the anchors copied from it match."""
    (tmp_path / "app.py").write_bytes(b"def build_widget(name): return name\r\nx = 1\r\n")
    assert _drafted(tmp_path, "app.py") == "def build_widget(name): return name\nx = 1\n"
