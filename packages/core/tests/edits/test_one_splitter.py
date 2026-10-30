"""Every place that numbers a file's lines splits them with `edits/lines.split_lines`: only
`\\n` splits, so a form feed, a vertical tab, `\\u2028` or a lone `\\r` inside a
line never moves a line number away from the file's, which the reader, tree-sitter and rg all
count by `\\n`.

Each answer moved in `T-3.127`; each test names the one it replaced. On a file of three
lines by `\n`, the four numbered five, because `str.splitlines` also split on the form feed and
the lone `\r` inside the first two lines.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.graph.lsp import _column
from chickadee.scout.findings import Checker
from chickadee.scout.search import Search
from chickadee.secrets.suspect import scan_text

# Three lines by `\n`: a form feed inside the first, a lone `\r` inside the second.
FILE = "a = 1\x0cb = 2\nc = 3\rd = 4\nx = build_widget\n"


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "f.py").write_bytes(FILE.encode())
    return tmp_path


def test_the_scout_s_literal_search_numbers_lines_as_the_file_does(tmp_path: Path) -> None:
    """Before: `f.py:1: a = 1` … `f.py:5: x = build_widget`, five hits on three lines."""
    out = Search(root=_repo(tmp_path), rg=None)("= ")
    assert out.split("\n")[1:] == [
        "f.py:1: a = 1\x0cb = 2",
        "f.py:2: c = 3\rd = 4",
        "f.py:3: x = build_widget",
    ]


def test_the_citation_checker_reads_the_file_s_lines(tmp_path: Path) -> None:
    """Before: five lines, so a citation of line 3 was checked against `c = 3`."""
    assert Checker(_repo(tmp_path), None).lines("f.py") == [
        "a = 1\x0cb = 2",
        "c = 3\rd = 4",
        "x = build_widget",
    ]


def test_the_lsp_column_is_found_on_the_graph_s_line(tmp_path: Path) -> None:
    """Before: 0, because the graph's line 3 (tree-sitter counts `\n`) was read as `c = 3`."""
    assert _column(_repo(tmp_path), "f.py", 3, "build_widget") == 4


def test_a_secret_scan_finding_names_the_file_s_line() -> None:
    """Before: line 3. The known-format detectors already counted lines by `\n`
    (`text.count("\n", …)`); the entropy pass now agrees with them."""
    text = "x = 1\x0cy = 2\nblob_value = 'Zq8xLm3Np0Rt7Vw2Ks9Yb4Jh6G'\n"
    assert [f.line for f in scan_text("s.py", text)] == [2]
