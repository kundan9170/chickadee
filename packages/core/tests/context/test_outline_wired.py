"""A read of a code file through the box's real reader, over a repository the graph indexed:
what the worker and Athena are given (`box/compose._reader`).

Pinned before the readers were given the graph as their index, when a read of every code file
showed its head and tail and refused `at=` and `depth="signatures"`. Moved with T-3.185, listed
in its Log: a code file over the inline budget reads as its outline, `at=` and signatures open
its parts, and one that fits is read whole. The outline is of the file as it is on disk, the
index's own while it holds those bytes, and parsed again when they changed with no reindex."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from context_helpers import python_module
from repos import init_repo

from chickadee.box.compose import _reader  # pyright: ignore[reportPrivateUsage]
from chickadee.context.outline import whole_file
from chickadee.context.parts import WHOLE_FILE
from chickadee.context.reader import Reader
from chickadee.context.results import ToolResult
from chickadee.context.terms import INLINE_BYTES, Vision
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.journal.port import JournalPort
from chickadee.secrets.denylist import Denylist

RUN = cast(JournalPort, SimpleNamespace(run_id="2026-10-20-00000000"))
"""The one field a box reader reads of its journal before a stored handle opens."""


FUNCTIONS, _ = python_module(50, body=4)
"""Three hundred lines: fifty functions of five lines, each with a blank after."""

NESTED = "class Box:\n    def open(self, lid):\n        return lid\n"

MOD = FUNCTIONS + NESTED

NO_INDEX = "no index for this file, so no outline"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo", {"mod.py": MOD, "small.py": "def one():\n    return 1\n"}
    )


@pytest.fixture
def graph(repo: Path) -> Graph:
    indexer = Indexer(repo)
    indexer.full_index()
    return Graph(repo, indexer.conn)


def _box_reader(repo: Path, graph: Graph | None) -> Reader:
    return _reader(repo, Denylist(), Vision("worker"), RUN, graph)


def _ok(got: object) -> ToolResult:
    assert isinstance(got, ToolResult), got
    return got


def _row(line: int, label: str) -> str:
    """An outline row of `mod.py`, whose line numbers are three digits wide."""
    return f"{line:>3}  {label}"


def test_a_code_file_over_the_budget_reads_as_its_outline(repo: Path, graph: Graph) -> None:
    got = _ok(_box_reader(repo, graph).read("mod.py"))
    text = got.render()
    assert got.content is None, "a bare read is the preview, never the file"
    assert NO_INDEX not in text
    assert got.handle.shape.symbols == 52
    for i in range(1, 51):
        assert _row(6 * i - 5, f"f{i}(x)") in text
    assert _row(301, "class Box") in text
    assert _row(302, "  open(self, lid)") in text, "a method nests under its class"
    assert f"[{got.handle.id}]" in text


def test_at_a_name_returns_that_functions_lines(repo: Path, graph: Graph) -> None:
    got = _ok(_box_reader(repo, graph).read("mod.py", at="f3"))
    assert got.content is not None
    assert got.content.splitlines() == [
        f"{n}│{line}" for n, line in enumerate(MOD.splitlines()[12:17], start=13)
    ]


def test_signatures_elide_the_bodies(repo: Path, graph: Graph) -> None:
    got = _ok(_box_reader(repo, graph).read("mod.py", lines="1-12", depth="signatures"))
    assert got.content is not None
    assert "1│def f1(x):" in got.content and "7│def f2(x):" in got.content
    assert "2│    y1 = x + 101" in got.content, "with the body's first line"
    assert "y2 = x + 102" not in got.content, "the rest of the body elided"


@pytest.mark.writes_worktree("mod.py")
def test_a_file_edited_since_it_was_indexed_outlines_as_it_is_now(
    repo: Path, graph: Graph
) -> None:
    """A file changed with no reindex after it (here, by hand; in a session, the person's own
    edit, or a step's tree leaving the file out): the outline is parsed from the file on disk,
    never the index's lines."""
    edited = "def added(y):\n    return y\n\n" + MOD
    (repo / "mod.py").write_text(edited, encoding="utf-8")
    reader = _box_reader(repo, graph)
    text = _ok(reader.read("mod.py")).render()
    assert _row(1, "added(y)") in text
    assert _row(4, "f1(x)") in text and _row(1, "f1(x)") not in text
    assert _row(304, "class Box") in text
    f3 = _ok(reader.read("mod.py", at="f3"))
    assert f3.content is not None and f3.content.startswith("16│def f3(x):")


def test_a_small_code_file_is_read_whole(repo: Path, graph: Graph) -> None:
    got = _ok(_box_reader(repo, graph).read("small.py"))
    text = got.render()
    assert "1│def one():" in text and "2│    return 1" in text
    assert WHOLE_FILE.format(n=2) in text


def test_the_size_rule_is_the_inline_budget(tmp_path: Path) -> None:
    """The largest module whose every numbered line fits the inline budget is read whole; one
    function more, and it reads as its outline."""
    n = 1
    while whole_file(python_module(n + 1)[0].splitlines(), INLINE_BYTES) is not None:
        n += 1
    fits, over = python_module(n)[0], python_module(n + 1)[0]
    repo = init_repo(tmp_path / "repo", {"fits.py": fits, "over.py": over})
    indexer = Indexer(repo)
    indexer.full_index()
    reader = _box_reader(repo, Graph(repo, indexer.conn))
    whole = reader.read("fits.py").render()
    assert WHOLE_FILE.format(n=len(fits.splitlines())) in whole
    outlined = reader.read("over.py").render()
    assert WHOLE_FILE.split(":")[0] not in outlined
    assert f"f{n + 1}(x)" in outlined and "y1 = x +" not in outlined


def test_with_no_graph_a_read_is_as_before(repo: Path) -> None:
    reader = _box_reader(repo, None)
    assert NO_INDEX in _ok(reader.read("mod.py")).render()
    small = _ok(reader.read("small.py")).render()
    assert NO_INDEX in small, "the size rule is the outline's: no index, no rule"
    refused = reader.read("mod.py", at="f3")
    assert not isinstance(refused, ToolResult)
    assert "INDEX_UNAVAILABLE" in refused.render()
