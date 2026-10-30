"""Fixtures for the tool pins: one small repository per role, as
each role's own suite builds it, so a pin reads what that role's real code does today."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from scout.scout_helpers import FileReader, GraphNavigate, ListSink, make_repo

from chickadee.executor.executor import Executor
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.isolation.none import Passthrough
from chickadee.scout.escapes import Escapes
from chickadee.scout.search import Search
from chickadee.scout.tools import ScoutTools

SOURCE = "def go():\n    return 1\n\n\ndef stop():\n    return 2\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Zeus's workspace: a source file and a deny-listed one."""
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "a.py").write_text(SOURCE, encoding="utf-8")
    (ws / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    return ws


@pytest.fixture
def scout_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return make_repo(root)


@pytest.fixture
def scout_graph(scout_repo: Path) -> Iterator[Graph]:
    ix = Indexer(scout_repo)
    ix.full_index()
    graph = Graph(scout_repo, ix.conn)
    yield graph
    ix.close()


@pytest.fixture
def scout_tools(scout_graph: Graph, scout_repo: Path) -> ScoutTools:
    escapes = Escapes(
        executor=Executor(backend=Passthrough()), cwd=scout_repo, allow_unsandboxed=True
    )
    tools = ScoutTools(
        GraphNavigate(scout_graph),
        FileReader(scout_repo),
        Search(root=scout_repo, rg=None),
        escapes=escapes,
    )
    tools.begin()
    return tools


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def worker_tree(tmp_path: Path) -> Path:
    """The packer suite's tree (`tests/packer/test_expand.py`), plus a file it does not
    index."""
    (tmp_path / "app.py").write_text("def build_widget(name): return name\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text("def helper(x): return x\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def helper(x): return x\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    return tmp_path
