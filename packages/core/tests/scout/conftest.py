"""Fixtures for the scout suite: a real repository, a real graph over it, and the scout's
ports faked where the scout takes one."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from scout_helpers import FileReader, GraphNavigate, ListSink, make_repo

from chickadee.executor.executor import Executor
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.isolation.none import Passthrough
from chickadee.profiles.loader import load_profile
from chickadee.profiles.model import LanguageProfile
from chickadee.scout.escapes import Escapes
from chickadee.scout.findings import Checker
from chickadee.scout.search import Search
from chickadee.scout.tools import ScoutTools


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return make_repo(root)


@pytest.fixture
def graph(repo: Path) -> Iterator[Graph]:
    ix = Indexer(repo)
    ix.full_index()
    g = Graph(repo, ix.conn)
    yield g
    ix.close()


@pytest.fixture
def python() -> LanguageProfile:
    return load_profile("python")


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def executor() -> Executor:
    """No sandbox backend in the suite's environments; the policy under test is the scout's,
    and `reconcile` is what reads the level the escape ran at."""
    return Executor(backend=Passthrough())


@pytest.fixture
def escapes(executor: Executor, repo: Path) -> Escapes:
    return Escapes(executor=executor, cwd=repo, allow_unsandboxed=True)


@pytest.fixture
def tools(graph: Graph, repo: Path, escapes: Escapes) -> ScoutTools:
    return ScoutTools(
        GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None), escapes=escapes
    )


@pytest.fixture
def checker(repo: Path, graph: Graph) -> Checker:
    return Checker(repo, graph)
