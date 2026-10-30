"""A language's linter kept as a server of its own beside the language's.

What these pin: one file queued for its language server and its linter is two questions, each
asked at its own newest generation; a server's state is its own. In a pass, the linter (the
real `ruff server`) judges each file beside the language server, its verdict the file's own; a
dependency's change re-checks the language server's verdict and leaves the linter's; a task's
start is judged by both; a linter not installed is no source.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import pytest
from test_lsp import _unwrapped, _with_server
from test_servers import _languages, fake  # noqa: F401 # pyright: ignore[reportUnusedImport]

from chickadee.graph.diagnostics import DiagCoverage
from chickadee.graph.indexer import Indexer
from chickadee.graph.lsp import CheckerResolver
from chickadee.graph.queries import Graph
from chickadee.graph.servers import EpisodeServers, Recheck, Server, ServerSlots
from chickadee.process.supervisor import ProcessSupervisor
from chickadee.profiles import ProfileSet, load_profile


def _two(tmp_path: Path) -> tuple[EpisodeServers, Server, Server]:
    servers = EpisodeServers(tmp_path, ProcessSupervisor(), slots=ServerSlots())
    # A worker that is already there and never runs: what is queued stays queued.
    servers._worker = threading.Thread(target=lambda: None)  # pyright: ignore[reportPrivateUsage]
    profile = _languages("fake-ls --stdio").all()[0]
    return (
        servers,
        Server(profile, ("fake-ls",), "fake-ls"),
        Server(profile, ("fake-lint",), "fake-lint", local=True),
    )


def test_one_file_for_two_servers_is_two_questions(tmp_path: Path) -> None:
    servers, language, linter = _two(tmp_path)
    servers.recheck([Recheck("b.py", language, 2), Recheck("b.py", linter, 2)])
    servers.recheck([Recheck("b.py", language, 3)])
    assert servers._queued == {  # pyright: ignore[reportPrivateUsage]
        ("b.py", "fake-ls"): 3,
        ("b.py", "fake-lint"): 2,
    }


def test_each_server_keeps_its_own_state(tmp_path: Path) -> None:
    _, language, linter = _two(tmp_path)
    assert language.key != linter.key
    assert {language.key, linter.key} == {
        f"{language.profile.name}/fake-ls",
        f"{language.profile.name}/fake-lint",
    }


# -- the linter asked in a pass (step 4) ----------------------------------------------------

RUFF = shutil.which("ruff")
needs_ruff = pytest.mark.skipif(RUFF is None, reason="ruff is not installed")

UTIL = "def helper(x):\n    return x\n"
APP = "import os\n\nfrom util import helper\n\n\ndef go():\n    return helper(1)\n"


def _linted(
    root: Path, lint: str | None = "ruff server"
) -> tuple[Indexer, Graph, EpisodeServers]:
    servers = EpisodeServers(root.resolve(), ProcessSupervisor(), slots=ServerSlots())
    profiles = ProfileSet.of([_with_server(load_profile("python"), "fake-ls --stdio", lint)])
    resolver = CheckerResolver(
        root, ProcessSupervisor(), profiles, wrap=_unwrapped, servers=servers
    )
    ix = Indexer(root, resolver)
    ix.full_index()
    return ix, Graph(root, ix.conn), servers


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "r"
    root.mkdir()
    (root / "util.py").write_text(UTIL, encoding="utf-8")
    (root / "app.py").write_text(APP, encoding="utf-8")
    return root


@needs_ruff
@pytest.mark.usefixtures("fake")
def test_a_pass_asks_the_linter_beside_the_language_server(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix, graph, servers = _linted(root)
    try:
        found = graph.diagnostics_in("app.py")
        assert found.coverage is DiagCoverage.TOTAL
        assert [(d.code, d.severity, d.source) for d in found.in_span] == [
            ("F401", "warning", "Ruff")
        ]
        sources = {
            str(r[0]): bool(r[1])
            for r in ix.conn.execute("SELECT source, local FROM analyzed WHERE path = 'app.py'")
        }
        assert sources == {"fake-ls": False, "ruff": True}
    finally:
        servers.close()


@needs_ruff
@pytest.mark.usefixtures("fake")
def test_a_dependency_s_change_rechecks_the_language_server_only(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix, graph, servers = _linted(root)
    try:
        (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
        ix.reindex([root / "util.py"])
        assert ix.settle(30.0)
        found = graph.diagnostics_in("app.py")
        assert found.coverage is DiagCoverage.TOTAL
        assert ("F401", "Ruff") in [(d.code, d.source) for d in found.in_span]
    finally:
        servers.close()


@needs_ruff
@pytest.mark.usefixtures("fake")
def test_a_start_is_judged_by_both(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix, _, servers = _linted(root)
    try:
        assert isinstance(ix.resolver, CheckerResolver)
        found = ix.resolver.judge({"app.py": b"import sys\nBAD = 1\n"})
        by = {v.source: (v.local, [d.message for d in v.diagnostics]) for v in found}
        assert by["fake-ls"] == (False, ["bad"])
        assert by["ruff"][0] is True and any("sys" in m for m in by["ruff"][1])
    finally:
        servers.close()


@pytest.mark.usefixtures("fake")
def test_a_linter_not_installed_is_no_source(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    ix, _, servers = _linted(root, lint="no-such-linter server")
    try:
        sources = {str(r[0]) for r in ix.conn.execute("SELECT source FROM analyzed")}
        assert sources == {"fake-ls"}
    finally:
        servers.close()
