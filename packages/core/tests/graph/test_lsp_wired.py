"""The index's edges resolved by a real language server, and every production index wired to
it. The servers run under the harness's wrap; where a sandbox is required and none
seals, none starts."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import init_repo

from chickadee.box.spawn import HarnessInfraError, HarnessSpawn, harness_wrap, index_resolver
from chickadee.errors import InfraError
from chickadee.graph.edges import FALLBACK, RESOLVED
from chickadee.graph.indexer import Indexer
from chickadee.graph.lsp import CheckerResolver
from chickadee.graph.queries import Graph
from chickadee.init.orchestrate import IndexSandbox, init
from chickadee.isolation.none import Passthrough
from chickadee.process.supervisor import ProcessSupervisor, scrub_env
from chickadee.profiles import ProfileSet
from chickadee.profiles.tools import find_tool
from chickadee.secrets.denylist import Denylist


def _pyright_runs() -> bool:
    """Python's server as the index finds it (`find_tool`): basedpyright, which Chickadee ships.
    A runner where it still cannot start is a skip here, never a failed assertion."""
    found = find_tool("basedpyright", own=True)
    if found is None or find_tool("basedpyright-langserver", own=True) is None:
        return False
    try:
        done = ProcessSupervisor().run(
            (str(found), "--version"), cwd=Path.cwd(), env=scrub_env(), timeout_s=180
        )
    except InfraError:
        return False
    return done.exit_code == 0 and not done.timed_out


PYRIGHT = _pyright_runs()


def _repo(tmp_path: Path) -> Path:
    """A name defined twice, so name matching can only say `unknown`; the server knows which
    one `app.py` imports."""
    root = tmp_path / "r"
    init_repo(root)
    (root / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "other.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    (root / "app.py").write_text(
        "from util import helper\n\n\ndef go():\n    return helper()\n", encoding="utf-8"
    )
    return root


def _call(ix: Indexer) -> tuple[str, str]:
    row = ix.conn.execute(
        "SELECT dst, edge_class FROM edges WHERE file = 'app.py' AND type = 'calls'"
    ).fetchone()
    return row["dst"], row["edge_class"]


def _unsealed(require_sandbox: bool) -> HarnessSpawn:
    return HarnessSpawn(Passthrough(), ProcessSupervisor(), require_sandbox=require_sandbox)


@pytest.mark.skipif(not PYRIGHT, reason="pyright cannot run here")
def test_pyright_resolves_the_name_matching_cannot(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    profiles = ProfileSet.load(repo=root)
    resolver = index_resolver(root, _unsealed(require_sandbox=False), profiles)
    ix = Indexer(root, resolver, profiles=profiles)
    ix.full_index()
    assert _call(ix) == ("util.helper", RESOLVED)


def test_with_a_sandbox_required_and_none_here_no_server_starts(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    profiles = ProfileSet.load(repo=root)
    resolver = index_resolver(root, _unsealed(require_sandbox=True), profiles)
    ix = Indexer(root, resolver, profiles=profiles)
    ix.full_index()
    assert _call(ix)[1] != RESOLVED


def test_the_wrap_refuses_where_a_sandbox_is_required_and_none_seals(tmp_path: Path) -> None:
    with pytest.raises(HarnessInfraError, match="require_sandbox"):
        harness_wrap(_unsealed(require_sandbox=True), ("srv",), tmp_path, repo_writable=False)
    assert harness_wrap(_unsealed(require_sandbox=False), ("srv",), tmp_path) == ("srv",)


def test_the_index_sandbox_builds_a_resolver_over_the_repositorys_profiles(
    tmp_path: Path,
) -> None:
    """What `init` and `reindex` index with, rather than `NoResolver`."""
    root = _repo(tmp_path)
    profiles = ProfileSet.load(repo=root)
    resolver = IndexSandbox(Passthrough(), require_sandbox=True).resolver(
        root, Denylist(), profiles
    )
    assert isinstance(resolver, CheckerResolver)
    assert resolver.profiles is profiles


@pytest.mark.skipif(not PYRIGHT, reason="pyright cannot run here")
def test_init_resolves_where_the_sandbox_lets_the_server_run(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    init(root, sandbox=IndexSandbox(Passthrough(), require_sandbox=False))
    graph = Graph(root)
    try:
        callees = graph.callees("app.go")
    finally:
        graph.close()
    assert [(e.dst, e.edge_class) for e in callees] == [("util.helper", RESOLVED)]


def test_fallback_is_kept_where_no_server_answers(tmp_path: Path) -> None:
    root = tmp_path / "r"
    init_repo(root)
    (root / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "app.py").write_text("def go():\n    return helper()\n", encoding="utf-8")
    profiles = ProfileSet.load(repo=root)
    ix = Indexer(root, index_resolver(root, _unsealed(require_sandbox=True), profiles))
    ix.full_index()
    assert _call(ix) == ("util.helper", FALLBACK)
