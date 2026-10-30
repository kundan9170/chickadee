"""`PacketEngine.reindex` refreshes the checkout's graph.

Under the repository WRITE lease, with the engine's deny list, and `RepoNotIndexed`
before `init_repo()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo

from chickadee.config import EngineConfig
from chickadee.engine import PacketEngine
from chickadee.errors import RepoLeaseBusy, RepoNotIndexed
from chickadee.graph.queries import Graph
from chickadee.init import detect
from chickadee.init.lease import RepoLease

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {"app.py": APP, "vault/keys.py": "def secret() -> str:\n    return 'x'\n"},
    )


@pytest.fixture
def engine(repo: Path, raw: dict[str, Any]) -> Iterator[PacketEngine]:
    # The deny addition is the engine config's alone: no repository layer names it, so a
    # build that skips `vault/` read the engine's list.
    config = EngineConfig.with_defaults({**raw, "secrets": {"deny_add": ["vault/**"]}})
    engine = PacketEngine(repo, config)
    yield engine
    engine.close()


def knows(repo: Path, name: str) -> bool:
    graph = Graph(repo)
    try:
        return graph.symbol_exists(name)
    finally:
        graph.close()


def test_a_reindex_before_init_is_repo_not_indexed(engine: PacketEngine) -> None:
    with pytest.raises(RepoNotIndexed, match="init_repo"):
        engine.reindex()


def test_init_repo_returns_the_index_it_built(engine: PacketEngine, repo: Path) -> None:
    report = engine.init_repo()
    assert report.index.files_indexed == 1
    assert knows(repo, "go")


@pytest.mark.writes_worktree("vault/keys.py")
def test_the_engine_builds_and_refreshes_with_its_own_deny_list(
    engine: PacketEngine, repo: Path
) -> None:
    engine.init_repo()
    assert not knows(repo, "secret")
    (repo / "vault" / "keys.py").write_text("def secret2() -> str:\n    return 'y'\n")
    report = engine.reindex([Path("vault/keys.py")])
    assert "vault/keys.py" in {gap.path for gap in report.gaps}
    assert not knows(repo, "secret2")


@pytest.mark.writes_worktree("app.py")
def test_a_named_edit_is_answered(engine: PacketEngine, repo: Path) -> None:
    engine.init_repo()
    (repo / "app.py").write_text(APP + "\n\ndef later() -> int:\n    return 3\n")
    assert not knows(repo, "later")
    report = engine.reindex([Path("app.py")])
    assert report.files_indexed == 1
    assert knows(repo, "later")


@pytest.mark.writes_worktree("app.py")
def test_with_nothing_named_every_changed_file_is_found(
    engine: PacketEngine, repo: Path
) -> None:
    engine.init_repo()
    (repo / "app.py").write_text("def renamed() -> int:\n    return 1\n")
    engine.reindex()
    assert knows(repo, "renamed")
    assert not knows(repo, "go")


def test_a_reindex_waits_for_no_one_holding_the_lease(engine: PacketEngine, repo: Path) -> None:
    """A run's packet commits and a reindex never interleave. A held lease is refused
    in the engine's terms, naming the holder, never waited on silently."""
    engine.init_repo()
    identity = detect.resolve(repo)
    held = RepoLease.acquire(identity.git_common_dir, identity.repository_id)
    try:
        with pytest.raises(RepoLeaseBusy, match="pid"):
            engine.reindex()
    finally:
        held.release()
    engine.reindex()


@pytest.mark.writes_worktree("app.py")
def test_a_path_spelled_through_a_link_reaches_its_row(
    engine: PacketEngine, repo: Path, tmp_path: Path
) -> None:
    """The indexer names a file relative to the checkout's real root. A caller holding the
    checkout through a link, as `/var` is `/private/var` on macOS, is answered, not crashed."""
    engine.init_repo()
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    (repo / "app.py").write_text(APP + "\n\ndef later() -> int:\n    return 3\n")
    engine.reindex([alias / "app.py", tmp_path / "elsewhere.py"])
    assert knows(repo, "later")
