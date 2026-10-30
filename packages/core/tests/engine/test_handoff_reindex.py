"""The checkout's code index after a handoff: what Zeus's view reads once the
session's worktree is handed off and the view moves back to the person's checkout.

Pinned before a handoff refreshed the checkout's index, when a function a run added and
`apply` landed was missing from it until the next `chickadee init`. Moved with T-3.185, listed
in its Log: a choice that lands reindexes the paths it put on the branch, in the same hold of
the WRITE lease; one that lands nothing reindexes nothing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, CoderModel, build, make_repo, one_task, solved
from repos import git, init_repo

from chickadee.engine import PacketEngine
from chickadee.errors import RepoNotIndexed
from chickadee.graph.queries import Graph
from chickadee.handoff import landed_paths

ADDED = "return 1\n\n\ndef stop() -> int:\n    return 0"
"""`app.py`'s `return 1`, and a function after it."""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def _run(engine: PacketEngine, model: CoderModel) -> None:
    model.then(*solved("return 1", ADDED))
    assert engine.coder(one_task(), session_id=SID).result().outcome == "COMPLETED"


def _indexed(repo: Path, name: str) -> bool:
    graph = Graph(repo)
    try:
        return graph.symbol_exists(name)
    finally:
        graph.close()


@pytest.mark.writes_worktree("app.py")
@pytest.mark.parametrize("choice", ["apply", "squash"])
def test_a_handoff_that_lands_reindexes_what_it_put_on_the_branch(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    choice: str,
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    _run(engine, model)
    engine.finalize(SID, choice)
    assert "def stop()" in (repo / "app.py").read_text()
    assert _indexed(repo, "stop")


def test_keep_lands_nothing_and_reindexes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    _run(engine, model)
    asked: list[Sequence[Path]] = []
    monkeypatch.setattr(
        "chickadee.engine._reindex_held", lambda root, changed, **_: asked.append(changed)
    )
    engine.finalize(SID, "keep")
    assert asked == []
    assert not _indexed(repo, "stop"), "the branch did not move"


@pytest.mark.writes_worktree("app.py")
def test_a_checkout_never_indexed_still_hands_off(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    _run(engine, model)

    def unindexed(root: Path, changed: Sequence[Path], **_: object) -> None:
        raise RepoNotIndexed(f"{root} has no code index")

    monkeypatch.setattr("chickadee.engine._reindex_held", unindexed)
    result = engine.finalize(SID, "apply")
    assert result.commits_applied and "def stop()" in (repo / "app.py").read_text()


def test_landed_paths_name_each_changed_path_and_both_names_of_a_rename(
    tmp_path: Path,
) -> None:
    repo = init_repo(tmp_path / "r", {"a.py": "a = 1\n", "b.py": "b = 1\n", "c.py": "c = 1\n"})
    before = git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("a = 2\n", encoding="utf-8")
    git(repo, "mv", "b.py", "d.py")
    git(repo, "commit", "-qam", "change")
    after = git(repo, "rev-parse", "HEAD")
    assert landed_paths(repo, before, after) == ("a.py", "b.py", "d.py")
    assert landed_paths(repo, after, after) == ()
    assert landed_paths(repo, None, before) == ("a.py", "b.py", "c.py")
