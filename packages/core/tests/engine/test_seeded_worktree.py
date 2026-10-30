"""A worktree seeded from a dirty checkout.

"Try it in a worktree" from a checkout with uncommitted work no longer refuses: the worktree
starts with that work, untracked files included, in one first commit carrying a
`Chickadee-Seed` trailer. Landing writes only what the coder did after it, so the person's own
changes, still in their checkout, are never written twice.

The real box and workspace, the model scripted and no sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, Approving, build, make_ignoring_repo, solved
from engines import answer, call, calls
from repos import git
from zeus_turn_helpers import WAIT_S, result, state

from chickadee.context.view import held_worktree
from chickadee.engine import PacketEngine
from chickadee.errors import HandoffError
from chickadee.workspace import SEED_TRAILER

ONE_TASK = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}
THEIRS = "the person's own, uncommitted\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    made = make_ignoring_repo(tmp_path / "repo")
    (made / "notes.txt").write_text("committed\n", encoding="utf-8")
    git(made, "add", "notes.txt")
    git(made, "commit", "-q", "-m", "notes")
    return made


def _dirty(repo: Path) -> None:
    (repo / "notes.txt").write_text(THEIRS, encoding="utf-8")
    (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")


def _turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    *script: Any,
) -> PacketEngine:
    engine, _ = build(
        monkeypatch, repo, raw, [*script, answer("done")], interaction=Approving()
    )
    engines.append(engine)
    engine.turn(SID, "try it in a worktree").wait(timeout=WAIT_S)
    return engine


def _worktree(engine: PacketEngine) -> Path:
    held = held_worktree(state(engine, SID))
    assert held is not None and held.worktree_path is not None
    return Path(held.worktree_path)


@pytest.mark.writes_worktree("notes.txt", "scratch.txt")
def test_entering_from_a_dirty_checkout_carries_its_changes(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    head = git(repo, "rev-parse", "HEAD")
    _dirty(repo)

    engine = _turn(
        monkeypatch, repo, raw, engines, calls(call("worktree", "w1", action="enter"))
    )

    assert result(engine, "w1", SID).startswith("entered the worktree")
    tree = _worktree(engine)
    assert (tree / "notes.txt").read_text() == THEIRS
    assert (tree / "scratch.txt").read_text() == "untracked\n"
    assert git(tree, "rev-parse", "HEAD~1") == head, "one seed commit on the person's HEAD"
    assert f"{SEED_TRAILER}:" in git(tree, "log", "-1", "--format=%B")
    assert (repo / "notes.txt").read_text() == THEIRS, "the checkout is left as it was"
    assert git(repo, "rev-parse", "HEAD") == head
    held = held_worktree(state(engine, SID))
    assert held is not None and held.source_head == head, "the base is the person's HEAD"


@pytest.mark.writes_worktree("notes.txt", "scratch.txt", "app.py")
def test_bring_lands_only_the_coders_change(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    head = git(repo, "rev-parse", "HEAD")
    _dirty(repo)
    engine = _turn(
        monkeypatch,
        repo,
        raw,
        engines,
        calls(call("worktree", "w1", action="enter")),
        calls(call("coder", "c1", **ONE_TASK)),
        *solved("return 1", "return 2"),
    )

    summary = engine.handoff(SID)
    assert summary.git is not None and summary.git.overlapping == ()
    engine.finalize(SID, "bring")

    assert "return 2" in (repo / "app.py").read_text()
    assert (repo / "notes.txt").read_text() == THEIRS
    assert (repo / "scratch.txt").read_text() == "untracked\n"
    assert git(repo, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("choice", ["apply", "squash"])
@pytest.mark.writes_worktree("notes.txt", "scratch.txt", "app.py")
def test_apply_and_squash_refuse_a_seeded_worktree_naming_bring(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    choice: str,
) -> None:
    """Either would commit the person's own changes onto their branch for them."""
    head = git(repo, "rev-parse", "HEAD")
    _dirty(repo)
    engine = _turn(
        monkeypatch,
        repo,
        raw,
        engines,
        calls(call("worktree", "w1", action="enter")),
        calls(call("coder", "c1", **ONE_TASK)),
        *solved("return 1", "return 2"),
    )

    summary = engine.handoff(SID)
    why = (summary.refused or {}).get(choice, "")
    assert "uncommitted changes" in why and "bring" in why
    with pytest.raises(HandoffError, match="nothing was changed"):
        engine.finalize(SID, choice)
    assert git(repo, "rev-parse", "HEAD") == head
