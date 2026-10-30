"""A fork asks every time: the code back to the fork point, or kept as it is, through the
facade. Both choices in place and in a worktree; the parent's log
is never written."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_ignoring_repo, solved
from repos import git
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

from chickadee.context.session import Session
from chickadee.context.store import LOG_NAME, session_dir, sessions_root
from chickadee.engine import Forked, PacketEngine
from chickadee.errors import SessionError
from chickadee.state.roots import roots as resolve_roots

KEPT = "code changed in turns you no longer see: app.py"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository that ignores the caches its run's checks leave behind, as real ones do."""
    return make_ignoring_repo(tmp_path / "repo")


def talk_then_edit(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], workspace: str
) -> tuple[PacketEngine, int]:
    """A turn that only talks, then one that changes `return 1` to `return 2`. Returns the
    engine and the fork point: the last line before the editing turn."""
    work = {
        "title": "Return 2",
        "brief": "## Goal\n`go()` returns 2.\n",
        "workspace": workspace,
    }
    script = [
        answer("hi"),
        calls(call("coder", "c1", **work)),
        *solved("return 1", "return 2"),
        answer("changed"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engine.turn(SID, "hello").wait(timeout=WAIT_S)
    engine.turn(SID, "change go()").wait(timeout=WAIT_S)
    users = [t.seq for t in state(engine, SID).turns() if t.role == "user"]
    return engine, users[1] - 1


def parent_log(repo: Path) -> bytes:
    return (session_dir(sessions_root(resolve_roots(repo)), SID) / LOG_NAME).read_bytes()


def app(root: Path) -> str:
    return (root / "app.py").read_text(encoding="utf-8")


def test_in_place_restore_takes_the_checkout_back_and_tells_no_kept_code(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, at = talk_then_edit(monkeypatch, repo, raw, "in_place")
    try:
        log = parent_log(repo)
        forked = engine.fork(SID, at, "try it another way", code="restore")
        assert forked.code == "restore" and forked.worktree is None
        (done,) = forked.workspaces
        assert done.restored == ("app.py",)
        assert "return 1" in app(repo)
        assert not any(n.startswith(KEPT) for n in forked.notices)
        assert parent_log(repo) == log, "the parent's log is never written"
    finally:
        engine.close()


def test_in_place_keep_touches_nothing_and_names_the_code(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, at = talk_then_edit(monkeypatch, repo, raw, "in_place")
    try:
        log = parent_log(repo)
        forked = engine.fork(SID, at, "try it another way", code="keep")
        assert forked.workspaces == () and "return 2" in app(repo)
        assert sum(n.startswith(KEPT) for n in forked.notices) == 1
        assert parent_log(repo) == log
    finally:
        engine.close()


@pytest.mark.parametrize(("code", "expected"), [("restore", "return 1"), ("keep", "return 2")])
def test_a_worktree_parent_gives_the_child_its_own_worktree(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], code: str, expected: str
) -> None:
    engine, at = talk_then_edit(monkeypatch, repo, raw, "worktree")
    try:
        log = parent_log(repo)
        forked: Forked = engine.fork(SID, at, "try it another way", code=code)  # type: ignore[arg-type]
        assert forked.worktree is not None
        assert expected in app(Path(forked.worktree))
        assert "return 1" in app(repo), "the person's checkout is untouched"
        assert parent_log(repo) == log
        assert (sum(n.startswith(KEPT) for n in forked.notices) == 1) == (code == "keep")
    finally:
        engine.close()


def test_unattended_the_fork_keeps_the_code(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """`keep` is the default, which touches no file."""
    engine, at = talk_then_edit(monkeypatch, repo, raw, "in_place")
    try:
        forked = engine.fork(SID, at, "try it another way")
        assert forked.code == "keep" and "return 2" in app(repo)
    finally:
        engine.close()


def test_a_fork_point_past_the_log_is_the_callers_mistake_not_a_corrupt_log(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """Found in review: reading past the last line raised `SessionCorrupt`, telling the person
    their log was damaged when only the seq was wrong."""
    engine, _ = talk_then_edit(monkeypatch, repo, raw, "in_place")
    try:
        last = state(engine, SID).last_seq
        with pytest.raises(SessionError, match=f"no line {last + 5} to fork at"):
            engine.fork(SID, last + 5, "too far", code="restore")
        assert "return 2" in app(repo)
    finally:
        engine.close()


def test_the_child_worktree_names_the_branch_it_came_from(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """Found in review: `source_ref` is the person's branch, never the child's own."""
    engine, at = talk_then_edit(monkeypatch, repo, raw, "worktree")
    try:
        forked = engine.fork(SID, at, "try it another way", code="restore")
        root = sessions_root(resolve_roots(repo))
        child = Session.open(root, forked.child_id)
        with child:
            ref = child.state.workspace_at(child.state.last_seq)
        assert ref is not None
        assert ref.source_ref == git(repo, "branch", "--show-current")
        assert ref.branch is not None and ref.branch != ref.source_ref
    finally:
        engine.close()


def test_an_unsafe_fork_point_is_refused_with_nothing_changed(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, _ = talk_then_edit(monkeypatch, repo, raw, "in_place")
    try:
        (called,) = [t.seq for t in state(engine, SID).turns() if t.calls]
        log = parent_log(repo)
        with pytest.raises(SessionError, match="still waiting for its result"):
            engine.fork(SID, called, "mid-call", code="restore")
        assert "return 2" in app(repo) and parent_log(repo) == log
    finally:
        engine.close()
