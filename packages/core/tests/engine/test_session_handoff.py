"""The session's handoff.

`handoff()` describes the worktree the session still writes, its diff and the choices it
accepts; `finalize()` carries out one of `apply`, `squash`, `keep` or `discard` under the WRITE
lease, written ahead as `handoff_intended` and `handoff_completed`, and records the choice on a
new `workspace` line that the next `coder` call reads. Nothing is pushed. A crash between the
intent and the completion is settled by what the repository shows, never repeated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    CoderModel,
    build,
    make_ignoring_repo,
    one_task,
    solved,
)
from repos import git
from zeus_turn_helpers import state

import chickadee.session_handoff as session_handoff
from chickadee.context.session import Session
from chickadee.context.view import held_worktree, latest_workspace
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError, EngineBusy, HandoffError, InfraError
from chickadee.executor.reconcile import TRAILER
from chickadee.handoff import HandoffResult
from chickadee.handoff import finalize as handoff_finalize
from chickadee.ids import handoff_operation
from chickadee.session_handoff import (
    HANDOFF_COMPLETED,
    HANDOFF_INTENDED,
    NOT_VERIFIED,
    HandoffCompleted,
    HandoffIntended,
    dangling,
)
from chickadee.workspace import WORKTREES_DIR


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository that ignores `__pycache__/`, as most do: a test run leaves one in the
    worktree, and `git worktree remove` refuses an unignored file (see the test of it)."""
    return make_ignoring_repo(tmp_path / "repo")


def worktrees(repo: Path) -> list[str]:
    root = repo / ".git" / WORKTREES_DIR
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


def owned(engine: PacketEngine) -> list[WorkspaceRef]:
    return [w.ref for w in state(engine, SID)._workspaces]  # pyright: ignore[reportPrivateUsage]


def session_of(engine: PacketEngine) -> Session:
    conversation = engine._conversations.get(SID)  # pyright: ignore[reportPrivateUsage]
    assert conversation is not None
    opened = conversation.opened()
    assert opened is not None
    return opened.session


def solve(engine: PacketEngine, model: CoderModel, old: str, new: str) -> str:
    """A one-task run that changes `old` to `new` in `app.py`, committed."""
    model.then(*solved(old, new))
    handle = engine.coder(one_task(), session_id=SID)
    assert handle.result().outcome == "COMPLETED"
    return handle.run_id


def kinds(engine: PacketEngine) -> list[str]:
    return [
        event.kind
        for event, _ in state(engine, SID).lines()
        if event.kind in (HANDOFF_INTENDED, HANDOFF_COMPLETED)
    ]


def completions(engine: PacketEngine) -> list[HandoffCompleted]:
    session = session_of(engine)
    return [
        session_handoff._COMPLETED.validate_json(session.blob(e.payload_sha))  # pyright: ignore[reportPrivateUsage]
        for e, _ in session.state.lines()
        if e.kind == HANDOFF_COMPLETED
    ]


def two_calls(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> tuple[PacketEngine, CoderModel]:
    engine, model = build(monkeypatch, repo, raw)
    solve(engine, model, "return 1", "return 2")
    solve(engine, model, "return 2", "return 3")
    return engine, model


def test_the_summary_shows_the_worktree_its_diff_and_says_behaviour_is_not_verified(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = two_calls(monkeypatch, repo, raw)
    engines.append(engine)
    start = git(repo, "rev-parse", "HEAD")

    summary = engine.handoff(SID)

    assert summary.git is not None and summary.workspace is not None
    assert summary.git.base == start == summary.workspace.source_head
    assert len(summary.git.commits) == 2, "both calls' commits, on the one branch"
    assert summary.refused == {}, "a clean, unmoved checkout accepts all four"
    assert "-    return 1" in summary.diff and "+    return 3" in summary.diff
    assert len(summary.episodes) == 2 and summary.last_outcome == "COMPLETED"
    assert summary.unverified[0] == NOT_VERIFIED, "the record says so"
    assert summary.notes == (), "nothing bears on the choice"
    assert kinds(engine) == [], "a summary writes nothing"


@pytest.mark.writes_worktree("app.py")
def test_apply_fast_forwards_keeps_the_task_commits_and_the_next_call_opens_a_new_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = two_calls(monkeypatch, repo, raw)
    engines.append(engine)
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    tip = git(Path(ref.worktree_path), "rev-parse", "HEAD")

    result = engine.finalize(SID, "apply")

    assert isinstance(result, HandoffResult) and result.pushed is False
    assert git(repo, "rev-parse", "HEAD") == tip, "the task commits keep their own hashes"
    assert len(result.commits_applied) == 2
    assert "return 3" in (repo / "app.py").read_text()
    assert kinds(engine) == [HANDOFF_INTENDED, HANDOFF_COMPLETED]
    assert completions(engine)[0].operation_id == handoff_operation(SID, 1)
    assert owned(engine)[-1].handoff == "apply"
    assert engine.handoff(SID).git is None, "nothing left to hand off"
    with pytest.raises(HandoffError, match="no worktree to hand off"):
        engine.finalize(SID, "apply")

    solve(engine, model, "return 3", "return 4")
    assert worktrees(repo) == [SID, f"{SID}-2"]
    assert owned(engine)[-1].source_head == tip, "from the new HEAD"


@pytest.mark.writes_worktree("app.py")
def test_squash_lands_one_commit_carrying_the_operation_trailer(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = two_calls(monkeypatch, repo, raw)
    engines.append(engine)
    start = git(repo, "rev-parse", "HEAD")

    result = engine.finalize(SID, "squash")

    (commit,) = result.commits_applied
    assert git(repo, "rev-parse", "HEAD") == commit
    assert git(repo, "rev-parse", "HEAD^") == start, "one commit on the base"
    message = git(repo, "log", "-1", "--format=%B")
    assert f"{TRAILER}: {handoff_operation(SID, 1)}" in message
    assert owned(engine)[-1].handoff == "squash"


def test_keep_changes_nothing_and_later_calls_reattach_the_same_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    head = git(repo, "rev-parse", "HEAD")

    engine.finalize(SID, "keep")

    assert git(repo, "rev-parse", "HEAD") == head
    assert owned(engine)[-1].handoff == "keep"
    solve(engine, model, "return 2", "return 3")
    assert worktrees(repo) == [SID], "keep is the person's not yet: the same worktree"
    summary = engine.handoff(SID)
    assert summary.git is not None and len(summary.git.commits) == 2


def test_discard_removes_the_worktree_and_branch_and_never_touches_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = two_calls(monkeypatch, repo, raw)
    engines.append(engine)
    assert "discard" not in engine.handoff(SID).refused
    (ref,) = owned(engine)
    head = git(repo, "rev-parse", "HEAD")

    engine.finalize(SID, "discard")

    assert worktrees(repo) == []
    assert git(repo, "branch", "--list", ref.branch or "") == ""
    assert git(repo, "rev-parse", "HEAD") == head
    assert "return 1" in (repo / "app.py").read_text()
    after = owned(engine)[-1]
    assert (after.handoff, after.worktree_path) == ("discard", None)


@pytest.mark.writes_worktree(".gitignore")
def test_discard_over_unignored_files_deletes_them_with_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The `__pycache__/` a test run leaves, in a repository that does not ignore it, is
    named by the review and offered for `discard`, which deletes it with the worktree. It used
    to be refused up front, leaving git by hand as the only way to discard."""
    git(repo, "rm", "-q", ".gitignore")
    git(repo, "commit", "-q", "-m", "no ignores")
    engine, _ = two_calls(monkeypatch, repo, raw)
    engines.append(engine)
    path = owned(engine)[-1].worktree_path
    assert path is not None
    (Path(path) / "__pycache__").mkdir()
    (Path(path) / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00")
    summary = engine.handoff(SID)
    assert "discard" not in summary.refused
    assert any("__pycache__/" in n for n in summary.notes)

    result = engine.finalize(SID, "discard")

    assert "deleted work no commit held: __pycache__/" in result.detail
    assert worktrees(repo) == [] and not Path(path).exists()
    assert owned(engine)[-1].handoff == "discard"


@pytest.mark.writes_worktree("other.txt")
def test_a_moved_source_is_refused_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (repo / "other.txt").write_text("the person's own work\n")
    git(repo, "add", "other.txt")
    git(repo, "commit", "-q", "-m", "moved on")

    summary = engine.handoff(SID)

    assert set(summary.refused) == {"apply", "squash"}
    assert "moved during the run" in summary.refused["apply"]
    with pytest.raises(HandoffError, match="nothing was changed"):
        engine.finalize(SID, "apply")
    assert kinds(engine) == [], "a refusal writes no intent"
    assert owned(engine)[-1].handoff == "pending"


def test_a_crash_after_the_git_change_is_settled_as_landed_and_never_repeated(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A kill between the merge and its completion: resume observes the tip on the source
    branch, writes the completion and the `apply` line, and runs no second merge."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    tip = git(Path(ref.worktree_path), "rev-parse", "HEAD")
    real = handoff_finalize

    def killed(*args: Any, **kwargs: Any) -> HandoffResult:
        real(*args, **kwargs)
        raise InfraError("the process died after the merge")

    monkeypatch.setattr(session_handoff, "finalize", killed)
    with pytest.raises(InfraError):
        engine.finalize(SID, "apply")
    assert kinds(engine) == [HANDOFF_INTENDED], "the intent is open"
    assert git(repo, "rev-parse", "HEAD") == tip
    assert engine.handoff(SID).notes[-1].startswith("the apply handoff")

    monkeypatch.setattr(session_handoff, "finalize", lambda *_a, **_k: pytest.fail("repeated"))
    engine.resume(SID)

    (done,) = completions(engine)
    assert (done.outcome, done.reconciled) == ("done", True)
    assert owned(engine)[-1].handoff == "apply"
    session = session_of(engine)
    assert dangling(session.state, lambda e: session.blob(e.payload_sha)) is None


def test_a_crash_before_the_git_change_is_settled_as_not_landed_and_chosen_again(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    head = git(repo, "rev-parse", "HEAD")
    session = session_of(engine)
    with session.lock:
        session.append(HandoffIntended(handoff_operation(SID, 1), "discard", head, None))

    engine.finalize(SID, "keep")  # settles the open intent first, then carries out keep

    first, second = completions(engine)
    assert (first.operation_id, first.outcome, first.reconciled) == (
        handoff_operation(SID, 1),
        "not_landed",
        True,
    )
    assert ref.branch is not None and ref.branch in first.detail
    assert (second.operation_id, second.outcome) == (handoff_operation(SID, 2), "done")
    assert owned(engine)[-1].handoff == "keep"


def test_the_lines_survive_a_restart(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Replay: a new engine folds both kinds and reads the session's handoff from them."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    engine.finalize(SID, "keep")
    engine.close()

    again, _ = build(monkeypatch, repo, raw)
    engines.append(again)
    summary = again.handoff(SID)
    assert summary.workspace is not None and summary.workspace.handoff == "keep"
    again.finalize(SID, "discard")
    assert again.handoff(SID).git is None


def test_without_a_worktree_there_is_nothing_to_hand_off(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    summary = engine.handoff(SID)
    assert (summary.git, summary.workspace, summary.diff, summary.episodes) == (
        None,
        None,
        "",
        (),
    )
    assert summary.last_outcome is None
    with pytest.raises(HandoffError, match="no worktree to hand off"):
        engine.finalize(SID, "apply")
    with pytest.raises(ConfigError, match="not a handoff choice"):
        engine.finalize(SID, "push")


def test_a_run_in_flight_refuses_a_handoff(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    episodes = engine._episodes  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(episodes, "busy", lambda: "run R-1")
    with pytest.raises(EngineBusy, match="hand off when it ends"):
        engine.finalize(SID, "apply")
    assert kinds(engine) == []


# -- the review's findings, pinned ----------------------------------------------------------


@pytest.mark.writes_worktree("app.py")
def test_the_next_run_settles_a_crashed_apply_and_opens_a_new_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A session reopened without `resume()` (the REPL's `--continue`): its next run
    settles the interrupted apply first, so it never reattaches the handed-off worktree."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    real = handoff_finalize

    def killed(*args: Any, **kwargs: Any) -> HandoffResult:
        real(*args, **kwargs)
        raise InfraError("the process died after the merge")

    monkeypatch.setattr(session_handoff, "finalize", killed)
    with pytest.raises(InfraError):
        engine.finalize(SID, "apply")

    solve(engine, model, "return 2", "return 3")

    assert [c.outcome for c in completions(engine)] == ["done"]
    assert worktrees(repo) == [SID, f"{SID}-2"], "a new worktree, from the applied HEAD"
    assert owned(engine)[-1].handoff == "pending"


def test_settling_never_overwrites_a_newer_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """An intent over the first worktree, settled after a newer `workspace` line: its
    completion is written, and the newer line stays the session's."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    session = session_of(engine)
    with session.lock:
        intent = HandoffIntended(handoff_operation(SID, 1), "keep", ref.source_head, None)
        session.append(intent)
    newer = WorkspaceRef(
        mode="worktree",
        lease_id="l",
        source_ref="main",
        source_head=ref.source_head,
        branch=ref.branch,
        worktree_path=f"{ref.worktree_path}-elsewhere",
        handoff="pending",
    )
    session.record_workspace(newer)

    done = session_handoff.settle(session, repo)

    assert done is not None and done.outcome == "done"
    assert owned(engine)[-1] == newer


def test_a_git_failure_partway_is_settled_by_what_landed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A `HandoffError` after the intent is not taken as "nothing changed": the discard
    removed the worktree, then failed, and the completion says what the repository shows."""
    engine, _ = two_calls(monkeypatch, repo, raw)
    engines.append(engine)
    real = handoff_finalize

    def half(*args: Any, **kwargs: Any) -> HandoffResult:
        real(*args, **kwargs)
        raise HandoffError("git branch -D failed")

    monkeypatch.setattr(session_handoff, "finalize", half)
    with pytest.raises(HandoffError):
        engine.finalize(SID, "discard")
    (done,) = completions(engine)
    assert (done.outcome, done.reconciled) == ("done", True), "the branch is gone: it landed"
    assert owned(engine)[-1].handoff == "discard"


def test_uncommitted_work_is_named_exactly_and_said_in_the_summary(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Porcelain's first column may be a space (` M app.py`): the name is kept whole."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    (Path(ref.worktree_path) / "app.py").write_text("unsaved\n")

    summary = engine.handoff(SID)

    assert summary.git is not None and summary.git.uncommitted == (" M app.py",)
    assert "discard" not in summary.refused
    assert any("no choice lands: " in u and "app.py" in u for u in summary.notes)


def test_a_gone_branch_still_answers_a_summary(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None and ref.branch is not None
    git(repo, "worktree", "remove", "--force", ref.worktree_path)
    git(repo, "branch", "-D", ref.branch)

    summary = engine.handoff(SID)

    assert summary.git is None and len(summary.episodes) == 1
    assert any("cannot be described" in u for u in summary.notes)


def test_nothing_starts_while_a_handoff_is_under_way(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    engine._handing_off.add(SID)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(EngineBusy, match="handing off"):
        engine.coder(one_task(), session_id=SID)
    with pytest.raises(EngineBusy, match="handing off"):
        engine.finalize(SID, "keep")


def test_in_place_nothing_is_committed_and_the_message_is_offered(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """An in-place run commits nothing, and the handoff offers the
    message for the person's own commit: each accepted task, its goal and what ran."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    head = git(repo, "rev-parse", "HEAD")
    model.then(*solved("return 1", "return 2"))
    handle = engine.coder(one_task(workspace="in_place"), session_id=SID)
    assert handle.result().outcome == "COMPLETED"

    summary = engine.handoff(SID)
    assert git(repo, "rev-parse", "HEAD") == head, "nothing committed in place"
    assert summary.git is None
    assert summary.message.startswith(
        "chickadee: 1 commit from your checkout\n\n- chickadee: T1 "
    )
    assert "Ran:" in summary.message and "Chickadee-Operation" not in summary.message


def _left(engine: PacketEngine, ref: WorkspaceRef) -> None:
    """`worktree leave`: the checkout's line after the worktree's."""
    session_of(engine).record_workspace(
        WorkspaceRef(
            mode="in_place",
            lease_id=ref.lease_id,
            source_ref=ref.source_ref,
            source_head=ref.source_head,
            branch=None,
            worktree_path=None,
            handoff="pending",
        )
    )


def test_a_crash_after_landing_a_left_worktree_is_settled_as_landed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Review of 2026-10-25: `settle` read the line at the intent, which after a leave names
    the checkout, so a landed worktree was settled `unknown` and stayed pending."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    _left(engine, ref)
    real = handoff_finalize

    def killed(*args: Any, **kwargs: Any) -> HandoffResult:
        real(*args, **kwargs)
        raise InfraError("the process died after the merge")

    monkeypatch.setattr(session_handoff, "finalize", killed)
    with pytest.raises(InfraError):
        engine.finalize(SID, "apply")
    monkeypatch.setattr(session_handoff, "finalize", lambda *_a, **_k: pytest.fail("repeated"))
    engine.resume(SID)

    (done,) = completions(engine)
    assert (done.outcome, done.reconciled) == ("done", True)
    assert owned(engine)[-1].handoff == "apply"


def test_keeping_a_left_worktree_leaves_the_session_in_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Review of 2026-10-25: `keep` wrote the worktree's line last, so a session that had left
    its worktree was back in it, and its next in-place work went there."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    _left(engine, ref)

    engine.finalize(SID, "keep")

    here = latest_workspace(state(engine, SID))
    assert here is not None and here.mode == "in_place"
    held = held_worktree(state(engine, SID))
    assert held is not None and held.handoff == "keep"
