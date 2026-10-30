"""Undo by turn, redo, and task and operation undo, through the facade.

Each workspace mode end to end: a worktree before the handoff, after `apply`, after `squash`,
and in place. In place never commits. Undo then redo returns the tree byte for byte. A conflict
in an unattended session keeps the person's version and says so, and what reached past the
code is named as not undone."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_ignoring_repo, make_repo, solved
from repos import git
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

import chickadee.engine as engine_module
from chickadee.context.project import JournalRecords
from chickadee.context.rewind import taken
from chickadee.context.session import Session
from chickadee.engine import PacketEngine, UndoReport
from chickadee.errors import SessionError
from chickadee.handoff import finalize
from chickadee.journal.projection import Operation, RunProjection
from chickadee.schema.isolation import GitHandoffChoice, IsolationSpec
from chickadee.schema.journal import EffectIntendedP
from chickadee.undo import undo as undo_turns
from chickadee.workspace import snapshot_operation


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository that ignores what running its tests leaves behind, as real ones do: the
    run's checks write `__pycache__`, and an unignored cache is the harness's to undo too."""
    return make_ignoring_repo(tmp_path / "repo")


def live(engine: PacketEngine) -> Session:
    conversation = engine._conversations.get(SID)  # pyright: ignore[reportPrivateUsage]
    opened = conversation.opened() if conversation is not None else None
    assert opened is not None
    return opened.session


def one_edit(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    workspace: str,
    *,
    extra: list[Any] | None = None,
) -> tuple[PacketEngine, int]:
    """One turn whose `coder` call changes `return 1` to `return 2`; the engine and the turn."""
    work = {
        "title": "Return 2",
        "brief": "## Goal\n`go()` returns 2.\n",
        "workspace": workspace,
    }
    script = [
        calls(call("coder", "c1", **work)),
        *solved("return 1", "return 2"),
        answer("changed"),
        *(extra or []),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engine.turn(SID, "change go()\nplease").wait(timeout=WAIT_S)
    turn = next(t.seq for t in state(engine, SID).turns() if t.role == "user")
    return engine, turn


def worktree_of(engine: PacketEngine) -> Path:
    (line,) = {s.root for s in taken(live(engine))}
    return Path(line)


def read(root: Path) -> bytes:
    return (root / "app.py").read_bytes()


# -- in place ---------------------------------------------------------------------------------


def test_in_place_undo_restores_and_never_commits_and_redo_returns_the_tree(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "in_place")
    try:
        changed = read(repo)
        assert b"return 2" in changed
        head = git(repo, "rev-parse", "HEAD")
        (offered,) = engine.undoable(SID)
        assert offered.turn == turn and offered.options == ("both", "conversation", "code")
        assert offered.prompt == "change go()"
        report = engine.undo(SID, turn, code=True, conversation=True, commit=True)
        (done,) = report.workspaces
        assert done.where == "in_place" and done.restored == ("app.py",)
        assert done.committed is None, "in place never commits (D39)"
        assert git(repo, "rev-parse", "HEAD") == head
        assert b"return 1" in read(repo)
        assert report.prompt == "change go()\nplease", "turn N's words come back for editing"
        assert state(engine, SID).turn(turn) is None, "the conversation is hidden"
        again = engine.redo(SID)
        assert again.code and again.conversation
        assert read(repo) == changed, "undo then redo returns the tree byte for byte"
        assert state(engine, SID).turn(turn) is not None
    finally:
        engine.close()


@pytest.mark.writes_worktree("app.py")
def test_an_unattended_conflict_keeps_the_persons_version_and_says_so(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "in_place")
    try:
        mine = read(repo).replace(b"return 2", b"return 3  # mine")
        (repo / "app.py").write_bytes(mine)
        report = engine.undo(SID, turn, code=True, conversation=False)
        (done,) = report.workspaces
        assert done.kept == ("app.py",) and done.kept_by_default == ("app.py",)
        assert done.restored == ()
        assert read(repo) == mine
    finally:
        engine.close()


def test_a_conversation_only_undo_keeps_the_code_and_names_it(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "in_place")
    try:
        report = engine.undo(SID, turn, code=False, conversation=True)
        assert report.workspaces == ()
        assert b"return 2" in read(repo)
        notices = [t.text for t in state(engine, SID).turns() if t.role == "harness"]
        assert any(
            n.startswith("code changed in turns you no longer see: app.py") for n in notices
        )
    finally:
        engine.close()


# -- worktree before the handoff --------------------------------------------------------------


def test_before_the_handoff_the_branch_moves_back_and_redo_moves_it_forward(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "worktree")
    try:
        wt = worktree_of(engine)
        tip = git(wt, "rev-parse", "HEAD")
        source = git(repo, "rev-parse", "HEAD")
        (before,) = [s for s in taken(live(engine)) if s.phase == "before"]
        changed = read(wt)
        report = engine.undo(SID, turn, code=True, conversation=False)
        (done,) = report.workspaces
        assert done.where == "worktree" and done.restored == ("app.py",)
        assert git(wt, "rev-parse", "HEAD") == before.head, "the harness's branch moved back"
        assert git(repo, "rev-parse", "HEAD") == source, "the person's branch is untouched"
        assert b"return 1" in read(wt)
        assert git(wt, "status", "--porcelain") == "", "index and files agree after the move"
        engine.redo(SID)
        assert git(wt, "rev-parse", "HEAD") == tip
        assert read(wt) == changed
    finally:
        engine.close()


# -- worktree after the handoff ---------------------------------------------------------------


def handed_off(engine: PacketEngine, repo: Path, choice: GitHandoffChoice) -> None:
    wt = worktree_of(engine)
    (run,) = [e.run_id for e in state(engine, SID).episodes()]
    branch = git(wt, "branch", "--show-current")
    spec = IsolationSpec(mode="worktree", lease_id="L", branch=branch, worktree_path=str(wt))
    base = git(repo, "rev-parse", "HEAD")
    finalize(repo, spec, choice, operation_id=f"handoff-{run}", base=base)


@pytest.mark.parametrize(
    "choice", [GitHandoffChoice.APPLY, GitHandoffChoice.SQUASH, GitHandoffChoice.BRING]
)
def test_after_the_handoff_the_checkout_is_restored_uncommitted_by_default(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], choice: GitHandoffChoice
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "worktree")
    try:
        handed_off(engine, repo, choice)
        head = git(repo, "rev-parse", "HEAD")
        assert b"return 2" in read(repo)
        report = engine.undo(SID, turn, code=True, conversation=False)
        (done,) = report.workspaces
        assert done.where == "checkout" and done.committed is None
        assert b"return 1" in read(repo)
        assert git(repo, "rev-parse", "HEAD") == head, "the person's branch is never rewritten"
    finally:
        engine.close()


def test_after_apply_commit_makes_one_undo_commit(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "worktree")
    try:
        handed_off(engine, repo, GitHandoffChoice.APPLY)
        head = git(repo, "rev-parse", "HEAD")
        report = engine.undo(SID, turn, code=True, conversation=False, commit=True)
        (done,) = report.workspaces
        assert done.committed == git(repo, "rev-parse", "HEAD")
        assert git(repo, "rev-parse", "HEAD~1") == head, "one commit on top, nothing rewritten"
        assert git(repo, "log", "-1", "--format=%s").startswith(f"Undo turns {turn}-{turn}")
        assert git(repo, "status", "--porcelain", "--", "app.py") == ""
    finally:
        engine.close()


# -- refusals ---------------------------------------------------------------------------------


def test_the_code_options_go_when_a_snapshot_is_gone(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "in_place")
    try:
        for shot in taken(live(engine)):
            git(repo, "update-ref", "-d", shot.ref)
        (offered,) = engine.undoable(SID)
        assert offered.options == ("conversation",)
        assert offered.no_code is not None and "removed by retention" in offered.no_code
        with pytest.raises(SessionError, match="removed by retention"):
            engine.undo(SID, turn, code=True, conversation=False)
    finally:
        engine.close()


def test_an_undo_names_a_user_turn(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, turn = one_edit(monkeypatch, repo, raw, "in_place")
    try:
        with pytest.raises(SessionError, match="not a user turn"):
            engine.undo(SID, turn + 1, code=True, conversation=True)
        with pytest.raises(SessionError, match="the code, the conversation, or both"):
            engine.undo(SID, turn, code=False, conversation=False)
    finally:
        engine.close()


# -- what could not be undone (decision 3) ----------------------------------------------------


class NoRuns:
    """`RunRecords` for a session that dispatched no run."""

    def projection(self, run_id: str) -> RunProjection:
        raise FileNotFoundError(run_id)

    def blob(self, run_id: str, sha: str) -> bytes:
        raise FileNotFoundError(sha)


def test_an_action_that_reached_outside_the_worktree_is_named(tmp_path: Path) -> None:
    from chickadee.config import UndoPolicy
    from chickadee.engine import NonInteractive
    from chickadee.schema import ToolCall

    repo = make_repo(tmp_path / "repo")
    with Session.create(tmp_path / "sessions", "S") as session:
        turn = session.record_user("deploy it").seq
        command = {"command": "curl -X POST https://example.com", "network": ["example.com"]}
        session.record_assistant("", [ToolCall(id="s1", name="shell", arguments=command)])
        session.record_tool("s1", "ok")
        report: UndoReport = undo_turns(
            session,
            NoRuns(),
            repo,
            UndoPolicy(),
            NonInteractive(),
            to_turn=turn,
            code=False,
            conversation=True,
        )
    assert [(e.kind, e.detail) for e in report.not_undone] == [
        ("network", "curl -X POST https://example.com")
    ]


# -- task and operation undo (decision 12) ----------------------------------------------------


def test_a_task_undo_moves_the_run_back_before_the_task(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, _ = one_edit(monkeypatch, repo, raw, "worktree")
    try:
        wt = worktree_of(engine)
        (run,) = [e.run_id for e in state(engine, SID).episodes()]
        where = engine._where(SID)  # pyright: ignore[reportPrivateUsage]
        _, runs = engine._caches_for(where)  # pyright: ignore[reportPrivateUsage]
        projection = JournalRecords(where, runs).projection(run)
        (task,) = {
            op.task_id
            for op in projection.operations.values()
            if op.task_id
            and op.intended is not None
            and op.intended.effect_kind == "task_commit"
        }
        before = git(wt, "rev-parse", "HEAD~1")
        done = engine.undo_task(SID, run, task)
        assert done.restored == ("app.py",)
        assert git(wt, "rev-parse", "HEAD") == before
        assert b"return 1" in read(wt)
    finally:
        engine.close()


def test_an_operation_undo_restores_its_declared_paths(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """An action's snapshot ref, taken before it ran; the journal names its surface."""
    engine, _ = one_edit(monkeypatch, repo, raw, "worktree")
    try:
        wt = worktree_of(engine)
        (run,) = [e.run_id for e in state(engine, SID).episodes()]
        op = "shell:T9:1:s1"
        snapshot_operation(wt, op)
        (wt / "out.txt").write_text("written by the action\n", encoding="utf-8")
        intended = EffectIntendedP(
            operation_id=op,
            effect_kind="shell_write",
            approved_sha="a" * 64,
            preconditions=(),
            idempotent=False,
            declared_surface=("out.txt",),
        )

        class WithAction(JournalRecords):
            def projection(self, run_id: str) -> RunProjection:
                real = super().projection(run_id)
                if run_id != run:
                    return real
                ops = {**real.operations, op: Operation(op, intended, task_id="T9")}
                return replace(real, operations=ops)

        monkeypatch.setattr(engine_module, "JournalRecords", WithAction)
        done = engine.undo_operation(SID, op)
        assert done.restored == ("out.txt",)
        assert not (wt / "out.txt").exists()
    finally:
        engine.close()


def test_a_redo_after_a_new_turn_takes_back_the_code_and_says_why_not_the_rest(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The conversation half goes once the person sends a new turn,
    and the report says so (`withheld`) rather than leaving them to infer it."""
    engine, turn = one_edit(monkeypatch, repo, raw, "in_place", extra=[answer("again")])
    try:
        changed = read(repo)
        engine.undo(SID, turn, code=True, conversation=True)
        engine.turn(SID, "something else").wait(timeout=WAIT_S)
        again = engine.redo(SID)
        assert again.code and not again.conversation
        (why,) = again.withheld
        assert why.startswith("its conversation moved on at seq ")
        assert read(repo) == changed
    finally:
        engine.close()


def test_a_task_undo_of_a_run_the_session_never_had_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """A mistyped run id is the person's to correct, not an internal error."""
    engine, _turn = one_edit(monkeypatch, repo, raw, "in_place")
    try:
        with pytest.raises(SessionError, match="no run R-none in this session"):
            engine.undo_task(SID, "R-none", "T1")
    finally:
        engine.close()
