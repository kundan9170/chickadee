"""One worktree per session: the first `coder` call of a session in
worktree mode creates the session's worktree and records it as a `workspace` line, and every
later call reattaches it, so the second call builds on the first. The WRITE lease stays per
call. A dirty checkout at the first call is refused with an offer (decision 4.1). In place is
unchanged. The latest `workspace` line is the one the session owns: a fork's child worktree is
reattached, and after an `apply` handoff the next call opens a new one."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
from coder_helpers import SID, CoderModel, build, make_repo, one_task, solved
from repos import git
from zeus_turn_helpers import state

import chickadee.engine as engine_module
from chickadee.context.project import JournalRecords
from chickadee.context.session import Session
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.journal import paths
from chickadee.journal.projection import Operation, RunProjection
from chickadee.journal.reader import read_events
from chickadee.schema import EventKind
from chickadee.schema.journal import EffectIntendedP, RunStartP
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import (
    SEED_TRAILER,
    WORKTREES_DIR,
    acquire,
    restore_operation,
    snapshot_operation,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


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


def repo_head(repo: Path, run_id: str) -> str:
    for event in read_events(paths.journal_path(resolve_roots(repo), run_id)):
        if event.kind == EventKind.RUN_START:
            assert isinstance(event.payload, RunStartP)
            return event.payload.repo_head
    raise AssertionError(f"run {run_id} has no run_start")


def solve(
    engine: PacketEngine,
    model: CoderModel,
    old: str,
    new: str,
    workspace: Literal["worktree", "in_place"] = "worktree",
) -> str:
    """A one-task run that changes `old` to `new` in `app.py`, committed."""
    model.then(*solved(old, new))
    handle = engine.coder(one_task(workspace=workspace), session_id=SID)
    assert handle.result().outcome == "COMPLETED"
    return handle.run_id


LEASE_PROBE = """
import sys
from pathlib import Path
from chickadee.errors import RepoLeaseBusy
from chickadee.init.detect import resolve
from chickadee.init.lease import take_write_lease

try:
    take_write_lease(resolve(Path(sys.argv[1]))).release()
except RepoLeaseBusy:
    print("BUSY")
else:
    print("FREE")
"""


def lease(repo: Path) -> str:
    out = subprocess.run(
        [sys.executable, "-c", LEASE_PROBE, str(repo)], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_two_calls_in_one_session_share_one_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The owner's G3 session made a worktree per call, each from `main`, so the second edit
    could not see the first. Here the second edit's old text exists only after the first."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    start = git(repo, "rev-parse", "HEAD")

    first = solve(engine, model, "return 1", "return 2")
    assert lease(repo) == "FREE", "the WRITE lease is per call (D75)"
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    tip = git(Path(ref.worktree_path), "rev-parse", "HEAD")

    second = solve(engine, model, "return 2", "return 3")

    assert worktrees(repo) == [SID]
    assert owned(engine) == [ref], "reattaching records nothing new"
    assert (ref.mode, ref.handoff, ref.source_head) == ("worktree", "pending", start)
    assert ref.branch is not None and ref.branch.startswith("chickadee/")
    assert ref.branch.endswith(f"-{SID}"), "named for the session, not the run"
    assert repo_head(repo, first) == start
    assert repo_head(repo, second) == tip, "the second call starts at the first's commit"
    worktree = Path(ref.worktree_path)
    assert "return 3" in (worktree / "app.py").read_text()
    assert "return 1" in (repo / "app.py").read_text(), "the checkout is untouched"


@pytest.mark.writes_worktree("notes.txt")
def test_a_dirty_checkout_seeds_the_first_calls_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Was: refused with an offer to commit, stash or run in place (D188 decision 4.1). D230
    Amendment 2 decision 4 moves it: the worktree starts with the person's changes, in one
    seed commit, and the line's base is still their `HEAD`."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    start = git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text("the person's own work\n")

    solve(engine, model, "return 1", "return 2")

    (ref,) = owned(engine)
    assert ref.source_head == start
    worktree = Path(str(ref.worktree_path))
    assert (worktree / "notes.txt").read_text() == "the person's own work\n"
    seed = git(worktree, "rev-list", "--reverse", f"{start}..HEAD").split()[0]
    assert f"{SEED_TRAILER}: {SID}" in git(worktree, "log", "-1", "--format=%B", seed)
    assert (repo / "notes.txt").read_text() == "the person's own work\n"
    assert "return 1" in (repo / "app.py").read_text(), "the checkout is untouched"
    assert lease(repo) == "FREE"


@pytest.mark.writes_worktree("notes.txt")
def test_a_dirty_checkout_does_not_stop_a_later_call(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Once the worktree exists, the checkout is the person's to use, and the handoff answers
    for what overlaps (`HandoffPreview.overlapping`)."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (repo / "notes.txt").write_text("the person's own work\n")

    solve(engine, model, "return 2", "return 3")

    assert worktrees(repo) == [SID]


def test_in_place_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)

    solve(engine, model, "return 1", "return 2", workspace="in_place")

    assert "return 2" in (repo / "app.py").read_text()
    assert worktrees(repo) == []
    assert owned(engine) == []


def test_the_latest_recorded_worktree_is_the_one_reattached(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A fork records its child's worktree (`undo.fork`); the child's calls write there."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    engine.open_session(SID)
    with acquire(repo, run_id="fork-child", goal="fork") as ws:
        child = Path(ws.root)
        session_of(engine).record_workspace(
            WorkspaceRef(
                mode="worktree",
                lease_id=ws.spec.lease_id,
                source_ref="main",
                source_head=git(repo, "rev-parse", "HEAD"),
                branch=ws.spec.branch,
                worktree_path=ws.spec.worktree_path,
                handoff="pending",
            )
        )

    solve(engine, model, "return 1", "return 2")

    assert worktrees(repo) == ["fork-child"]
    assert "return 2" in (child / "app.py").read_text()


def test_after_an_apply_handoff_the_next_call_opens_a_new_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The handed-off worktree is left where it is; the session's next call
    finds no pending line, and opens its next worktree beside it, from the new `HEAD`."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    git(repo, "merge", "-q", "--ff-only", git(Path(ref.worktree_path), "rev-parse", "HEAD"))
    session_of(engine).record_workspace(replace(ref, handoff="apply"))

    solve(engine, model, "return 2", "return 3")

    assert worktrees(repo) == [SID, f"{SID}-2"]
    latest = owned(engine)[-1]
    assert (latest.handoff, latest.source_head) == ("pending", git(repo, "rev-parse", "HEAD"))
    assert latest.branch != ref.branch


def test_a_gone_worktree_whose_branch_remains_is_added_back_from_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The line is a reference, and the directory may be gone (`WorkspaceRef`). The branch
    still holds the session's work, so the next call builds on it at the same path, rather
    than starting again from `HEAD` and leaving that work behind."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    first = solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    tip = git(Path(ref.worktree_path), "rev-parse", "HEAD")
    git(repo, "worktree", "remove", "--force", ref.worktree_path)

    second = solve(engine, model, "return 2", "return 3")

    assert worktrees(repo) == [SID]
    assert owned(engine) == [ref], "the same worktree, so nothing new is recorded"
    assert repo_head(repo, second) == tip
    assert repo_head(repo, first) != tip
    assert "return 3" in (Path(ref.worktree_path) / "app.py").read_text()


def test_a_gone_worktree_and_branch_are_replaced(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """With the branch gone too, nothing of the worktree is left: a call opens a new one
    rather than failing to reattach what is not there."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    solve(engine, model, "return 1", "return 2")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None and ref.branch is not None
    git(repo, "worktree", "remove", "--force", ref.worktree_path)
    git(repo, "branch", "-D", ref.branch)

    solve(engine, model, "return 1", "return 2")

    assert worktrees(repo) == [SID]
    latest = owned(engine)[-1]
    assert len(owned(engine)) == 2, "a new worktree is recorded"
    assert latest.source_head == git(repo, "rev-parse", "HEAD")


def test_a_worktree_the_session_could_not_record_is_removed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Nothing names a worktree whose `workspace` line failed to write, so nothing would ever
    reattach or hand it off: it is removed with its branch, and the lease is free."""
    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    engine.open_session(SID)

    def fails(ref: WorkspaceRef) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(session_of(engine), "record_workspace", fails)

    with pytest.raises(OSError, match="disk full"):
        engine.coder(one_task(), session_id=SID).result()

    assert worktrees(repo) == []
    assert git(repo, "branch", "--list", "chickadee/*") == ""
    assert lease(repo) == "FREE"


# -- undo over the session's shared branch --------------------------------------------------


def task_of(engine: PacketEngine, run_id: str) -> str:
    """The one task run `run_id` committed."""
    where = engine._where(SID)  # pyright: ignore[reportPrivateUsage]
    _, runs = engine._caches_for(where)  # pyright: ignore[reportPrivateUsage]
    projection = JournalRecords(where, runs).projection(run_id)
    (task,) = {
        op.task_id
        for op in projection.operations.values()
        if op.task_id and op.intended is not None and op.intended.effect_kind == "task_commit"
    }
    return task


def test_a_task_undo_takes_back_its_own_run_not_a_later_one(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Both runs commit their task under the same operation id, since ids repeat across
    runs. Undoing the second run's task takes back the second run's commit, and then the
    first run's task can be undone too, and takes back the first's."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    first = solve(engine, model, "return 1", "return 2")
    second = solve(engine, model, "return 2", "return 3")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    worktree = Path(ref.worktree_path)

    engine.undo_task(SID, second, task_of(engine, second))
    assert "return 2" in (worktree / "app.py").read_text()
    assert git(worktree, "rev-parse", "HEAD") == repo_head(repo, second)

    engine.undo_task(SID, first, task_of(engine, first))
    assert "return 1" in (worktree / "app.py").read_text()
    assert git(worktree, "rev-parse", "HEAD") == repo_head(repo, first)


def test_a_task_undo_under_a_later_runs_commits_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Moving the shared branch back before the first run's task would take the second
    run's commit with it, and the first run's operation id would find the second's commit:
    refused by name, with nothing moved."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    first = solve(engine, model, "return 1", "return 2")
    second = solve(engine, model, "return 2", "return 3")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    worktree = Path(ref.worktree_path)
    tip = git(worktree, "rev-parse", "HEAD")

    with pytest.raises(SessionError, match=second):
        engine.undo_task(SID, first, task_of(engine, first))

    assert git(worktree, "rev-parse", "HEAD") == tip
    assert "return 3" in (worktree / "app.py").read_text()


def test_operation_snapshots_of_two_runs_on_one_branch_are_their_own(repo: Path) -> None:
    """Operation ids repeat across runs and a session's runs share a branch, so the refs are
    keyed by run: the second run's snapshot is taken, not refused, and each restores its own.
    A ref taken before refs were keyed by run is still found by its id alone."""
    with acquire(repo, run_id="ops", goal="ops") as ws:
        root = Path(ws.root)
        op = "shell:T9:1:s1"
        snapshot_operation(root, op, run_id="R-one")
        (root / "out.txt").write_text("the first run's\n")
        snapshot_operation(root, op, run_id="R-two")
        (root / "out.txt").write_text("the second run's\n")

        restore_operation(root, op, ("out.txt",), run_id="R-two")
        assert (root / "out.txt").read_text() == "the first run's\n"
        restore_operation(root, op, ("out.txt",), run_id="R-one")
        assert not (root / "out.txt").exists()

        snapshot_operation(root, "shell:T8:1:s1")
        (root / "old.txt").write_text("made\n")
        restore_operation(root, "shell:T8:1:s1", ("old.txt",), run_id="R-old")
        assert not (root / "old.txt").exists()


def test_an_operation_id_two_runs_journaled_is_undone_by_naming_the_run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Operation ids repeat across runs: two runs of one session each journal
    `shell:T9:1:s1`. Without a run the undo is refused, naming both, rather than taking
    the first; with one it restores that run's snapshot; a run not of this session is
    refused."""
    engine, model = build(monkeypatch, repo, raw)
    engines.append(engine)
    first = solve(engine, model, "return 1", "return 2")
    second = solve(engine, model, "return 2", "return 3")
    (ref,) = owned(engine)
    assert ref.worktree_path is not None
    worktree = Path(ref.worktree_path)
    op = "shell:T9:1:s1"
    snapshot_operation(worktree, op, run_id=first)
    (worktree / "out.txt").write_text("the first run's\n")
    snapshot_operation(worktree, op, run_id=second)
    (worktree / "out.txt").write_text("the second run's\n")
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
            ops = {**real.operations, op: Operation(op, intended, task_id="T9")}
            return replace(real, operations=ops)

    monkeypatch.setattr(engine_module, "JournalRecords", WithAction)

    with pytest.raises(SessionError, match="more than one run") as refused:
        engine.undo_operation(SID, op)
    assert first in str(refused.value) and second in str(refused.value)
    assert (worktree / "out.txt").read_text() == "the second run's\n", "nothing restored"

    with pytest.raises(SessionError, match="no run R-none in this session"):
        engine.undo_operation(SID, op, "R-none")

    done = engine.undo_operation(SID, op, second)
    assert done.restored == ("out.txt",)
    assert (worktree / "out.txt").read_text() == "the first run's\n"
    engine.undo_operation(SID, op, first)
    assert not (worktree / "out.txt").exists()
