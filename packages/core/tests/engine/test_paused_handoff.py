"""A paused run is never handed off from under itself.

One Ctrl-C pauses a run at a task boundary with no `run_end`, and `/resume` continues it in its
worktree. Applying, squashing or discarding that worktree first closed it under the run: the
resume then committed where no handoff reached, and every later `/handoff` found no worktree.
The four choices are the completion menu: while a run is paused only `keep` is offered.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, athena_accepts, build, make_ignoring_repo, worker_edit
from repos import git
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

import chickadee.session_handoff as session_handoff
from chickadee.box.desk import TaskDesk
from chickadee.client.request import ChatRequest
from chickadee.context.session import Session
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.engine import PacketEngine
from chickadee.errors import HandoffError, SessionError
from chickadee.schema.adapter import Completion
from chickadee.session_handoff import HANDOFF_COMPLETED, HANDOFF_INTENDED


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_ignoring_repo(tmp_path / "repo")


def stopped(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    *,
    hard: bool,
    workspace: str = "worktree",
    before: WorkspaceRef | None = None,
) -> PacketEngine:
    """Zeus delegates one task to a worktree and the person stops it. Once (`hard=False`), at
    the task boundary before the worker starts, pauses the run; twice (`hard=True`), while the
    worker answers, ends it `ABORTED`. `before` is a `workspace` line the session already holds
    when the turn starts."""
    gate, at = threading.Event(), threading.Event()
    edit = worker_edit("return 1", "return 2")
    first = edit[0]
    assert not callable(first)

    def worker(_: ChatRequest) -> Completion:
        """The worker's first response, held: it is still answering."""
        at.set()
        assert gate.wait(WAIT_S), "the test never stopped the run"
        return first

    boundary = TaskDesk.paused_before_start

    def at_boundary(desk: TaskDesk) -> bool:
        if not hard:
            at.set()
            assert gate.wait(WAIT_S), "the test never stopped the run"
        return boundary(desk)

    monkeypatch.setattr(TaskDesk, "paused_before_start", at_boundary)
    work = {
        "title": "Return 2",
        "brief": "## Goal\n`go()` returns 2.\n",
        "workspace": workspace,
    }
    script = [
        calls(call("coder", "c1", **work)),
        worker,
        edit[1],
        *athena_accepts(),
        answer("done"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    if before is not None:
        _session(engine).record_workspace(before)
    handle = engine.turn(SID, "make go return 2")
    assert at.wait(WAIT_S), "the run never reached the point it is stopped at"
    engine.interrupt(SID, hard=hard)
    gate.set()
    handle.wait(timeout=WAIT_S)
    return engine


def _session(engine: PacketEngine) -> Session:
    engine.open_session(SID)
    opened = engine._conversations.open(SID).opened()  # pyright: ignore[reportPrivateUsage]
    assert opened is not None
    return opened.session


def _applied(repo: Path) -> WorkspaceRef:
    """An earlier worktree of the session's, applied."""
    return WorkspaceRef(
        mode="worktree",
        lease_id="earlier",
        source_ref="main",
        source_head=git(repo, "rev-parse", "HEAD"),
        branch="chickadee/earlier",
        worktree_path=str(repo.parent / "earlier"),
        handoff="apply",
    )


def handoff_lines(engine: PacketEngine) -> list[str]:
    return [
        e.kind
        for e, _ in state(engine, SID).lines()
        if e.kind in (HANDOFF_INTENDED, HANDOFF_COMPLETED)
    ]


def test_a_paused_run_refuses_every_choice_but_keep_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine = stopped(monkeypatch, repo, raw, engines, hard=False)
    head = git(repo, "rev-parse", "HEAD")

    summary = engine.handoff(SID)

    assert set(summary.refused) == {"apply", "squash", "discard", "bring"}, (
        "keep is still offered"
    )
    assert "is paused, not finished: /resume" in summary.refused["apply"]
    for choice in ("apply", "squash", "discard", "bring"):
        with pytest.raises(HandoffError, match="is paused, not finished"):
            engine.finalize(SID, choice)
    assert handoff_lines(engine) == [], "a refusal writes no intent"
    assert git(repo, "rev-parse", "HEAD") == head


def test_keep_leaves_the_paused_runs_worktree_pending_for_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine = stopped(monkeypatch, repo, raw, engines, hard=False)

    result = engine.finalize(SID, "keep")

    assert result.choice == "keep"
    summary = engine.handoff(SID)
    assert summary.workspace is not None and summary.workspace.handoff == "keep"
    assert "apply" in summary.refused, "still paused, still refused"


def test_a_run_stopped_for_good_can_be_handed_off(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine = stopped(monkeypatch, repo, raw, engines, hard=True)

    summary = engine.handoff(SID)

    assert "apply" not in summary.refused, "an ABORTED end is an end"
    assert engine.finalize(SID, "apply").choice == "apply"


@pytest.mark.parametrize("choice", ["apply", "discard"])
def test_resume_refuses_a_run_whose_worktree_was_handed_off(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    choice: str,
) -> None:
    """A session written before the refusal can hold such a handoff; resuming it would commit
    in a worktree no handoff reaches."""
    engine = stopped(monkeypatch, repo, raw, engines, hard=False)
    with monkeypatch.context() as before_the_fix:
        before_the_fix.setattr(session_handoff, "paused_run", lambda *_: None)
        engine.finalize(SID, choice)

    with pytest.raises(SessionError, match="cannot be continued") as raised:
        engine.resume(SID)

    branch = "removed" if choice == "discard" else "git merge chickadee/"
    assert branch in str(raised.value)


def test_resume_continues_an_in_place_run_whatever_worktree_was_handed_off_before(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Review of 2026-10-25: the check read the session's last worktree line, whichever run
    wrote in it, so a paused in-place run could not be continued once
    any earlier worktree of the session had been applied."""
    engine = stopped(monkeypatch, repo, raw, engines, hard=False, workspace="in_place")
    _session(engine).record_workspace(_applied(repo))

    resumed = engine.resume(SID)

    handle = resumed.in_flight
    assert handle is not None
    assert handle.result().outcome == "COMPLETED"


def test_resume_continues_an_in_place_run_that_started_after_a_worktree_was_applied(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Final review of 2026-10-25: an in-place run writes no `workspace` line, so the line at
    its episode was the applied worktree's, read as the one it wrote in."""
    engine = stopped(
        monkeypatch, repo, raw, engines, hard=False, workspace="in_place", before=_applied(repo)
    )

    resumed = engine.resume(SID)

    handle = resumed.in_flight
    assert handle is not None
    assert handle.result().outcome == "COMPLETED"
