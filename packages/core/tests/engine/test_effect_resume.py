"""A crash mid-episode reconciles without a duplicate effect, through the engine (`T-3.8` item
7). The box's own suites kill an episode at each step
(`tests/box`); this is the same guarantee end to end: the process dies right after an effect
is recorded, a new engine resumes the session, and the task's commit exists exactly once."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import REPORT, SID, ProcessDied, athena_accepts, box_call, build, make_repo
from repos import git
from zeus_turn_helpers import WAIT_S, Answering, Step, answer, call, calls, state

import chickadee.engine as engine_module
from chickadee.context.view import run_worktree
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.journal.writer import Journal
from chickadee.schema.journal import EventKind

LIB = "def h() -> int:\n    return 5\n"
EDIT_LIB = box_call("edit", "w2", path="lib.py", old="return 5", new="return 6")
TWO_FILES = calls(
    box_call("edit", "w1", path="app.py", old="return 1", new="return 2"), EDIT_LIB
)
"""The worker's one response: both edits, each its own step and its own effect."""

AFTER_RESUME: dict[int, list[Step]] = {
    # The second edit was cut off before it ran: the worker is told so, and calls it again.
    1: [calls(EDIT_LIB), answer("T1: edited both"), *athena_accepts()],
    # Both ran; the round's reply was never given.
    2: [answer("T1: edited both"), *athena_accepts()],
    # The task's commit is made; Athena's review is answered from it, and she reports.
    3: [answer(REPORT)],
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text(LIB)
    git(repo, "add", "lib.py")
    git(repo, "commit", "-qm", "lib")
    engine_module.init_repo(repo)
    return repo


@pytest.mark.parametrize("after", [1, 2, 3])
def test_a_crash_after_an_effect_resumes_without_repeating_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    after: int,
) -> None:
    """Killed right after the `after`-th `effect_completed` line: the worker's first edit, its
    second, the task's commit. What each model is asked after the resume is scripted as the
    resumed episode asks it."""
    base = git(repo, "rev-parse", "HEAD")
    real = Journal.append
    completed = 0

    def dies_after(self: Journal, kind: EventKind, *args: Any, **kwargs: Any) -> int:
        nonlocal completed
        seq = real(self, kind, *args, **kwargs)
        if kind is EventKind.EFFECT_COMPLETED:
            completed += 1
            if completed == after:
                raise ProcessDied
        return seq

    script = [
        calls(
            call(
                "coder",
                "c1",
                title="Edit both",
                brief="## Goal\nBoth files.\n",
                workspace="worktree",
            )
        ),
        TWO_FILES,
        answer("T1: edited both"),
        *athena_accepts(),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.interaction = Answering("yes")
    monkeypatch.setattr(Journal, "append", dies_after)
    with pytest.raises(SessionError):
        engine.turn(SID, "edit both").wait(timeout=WAIT_S)
    (episode,) = state(engine, SID).episodes()
    monkeypatch.setattr(Journal, "append", real)
    engine.close()

    again, _ = build(monkeypatch, repo, raw, AFTER_RESUME[after])
    engines.append(again)
    resumed = again.resume(SID)
    assert resumed.reconciled == (episode.run_id,)
    handle = resumed.in_flight
    assert handle is not None
    result = handle.result()

    assert (result.outcome, result.tasks_committed) == ("COMPLETED", 1)
    worktree = run_worktree(state(again, SID), episode.run_id)
    assert worktree is not None, "the run wrote in the session's worktree (D188)"
    commits = git(worktree, "rev-list", f"{base}..HEAD").split()
    assert len(commits) == 1, "one commit per task, none made twice"
    assert (worktree / "app.py").read_text().count("return 2") == 1
    assert (worktree / "lib.py").read_text() == LIB.replace("return 5", "return 6")
