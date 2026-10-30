"""`bring` through the engine: "bring it back" leaves the coder's work in the person's checkout,
uncommitted, for them to review; `/handoff bring` does the same; a crash after it is settled
by what the checkout holds, never repeated.

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

import chickadee.session_handoff as session_handoff
from chickadee.context.view import latest_workspace
from chickadee.engine import PacketEngine
from chickadee.errors import InfraError
from chickadee.handoff import HandoffResult
from chickadee.handoff import finalize as handoff_finalize

ONE_TASK = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_ignoring_repo(tmp_path / "repo")


def _worked_in_a_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    *then: Any,
    person: Any = None,
) -> PacketEngine:
    script = [
        calls(call("worktree", "w1", action="enter")),
        calls(call("coder", "c1", **ONE_TASK)),
        *solved("return 1", "return 2"),
        *then,
        answer("done"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script, interaction=person or Approving())
    engines.append(engine)
    engine.turn(SID, "try it in a worktree, then bring it back").wait(timeout=WAIT_S)
    return engine


def test_bring_it_back_leaves_the_work_uncommitted_in_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Question 5's answer, now true: `land` with no `how` brings, asking once."""
    head = git(repo, "rev-parse", "HEAD")
    person = Approving()
    engine = _worked_in_a_worktree(
        monkeypatch,
        repo,
        raw,
        engines,
        calls(call("worktree", "w2", action="land")),
        person=person,
    )

    assert result(engine, "w2", SID).startswith("landed")
    assert "return 2" in (repo / "app.py").read_text(), "the work is in the checkout"
    assert git(repo, "rev-parse", "HEAD") == head, "nothing committed on the person's branch"
    assert git(repo, "diff", "--name-only") == "app.py", "changed in the working tree"
    assert git(repo, "diff", "--cached", "--name-only") == "", "and not staged"
    (asked,) = person.shown
    assert "uncommitted" in asked.display
    here = latest_workspace(state(engine, SID))
    assert here is not None and here.handoff == "bring"


def test_handoff_bring_from_the_person(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine = _worked_in_a_worktree(monkeypatch, repo, raw, engines)
    head = git(repo, "rev-parse", "HEAD")

    summary = engine.handoff(SID)
    assert "bring" not in (summary.refused or {})
    result = engine.finalize(SID, "bring")

    assert result.choice == "bring" and result.commits_applied == ()
    assert "return 2" in (repo / "app.py").read_text()
    assert git(repo, "rev-parse", "HEAD") == head


@pytest.mark.writes_worktree("app.py")
def test_a_crash_after_bring_is_settled_as_landed_and_never_repeated(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine = _worked_in_a_worktree(monkeypatch, repo, raw, engines)
    real = handoff_finalize

    def killed(*args: Any, **kwargs: Any) -> HandoffResult:
        real(*args, **kwargs)
        raise InfraError("the process died after the bring")

    monkeypatch.setattr(session_handoff, "finalize", killed)
    with pytest.raises(InfraError):
        engine.finalize(SID, "bring")
    monkeypatch.setattr(session_handoff, "finalize", lambda *_a, **_k: pytest.fail("repeated"))
    engine.resume(SID)

    here = latest_workspace(state(engine, SID))
    assert here is not None and here.handoff == "bring"
    assert "return 2" in (repo / "app.py").read_text()
