"""Leaving pauses the run, and resume asks again what nobody answered.

Through the engine and its real session log: a run whose worker waits on a command's permission
question when the engine closes is not ended (no `run_end`), its question is closed as
withdrawn and never as the deny its wait fell back to, and close does not wait out the
question's timeout. A later engine's `resume()` reads the session log as reopened, and the same
command is put to the person again."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, athena_accepts, box_call, build, make_repo
from engines import answer, call, calls
from pydantic import TypeAdapter
from zeus_turn_helpers import WAIT_S

from chickadee.context.questions import CANCELLED, WITHDRAWN, QuestionAnsweredLine
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import NonInteractive, PacketEngine, ScriptedInteraction
from chickadee.schema.interaction import ActionApproval, ApprovalDecision
from chickadee.state.roots import roots as resolve_roots

BRIEF = "## Goal\n`go()` returns 2.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


class _LeavesAtTheQuestion(NonInteractive):
    """A person who is asked about a command, and leaves without answering."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = threading.Event()

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self.asked.set()
        assert cancel is not None, "the ask is handed the run's stop"
        cancel.wait(30)
        return ApprovalDecision(action_id=approval.action_id, verdict="deny", source="default")


def _answers(repo: Path) -> list[QuestionAnsweredLine]:
    log = SessionLog.read(sessions_root(resolve_roots(repo)), SID)
    adapter = TypeAdapter(QuestionAnsweredLine)
    return [
        adapter.validate_json(log.payload(e))
        for e in log.events()
        if e.kind == "question_answered"
    ]


def _told_zeus(repo: Path) -> list[str]:
    """The notices on the session log, as Zeus will read them."""
    log = SessionLog.read(sessions_root(resolve_roots(repo)), SID)
    told: list[str] = []
    for e in log.events():
        if e.kind == "turn":
            line = json.loads(log.payload(e))
            if line.get("role") == "harness":
                told.append(line["text"])
    return told


def _resumed_told(repo: Path) -> list[str]:
    return [t for t in _told_zeus(repo) if "resumed after a pause" in t]


def test_leaving_at_a_permission_question_pauses_and_resume_asks_it_again(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The reported run: Zeus delegates, the worker asks to run a command, the person leaves,
    and comes back with `--continue`."""
    person = _LeavesAtTheQuestion()
    engine, _ = build(
        monkeypatch,
        repo,
        dict(raw),
        [
            calls(call("coder", "c1", title="Return 2", brief=BRIEF, workspace="worktree")),
            calls(box_call("shell", "w1", command="git checkout -- app.py")),
        ],
        interaction=person,
    )
    engine.turn(SID, "make go return 2")
    assert person.asked.wait(30), "the worker's command was put to the person"

    started = time.monotonic()
    engine.close()
    assert time.monotonic() - started < 10, "close never waits out the question's timeout"

    [withdrawn] = [a for a in _answers(repo) if a.id.startswith("shell-")]
    assert (withdrawn.source, withdrawn.note, withdrawn.choice) == (WITHDRAWN, CANCELLED, "")
    back = ScriptedInteraction(actions=["approve"])
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [answer("T1: ran it."), *athena_accepts(), answer("Noted."), answer("Still noted.")],
        interaction=back,
    )
    engines.append(engine)
    resumed = engine.resume(SID)

    # Found unfinished: the pause wrote no `run_end`.
    assert resumed.in_flight is not None, "paused, not ended: resume continues it"
    result = resumed.in_flight.result()
    assert result.outcome == "COMPLETED", result.why
    [again] = back.approvals
    assert "git checkout -- app.py" in again.display, "the same command, asked again"

    # Zeus's `coder` call was answered PAUSED: he hears how the run ended as his next turn
    # starts, once.
    assert _resumed_told(repo) == [], "never mid-turn: told at the next turn's start"
    engine.turn(SID, "how did it go?").wait(timeout=WAIT_S)
    [told] = _resumed_told(repo)
    assert "ended COMPLETED" in told
    engine.turn(SID, "and now?").wait(timeout=WAIT_S)
    assert len(_resumed_told(repo)) == 1, "told once"
