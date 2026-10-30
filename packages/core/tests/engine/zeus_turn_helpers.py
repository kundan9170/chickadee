"""What the suites that drive `PacketEngine.turn()` share (`test_zeus_turn.py`,
`test_session_lifecycle.py`). The scripted rung and the engine around it are `engines.py`'s, in
`packages/testing`, re-exported here so these suites import one module.

A scripted model stands behind the zeus rung. `state()` is the one place a test reads the
session's writer through the engine's private registry, so a change to that registry changes
one line here, never an assertion.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from engines import (
    ScriptedRung,
    Step,
    answer,
    call,
    calls,
    config_for,
    start,
    with_sessions,
)

from chickadee.box.report import CoderReport, ReportHandles, stored_handle
from chickadee.client.request import ChatRequest
from chickadee.context.session import SessionState
from chickadee.engine import CoderHandle, PacketEngine, RunResult
from chickadee.journal.projection import RunSpend
from chickadee.schema.adapter import Completion
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee.workspace import IN_PLACE_KEY, SnapshotSite

SID = "S-turn"
RUN = "2026-10-01-0000000a"
WAIT_S = 20.0

__all__ = [
    "RUN",
    "SID",
    "WAIT_S",
    "Answering",
    "FakeHandle",
    "ScriptedRung",
    "Step",
    "answer",
    "appears",
    "call",
    "calls",
    "config_for",
    "held",
    "result",
    "said",
    "start",
    "state",
    "with_sessions",
]


# -- the scripted rung ------------------------------------------------------------------------


def held(gate: threading.Event, then: Completion | None = None) -> Step:
    """A response that waits for the test to release it: the model is still answering."""

    def step(_: ChatRequest) -> Completion:
        assert gate.wait(WAIT_S), "the test never released the model"
        return then or answer()

    return step


class FakeHandle(CoderHandle):
    """An episode that completes at once, and records each stop it receives. Its report is
    read from `result()`, so a subclass that slows `result()` slows the report. It runs in
    `root`, in place, which is where a turn snapshot of it is taken."""

    def __init__(self, run_id: str, root: Path) -> None:
        super().__init__(run_id)
        self.root = root
        self.interrupted: list[bool] = []
        self.abandoned = False

    @property
    def workspace(self) -> SnapshotSite:
        return SnapshotSite(self.root, "in_place", IN_PLACE_KEY)

    def result(self) -> RunResult:
        return RunResult(self.run_id, "COMPLETED", tasks_committed=1, tasks_total=1)

    def report(self) -> CoderReport:
        outcome = self.result().outcome
        return CoderReport(
            outcome="COMPLETED" if outcome == "COMPLETED" else "ABORTED",
            why=None if outcome == "COMPLETED" else f"the run ended {outcome}",
            run_id=self.run_id,
            handles=ReportHandles(journal=stored_handle("journal", self.run_id, b"{}")),
            cost=RunSpend(),
        )

    def interrupt(self, hard: bool = False) -> None:
        self.interrupted.append(hard)

    def abandon(self) -> None:
        self.abandoned = True


class Answering:
    """An `InteractionPort` whose person answers every question with `choice`, and that
    keeps what it was asked."""

    def __init__(self, choice: str) -> None:
        self.choice = choice
        self.asked: list[Question] = []

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.asked.append(q)
        return Answer(question_id=q.id, choice=self.choice, source=AnswerSource.HUMAN)

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        raise AssertionError(f"not asked to confirm {approval.display!r}")


# -- the engine -------------------------------------------------------------------------------


def state(engine: PacketEngine, sid: str = SID) -> SessionState:
    conversation = engine._conversations.get(sid)  # pyright: ignore[reportPrivateUsage]
    assert conversation is not None, f"session {sid} is not open"
    opened = conversation.opened()
    assert opened is not None, f"session {sid} is not built"
    return opened.session.state


def said(engine: PacketEngine, sid: str = SID) -> list[str]:
    return [t.text for t in state(engine, sid).turns() if t.role == "user"]


def result(engine: PacketEngine, cid: str, sid: str = SID) -> str:
    line = state(engine, sid).result(cid)
    assert line is not None, f"call {cid} has no result"
    return line.text


def appears(path: Path) -> None:
    deadline = time.monotonic() + WAIT_S
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.02)
