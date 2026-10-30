"""`tell`: Zeus passes on what the person said
while the team works, citing their turn; the worker's round is held at its next tool boundary,
Athena reads the round so far and the message, decides what the task needs and replies, and
the reply reaches Zeus as a trigger-10 notice. Through the engine, its real session log, box,
journal and git, the models scripted: Zeus on one script, the team on its own, since they run
at once."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    Approving,
    CoderModel,
    athena_accepts,
    athena_plans_folder,
    box_call,
    build,
    make_repo,
    worker_edit,
    zeus_designs,
)
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

import chickadee.delegation as delegation_module
import chickadee.engine as engine_module
from chickadee.client.request import ChatRequest
from chickadee.context.standing import NoticeTrigger
from chickadee.engine import PacketEngine
from chickadee.journal.projection import RunProjection
from chickadee.journal.runs import RunProjections
from chickadee.schema.journal import TaskReportedP, TellDeliveredP, TellReplyP
from chickadee.state.roots import roots as resolve_roots

TITLE = "Return 2"
SAID = "use argon2 instead of bcrypt"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def raw(raw: dict[str, Any]) -> dict[str, Any]:
    raw["workspace"] = {"mode": "worktree"}
    return raw


def _attended(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    zeus: list[Any],
    team: list[Any],
) -> PacketEngine:
    engine, _ = build(monkeypatch, repo, raw, zeus, interaction=Approving())
    # One rung for every box role: Athena's steps and the worker's follow one another.
    model = CoderModel(team)
    monkeypatch.setattr(delegation_module, "client_for", lambda _endpoint: model)
    engine.attended = True
    return engine


def _until(what: str, done: Any) -> Any:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        got = done()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError(f"never: {what}")


def _ended(repo: Path, run_id: str) -> RunProjection | None:
    """The run's journal once it has its `run_end`; None before, a journal not written yet
    included."""
    try:
        run = RunProjections(resolve_roots(repo)).readable(run_id)
    except FileNotFoundError:
        return None
    return run if run.end is not None else None


def _last_user(engine: PacketEngine) -> int:
    return max(t.seq for t in state(engine, SID).turns() if t.role == "user")


def _last_said(request: ChatRequest) -> str:
    content = request.messages[-1].content
    return content if isinstance(content, str) else str(content)


def test_a_tell_holds_the_round_athena_reads_it_continues_and_replies(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine: PacketEngine | None = None
    seen: list[str] = []
    working = threading.Event()

    def waits_for_the_tell(request: object) -> Any:
        assert engine is not None
        live = engine
        working.set()
        # His answer after the call is written once the call returned, the run woken by then.
        _until(
            "Zeus told the team",
            lambda: any(t.text == "Passed on." for t in state(live, SID).turns()),
        )
        # Its call is answered `not run`: the round is held before it runs.
        return calls(box_call("read", "w0", path="app.py"))

    def reads_the_message(request: ChatRequest) -> Any:
        seen.append(_last_said(request))
        return calls(
            box_call("reply", "r1", tell="tell-1", outcome="applied", text="T1 does it")
        )

    def tells(request: object) -> Any:
        # Told while the worker's model call is in flight: its round is the one held.
        assert engine is not None and working.wait(WAIT_S), "the worker never started"
        return calls(call("tell", "t1", text=SAID, **{"from": [_last_user(engine)]}))

    team = [
        *athena_plans_folder(),
        waits_for_the_tell,
        reads_the_message,
        calls(box_call("review", "c1", task="T1", answer="continue")),
        *worker_edit("return 1", "return 2"),
        *athena_accepts(),
    ]
    zeus = [
        *zeus_designs(TITLE),
        answer("The team is on it."),
        tells,
        answer("Passed on."),
        answer("Done: go returns 2."),
        answer("Nothing more."),
    ]
    engine = _attended(monkeypatch, repo, raw, zeus, team)
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        started = state(engine, SID).result("c1")
        assert started is not None and started.text.startswith("RUNNING · run ")
        run_id = started.text.split()[3]
        engine.turn(SID, SAID).wait(timeout=WAIT_S)
        told = state(engine, SID).result("t1")
        assert told is not None
        assert (
            told.text == "told the team (tell-1); delivered at the worker's next tool boundary"
        )

        run = _until("the run ended", lambda: _ended(repo, run_id))
        held = [p.payload for p in run.lines(TaskReportedP)]
        assert (held[0].round, held[0].stopped, held[0].report_sha) == (1, "held", None)
        assert held[-1].stopped == "replied", "continue opened the round that finished it"
        [delivered] = [p.payload for p in run.lines(TellDeliveredP)]
        assert (delivered.tell, delivered.task_id, delivered.round) == ("tell-1", "T1", 1)
        [replied] = [p.payload for p in run.lines(TellReplyP)]
        assert (replied.tell, replied.outcome) == ("tell-1", "applied")
        assert run.end is not None and run.end.outcome == "COMPLETED"

        [message] = seen
        assert "T1 'Return 2' held for a message (round 1" in message
        assert "Message (tell-1), passed on from the user's turn" in message
        assert SAID in message

        notices = [
            t.notice
            for t in state(engine, SID).turns()
            if t.notice is not None and t.notice.trigger is NoticeTrigger.TEAM_REPLIED
        ]
        assert [(n.text, n.handle) for n in notices] == [("tell-1 · applied: T1 does it", None)]
    finally:
        engine.close()


def test_a_tell_with_no_team_working_is_refused(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine: PacketEngine | None = None

    def tells(request: object) -> Any:
        assert engine is not None
        return calls(call("tell", "t1", text=SAID, **{"from": [_last_user(engine)]}))

    engine = _attended(monkeypatch, repo, raw, [tells, answer("Nobody to tell.")], [])
    try:
        engine.turn(SID, SAID).wait(timeout=WAIT_S)
        refused = state(engine, SID).result("t1")
        assert refused is not None and refused.text.startswith("INVALID_ARGUMENTS")
        assert "no team is working" in refused.text and "coder" in refused.text
        assert state(engine, SID).tells() == ()
    finally:
        engine.close()


def test_a_tell_citing_what_is_not_the_person_s_words_is_refused(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """D120's check, as a constraint's: Zeus words a tell, and the person is its authority."""
    engine: PacketEngine | None = None

    def tells(request: object) -> Any:
        assert engine is not None
        # The turn after the person's: the frame or Zeus's own, never their words.
        return calls(call("tell", "t1", text=SAID, **{"from": [_last_user(engine) + 1]}))

    engine = _attended(monkeypatch, repo, raw, [tells, answer("Not theirs.")], [])
    try:
        engine.turn(SID, SAID).wait(timeout=WAIT_S)
        refused = state(engine, SID).result("t1")
        assert refused is not None and refused.text.startswith("INVALID_ARGUMENTS")
        assert "not the person's words" in refused.text
    finally:
        engine.close()


def test_the_same_tell_asked_again_is_the_same_tell(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """A turn cut off after its `tell` line was written is asked again on resume: the same
    words from the same turns to the same run answer with that tell, never a second one."""
    engine = _attended(monkeypatch, repo, raw, [answer("hi")], [])
    monkeypatch.setattr(engine_module, "_paused_of", lambda _session, _runs: "R-paused")
    monkeypatch.setattr(engine, "_work_key", lambda _run: None)
    try:
        engine.turn(SID, SAID).wait(timeout=WAIT_S)
        turn = _last_user(engine)
        first = engine._tell_team(SID, SAID, (turn,))  # pyright: ignore[reportPrivateUsage]
        again = engine._tell_team(SID, SAID, (turn,))  # pyright: ignore[reportPrivateUsage]
        other = engine._tell_team(SID, "and argon2id", (turn,))  # pyright: ignore[reportPrivateUsage]
        assert first == again == "told the team (tell-1); delivered when the run continues"
        assert other.startswith("told the team (tell-2)")
        assert [t.id for t in state(engine, SID).tells()] == ["tell-1", "tell-2"]
    finally:
        engine.close()
