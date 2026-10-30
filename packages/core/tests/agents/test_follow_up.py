"""D213 Amendment 1 decision 2 (T-3.188 Part 4): `follow_up` continues an agent.

The agent keeps its conversation, read back from the session log, and answers with a new
report, round by round. It is refused, saying what to do instead, for an id the session does
not have, an agent still running, one past `agents.max_follow_ups`, and a conversation that
would no longer fit its window. After a crash its conversation is rebuilt from the log, and a
call the crash left without a result is answered `INTERRUPTED`.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from agents_helpers import (
    ENDPOINT,
    AgentClient,
    LineReader,
    agents_for,
    reader_profile,
    report,
    turn_calls,
    turn_text,
    write_repo,
)

from chickadee.agents.resume import histories, next_agent_id
from chickadee.agents.spawn import Agents
from chickadee.client.request import ChatRequest, Message
from chickadee.context.kept_lines import AgentRecord
from chickadee.context.results import InvalidArgument
from chickadee.context.session import Session
from chickadee.schema.adapter import Completion
from chickadee.schema.journal import AgentMessageP, AgentStartP


def _session(tmp_path: Path) -> Session:
    return Session.create(tmp_path / "sessions", "S-follow")


def _spawned(session: Session, repo: Path, *, client: AgentClient | None = None) -> str:
    first = client or AgentClient(
        turn_calls(("read", {"path": "src/auth.py"})),
        turn_text(report("- the check: `src/auth.py:2-2`.")),
    )
    out = agents_for(session, repo, first).spawn(
        None, "reader", "Where is the user checked?", (), call_id="z1"
    )
    assert isinstance(out, str) and out.startswith("[agent A1 ")
    return out


def _messages_of(session: Session, agent: str, round_: int) -> int:
    return sum(
        1
        for _, r in session.state.lines()
        if isinstance(r, AgentRecord)
        and r.line is not None
        and isinstance(r.line.payload, AgentMessageP)
        and r.line.payload.agent == agent
        and r.line.payload.round == round_
    )


def test_a_follow_up_continues_the_conversation_with_a_new_report(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        _spawned(session, repo)
        spawn_messages = _messages_of(session, "A1", 0)
        second = AgentClient(turn_text(report("- it returns a token: `src/auth.py:3-3`.")))
        out = agents_for(session, repo, second).follow_up(
            "A1", "And what does it return?", call_id="z2"
        )
        assert isinstance(out, str), out
        assert out.startswith("[agent A1 · round 1 · reader · complete")
        (request,) = second.requests
        roles = [m.role for m in request.messages]
        assert roles[:2] == ["system", "user"] and roles[-1] == "user"
        assert request.messages[-1].content == "And what does it return?"
        assert "Where is the user checked?" in str(request.messages[1].content)
        assert any(m.role == "tool" for m in request.messages), "what it read is still there"
        # Round 0's messages are not written again: round 1 holds its message and its reply.
        assert _messages_of(session, "A1", 0) == spawn_messages
        assert _messages_of(session, "A1", 1) == 2
        history = histories(session.state)["A1"]
        assert history.rounds == 2 and history.report is not None
        assert history.report.round == 1
        # It cites lines it read in round 0: they are in its conversation, so they count as
        # seen, and nothing is flagged.
        assert "1 checked · 1 against the lines read" in out and "FLAGGED" not in out
    finally:
        session.close()


def test_an_unknown_agent_is_refused_naming_the_ones_there_are(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        agents = agents_for(session, repo, AgentClient())
        none = agents.follow_up("A1", "more", call_id="z1")
        assert isinstance(none, InvalidArgument) and "none yet" in none.why
        _spawned(session, repo)
        got = agents.follow_up("A7", "more", call_id="z2")
        assert isinstance(got, InvalidArgument) and got.field == "agent"
        assert "its agents are: A1" in got.why
    finally:
        session.close()


def test_an_agent_still_running_is_refused(tmp_path: Path) -> None:
    """While A1's round runs, a follow-up of it is refused, and the round goes on."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    asked: list[object] = []
    agents: list[Agents] = []

    class Meanwhile(AgentClient):
        def send(self, request: ChatRequest) -> Completion:
            if not self.requests:
                asked.append(agents[0].follow_up("A1", "more", call_id="z9"))
            return super().send(request)

    try:
        agents.append(
            agents_for(session, repo, Meanwhile(turn_text(report()), turn_text(report())))
        )
        out = agents[0].spawn(None, "reader", "Look.", (), call_id="z1")
        assert isinstance(out, str) and "complete" in out
        (refused,) = asked
        assert isinstance(refused, InvalidArgument) and "still running" in refused.why
        # Once its report is back, it can be.
        assert isinstance(agents[0].follow_up("A1", "more", call_id="z2"), str)
    finally:
        session.close()


def test_past_its_follow_ups_an_agent_is_refused(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        _spawned(session, repo)
        client = AgentClient(turn_text(report()))
        agents = agents_for(session, repo, client, max_follow_ups=1)
        assert isinstance(agents.follow_up("A1", "once", call_id="z2"), str)
        twice = agents.follow_up("A1", "twice", call_id="z3")
        assert isinstance(twice, InvalidArgument)
        assert "its 1 follow-ups" in twice.why and "spawn a fresh agent" in (twice.do or "")
    finally:
        session.close()


def test_a_conversation_that_would_not_fit_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        _spawned(session, repo)
        client = AgentClient()
        huge = "word " * (ENDPOINT.context_window * 2)
        got = agents_for(session, repo, client).follow_up("A1", huge, call_id="z2")
        assert isinstance(got, InvalidArgument)
        assert "no longer fit" in got.why and "spawn a fresh agent" in (got.do or "")
        assert client.requests == []
        assert histories(session.state)["A1"].rounds == 1, "no round was started"
    finally:
        session.close()


def test_a_partial_agent_can_be_followed_up(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        capped = AgentClient(
            turn_calls(("read", {"path": "src/auth.py"})),
            turn_calls(("read", {"path": "src/auth.py"})),
            turn_text(report(not_done="the callers")),
        )
        profiles = {"reader": reader_profile(max_tool_calls=1)}
        out = agents_for(session, repo, capped, profiles=profiles).spawn(
            None, "reader", "Look.", (), call_id="z1"
        )
        assert isinstance(out, str) and "· partial ·" in out
        more = AgentClient(turn_text(report()))
        got = agents_for(session, repo, more, profiles=profiles).follow_up(
            "A1", "Now the callers.", call_id="z2"
        )
        assert isinstance(got, str) and got.startswith("[agent A1 · round 1 ")
    finally:
        session.close()


class _Crash(BaseException):
    """The process dying mid-round: nothing of the harness catches it."""


class _CrashingReader(LineReader):
    def read(self, target: str, **kwargs: object) -> object:  # type: ignore[override]
        raise _Crash


def test_after_a_crash_the_conversation_is_rebuilt_and_its_open_call_answered(
    tmp_path: Path,
) -> None:
    """A round a crash cut has its start and no report. Resumed, the session numbers its next
    agent past it, and a follow-up continues it from the log, the call the crash left without
    a result answered `INTERRUPTED`."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    crashing = AgentClient(turn_calls(("read", {"path": "src/auth.py"})))
    with contextlib.suppress(_Crash):
        agents_for(session, repo, crashing, reader=_CrashingReader(repo)).spawn(
            None, "reader", "Where is the user checked?", (), call_id="z1"
        )
    session.close()

    again = Session.open(tmp_path / "sessions", "S-follow")
    try:
        cut = histories(again.state)["A1"]
        assert cut.report is None and cut.rounds == 1
        assert next_agent_id(again.state) == "A2"
        client = AgentClient(turn_text(report()))
        out = agents_for(again, repo, client).follow_up("A1", "Go on.", call_id="z2")
        assert isinstance(out, str) and out.startswith("[agent A1 · round 1 ")
        (request,) = client.requests
        tool = [m for m in request.messages if m.role == "tool"]
        assert len(tool) == 1 and "INTERRUPTED" in str(tool[0].content)
        starts = [
            r.line.payload
            for _, r in again.state.lines()
            if isinstance(r, AgentRecord)
            and r.line is not None
            and isinstance(r.line.payload, AgentStartP)
        ]
        assert [(p.agent, p.round) for p in starts] == [("A1", 0), ("A1", 1)]
    finally:
        again.close()


def _answered(messages: tuple[Message, ...]) -> bool:
    """Every tool call has its result before the next message that is not one: the shape
    every endpoint requires."""
    owed: set[str] = set()
    for m in messages:
        if m.role == "tool":
            owed.discard(m.tool_call_id or "")
            continue
        if owed:
            return False
        if m.role == "assistant":
            owed = {c.id for c in m.tool_calls}
    return not owed


def test_after_a_crash_every_later_follow_up_still_sends_every_call_answered(
    tmp_path: Path,
) -> None:
    """The review of #640: the crash's unanswered call, answered `INTERRUPTED` when read back
    but never journaled, must be answered again on every later read, not only while it is the
    last call of the conversation."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    crashing = AgentClient(turn_calls(("read", {"path": "src/auth.py"})))
    with contextlib.suppress(_Crash):
        agents_for(session, repo, crashing, reader=_CrashingReader(repo)).spawn(
            None, "reader", "Look.", (), call_id="z1"
        )
    try:
        reading = AgentClient(
            turn_calls(("read", {"path": "src/auth.py"})), turn_text(report())
        )
        agents_for(session, repo, reading).follow_up("A1", "Go on.", call_id="z2")
        last = AgentClient(turn_text(report()))
        out = agents_for(session, repo, last).follow_up("A1", "And now?", call_id="z3")
        assert isinstance(out, str) and out.startswith("[agent A1 · round 2 ")
        assert all(_answered(r.messages) for r in (*reading.requests, *last.requests))
    finally:
        session.close()


def test_a_harness_error_mid_round_is_a_failed_report_on_the_log(tmp_path: Path) -> None:
    """The review of #640: a round the harness fails is reported `failed`, naming the error,
    and its `agent_report` is written, so it never reads as a round a crash cut."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)

    def broken() -> AgentClient:
        raise RuntimeError("no endpoint key")

    try:
        out = agents_for(session, repo, AgentClient(), clients=broken).spawn(
            None, "reader", "Look.", (), call_id="z1"
        )
        assert isinstance(out, str) and "· failed ·" in out
        assert "the harness failed: RuntimeError: no endpoint key" in out
        history = histories(session.state)["A1"]
        assert history.report is not None and history.report.status == "failed"
    finally:
        session.close()


def test_an_undone_agents_id_is_never_given_again(tmp_path: Path) -> None:
    """The review of #640: an undo hides a turn's `agent` lines, and the next spawn still
    numbers past them, so the whole log never holds two agents of one id."""
    from chickadee.context.lines import UndoLine
    from chickadee.context.rewind import record_rewind

    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        turn = session.record_user("look at login").seq
        _spawned(session, repo)
        record_rewind(session, UndoLine(turn, False, True, ()), (), ())
        assert histories(session.state) == {}, "the undone agent is gone from the conversation"
        assert next_agent_id(session.state) == "A2"
        out = agents_for(session, repo, AgentClient(turn_text(report()))).spawn(
            None, "reader", "Look again.", (), call_id="z2"
        )
        assert isinstance(out, str) and out.startswith("[agent A2 ")
    finally:
        session.close()
