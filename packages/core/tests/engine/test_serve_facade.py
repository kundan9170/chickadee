"""What T-5.8 adds to the facade for a frontend that only reads (`30-serve.md` SV-1): the
session list and one session's facts, what rewinds hide, a scope's records past a cursor,
liveness, and the questions every asker's port now puts on the session log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID as CODER_SID
from coder_helpers import build, make_repo, zeus_delegates
from pydantic import TypeAdapter
from repos import init_repo
from zeus_turn_helpers import (
    SID,
    WAIT_S,
    Answering,
    Step,
    answer,
    call,
    calls,
    config_for,
    start,
    state,
    with_sessions,
)

from chickadee.client.errors import TransportExhausted
from chickadee.client.request import ChatRequest
from chickadee.context.questions import QuestionAnsweredLine
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.liveness import Liveness, LivenessKind
from chickadee.schema.adapter import Completion
from chickadee.scopes import Cursor
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID, "S-two")


def _turn(engine: PacketEngine, text: str, sid: str = SID) -> None:
    engine.turn(sid, text).wait(timeout=WAIT_S)


def test_ask_user_lands_on_the_session_log_with_its_answer(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script: list[Step] = [calls(call("ask_user", "q1", question="which branch?")), answer()]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    engine.interaction = Answering("main")
    _turn(engine, "go")
    log = SessionLog.read(sessions_root(resolve_roots(repo)), SID)
    kinds = [e.kind for e in log.events() if e.kind.startswith("question_")]
    assert kinds == ["question_asked", "question_answered"]
    [answered] = [
        TypeAdapter(QuestionAnsweredLine).validate_json(log.payload(e))
        for e in log.events()
        if e.kind == "question_answered"
    ]
    assert (answered.choice, answered.source) == ("main", "human")


def test_sessions_are_listed_and_titled_without_being_opened(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw), [answer("hi")])
    engines.append(engine)
    _turn(engine, "fix the fetcher\nand its tests")
    listed = {info.session_id: info.title for info in engine.sessions()}
    assert listed == {SID: "fix the fetcher", "S-two": None}
    assert engine.session_info(SID).title == "fix the fetcher"
    assert engine.session_info(SID).last_seq >= 2
    with pytest.raises(SessionError):
        engine.session_info("S-nobody")


def test_a_session_journal_pages_past_its_cursor(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw), [answer("one"), answer("two")])
    engines.append(engine)
    _turn(engine, "first")
    first = engine.journal(SID)
    assert [r.payload.get("text") for r in first.records if r.kind == "turn"] == [
        "first",
        "one",
    ]
    _turn(engine, "second")
    more = engine.journal(SID, first.cursor)
    assert [r.payload.get("text") for r in more.records if r.kind == "turn"] == [
        "second",
        "two",
    ]
    assert all(r.seq > first.cursor.after for r in more.records)
    with pytest.raises(SessionError):
        engine.journal("S-nobody", Cursor())


def test_what_a_conversation_undo_hides_is_named(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw), [answer("one"), answer("two")])
    engines.append(engine)
    _turn(engine, "first")
    assert engine.hidden(SID) == ()
    _turn(engine, "second")
    turns = {
        r.payload.get("text"): r.seq for r in engine.journal(SID).records if r.kind == "turn"
    }
    engine.undo(SID, turns["second"], code=False, conversation=True)
    hidden = engine.hidden(SID)
    assert turns["second"] in hidden and turns["two"] in hidden
    assert turns["first"] not in hidden and turns["one"] not in hidden


def test_a_replied_session_is_idle(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw), [answer("hi")])
    engines.append(engine)
    _turn(engine, "hello")
    assert engine.liveness(SID).kind is LivenessKind.IDLE
    assert engine.liveness("S-two").kind is LivenessKind.IDLE


def test_a_sessions_metrics_are_refused_not_answered_empty(
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    with pytest.raises(NotImplementedError, match="not a run"):
        engine.metrics(SID)


def test_a_live_runs_journal_pages_by_seq_as_a_session_does(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`/run/{id}/events` reads this: a coder run Zeus dispatched, from its `run_start` to its
    `run_end`, every seq once across pages, the cursor carrying the byte offset."""
    repo = make_repo(tmp_path / "repo")
    engine, _ = build(monkeypatch, repo, raw, zeus_delegates())
    engines.append(engine)
    engine.turn(CODER_SID, "edit it").wait(timeout=WAIT_S)
    (episode,) = state(engine, CODER_SID).episodes()

    whole = engine.journal(episode.run_id)
    kinds = [r.kind for r in whole.records]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert {r.scope for r in whole.records} == {episode.run_id}
    assert [r.seq for r in whole.records] == list(range(1, len(kinds) + 1))

    paged: list[int] = []
    cursor = Cursor()
    while True:
        page = engine.journal(episode.run_id, cursor, limit=3)
        if not page.records:
            break
        paged.extend(r.seq for r in page.records)
        assert page.cursor.offset is not None, "every page names where its last record ends"
        cursor = page.cursor
    assert paged == [r.seq for r in whole.records]


def test_a_turn_that_failed_reads_idle_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The model unreachable ends the turn with no reply. The engine hosting the session
    knows nobody is working on it, so liveness reads idle and says why, not `working` forever;
    the next turn that replies clears the reason."""

    def unreachable(_request: ChatRequest) -> Completion:
        raise TransportExhausted("connection refused", 3)

    engine, _ = start(monkeypatch, repo, config_for(raw), [unreachable, answer("hi")])
    engines.append(engine)
    _turn(engine, "hello")
    live = engine.liveness(SID)
    assert live.kind is LivenessKind.IDLE
    assert live.why is not None and "connection refused" in live.why
    _turn(engine, "again")
    assert engine.liveness(SID) == Liveness(LivenessKind.IDLE, engine.liveness(SID).last_seq)
