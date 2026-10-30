"""A superseded result is stubbed and MARKED; the call it answered is never touched.

Two paths supersede. A read merged into a later view of the same file stubs the earlier
result when the conversation is assembled; a cut drops an older duplicate of a call made
again. In both, the earlier call's arguments stay byte-identical in what the model is sent
(the transcript) and in the session log, which is the journal of record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from context_helpers import Caller, cut, exchange, numbered, reader_for, reopen, write_files

from chickadee.context.compaction.receipts import exchanges_in, supersede
from chickadee.context.lines import canonical_json
from chickadee.context.request import assemble
from chickadee.context.results import SUPERSEDED_STUB
from chickadee.context.session import Session, Turn
from chickadee.context.store import SessionLog

READ_1 = {"path": "a.txt", "lines": "1-10"}
READ_2 = {"path": "a.txt", "lines": "11-20"}
TESTS = {"command": "pytest -q", "network": []}


def logged_args(session: Session, call_id: str) -> bytes:
    """The arguments of `call_id` exactly as the log on disk holds them."""
    log = SessionLog.read(session.root, session.session_id)
    for event in log.events():
        calls = json.loads(log.payload(event)).get("calls") if event.kind == "turn" else None
        for call in calls if isinstance(calls, list) else []:
            if isinstance(call, dict) and call.get("id") == call_id:
                return canonical_json(call["arguments"])
    raise AssertionError(f"no call {call_id} in the log")


def sent_args(session: Session, call_id: str) -> bytes:
    """The arguments of `call_id` exactly as the assembled transcript sends them."""
    for message in assemble(session.state, session.blob).messages:
        for call in message.tool_calls:
            if call.id == call_id:
                return canonical_json(call.arguments)
    raise AssertionError(f"no call {call_id} in the transcript")


def tool_text(session: Session, call_id: str) -> str:
    for message in assemble(session.state, session.blob).messages:
        if message.tool_call_id == call_id:
            assert isinstance(message.content, str)
            return message.content
    raise AssertionError(f"no result for {call_id} in the transcript")


def as_sent(args: Mapping[str, object]) -> bytes:
    return canonical_json(dict(args))


# -- a read merged into a later view ---------------------------


def merged(session: Session, ws: Path) -> Caller:
    write_files(ws, {"a.txt": numbered(100)})
    call = Caller(session, reader_for(session, ws))
    session.record_user("read a.txt")
    call("read", **READ_1)
    call("read", **READ_2)
    return call


def test_a_merged_read_stubs_the_earlier_result_and_names_the_view_that_absorbed_it(
    session: Session, ws: Path
) -> None:
    merged(session, ws)
    assert session.state.superseded_by("c1") == "c2"
    assert tool_text(session, "c1") == SUPERSEDED_STUB.format(call_id="c2")
    later = tool_text(session, "c2")
    assert " 1│line 1" in later
    assert "20│line 20" in later


def test_the_earlier_calls_arguments_are_byte_identical_in_transcript_and_log(
    session: Session, ws: Path
) -> None:
    merged(session, ws)
    for call_id, args in (("c1", READ_1), ("c2", READ_2)):
        assert sent_args(session, call_id) == as_sent(args)
        assert logged_args(session, call_id) == as_sent(args)


def test_the_log_keeps_the_superseded_result_whole(session: Session, ws: Path) -> None:
    """The stub is how the result is ASSEMBLED, not an edit: the line that recorded what
    the model read is untouched, so replay reproduces the stub rather than depending on
    it."""
    merged(session, ws)
    first = session.state.result("c1")
    assert first is not None
    assert "10│line 10" in first.text
    with reopen(session) as again:
        assert again.state.superseded_by("c1") == "c2"
        assert tool_text(again, "c1") == SUPERSEDED_STUB.format(call_id="c2")
        assert logged_args(again, "c1") == as_sent(READ_1)


def test_every_superseded_call_still_has_its_result_message(session: Session, ws: Path) -> None:
    """A stubbed result is still a result: the call it answers keeps its pair."""
    merged(session, ws)
    messages = assemble(session.state, session.blob).messages
    calls = [c.id for m in messages for c in m.tool_calls]
    answered = [m.tool_call_id for m in messages if m.role == "tool"]
    assert calls == answered == ["c1", "c2"]


# -- an older duplicate dropped by a cut -----------------------------------------


def twice_run(session: Session) -> None:
    session.record_user("run the tests")
    exchange(session, "t1", "shell", TESTS, "1 failed\n" * 40, finding="test_a fails")
    exchange(session, "t2", "shell", TESTS, "all passed\n" * 40, finding="now green")
    session.record_user("good, carry on")
    session.record_assistant("carrying on")


def test_a_cut_drops_an_older_duplicate_and_marks_it_with_the_newer_call(
    session: Session, ws: Path
) -> None:
    twice_run(session)
    done = cut(session, ws)
    assert done.cut is not None
    block = done.cut.block
    assert "[superseded · shell pytest -q · the same call at seq 5 is newer" in block
    assert '"test_a fails"' in block
    assert "1 failed" not in block
    assert done.cut.facts["superseded"] == ["t1"]


def test_a_cut_leaves_both_duplicate_calls_byte_identical_in_the_log(
    session: Session, ws: Path
) -> None:
    twice_run(session)
    assert cut(session, ws).cut is not None
    assert logged_args(session, "t1") == logged_args(session, "t2") == as_sent(TESTS)
    turn = session.state.turn(2)
    assert isinstance(turn, Turn)
    assert as_sent(turn.calls[0].args) == as_sent(TESTS)


def test_supersede_keeps_the_newest_of_each_identical_call_and_never_an_ask_user_answer(
    session: Session,
) -> None:
    session.record_user("go")
    exchange(session, "s1", "shell", TESTS, "old")
    exchange(session, "s2", "shell", {"command": "pytest -q", "network": ["pypi"]}, "other")
    exchange(session, "s3", "shell", TESTS, "new")
    exchange(session, "q1", "ask_user", {"question": "which?"}, "the first")
    exchange(session, "q2", "ask_user", {"question": "which?"}, "the second")
    kept, dropped = supersede(exchanges_in(session.state, 1, session.state.last_seq))
    assert {ex.call_id for ex in kept} == {"s2", "s3", "q1", "q2"}
    assert {old: new.call_id for old, new in dropped.items()} == {"s1": "s3"}
