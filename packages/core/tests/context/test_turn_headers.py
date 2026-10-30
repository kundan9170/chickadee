"""Every user turn and ask_user answer carries `[turn N]`, N its seq; the header never
changes bytes already sent.

The two lines a constraint may cite are numbered so it can cite them, and the number is the
line's seq: the model never numbers anything, it only cites. The header is part of the
turn's bytes from the first request that carries it, so citing a turn later, cutting before
it, or replaying the log never edits a prefix a cache already holds.
"""

from __future__ import annotations

from pathlib import Path

from context_helpers import cut, exchange, reopen, work_unit

from chickadee.client.request import Message
from chickadee.context.ledger import note_constraint
from chickadee.context.request import assemble, turn_header
from chickadee.context.session import Session
from chickadee.schema import ToolCall


def messages(session: Session) -> tuple[Message, ...]:
    return assemble(session.state, session.blob).messages


def test_a_user_turn_opens_with_its_seq(session: Session) -> None:
    session.record_user("first")
    session.record_assistant("reply")
    session.record_user("second")
    rendered = [m.content for m in messages(session)]
    assert rendered == ["[turn 1]\nfirst", "reply", "[turn 3]\nsecond"]
    assert turn_header(3) == "[turn 3]"


def test_an_ask_user_answer_is_numbered_and_no_other_result_is(session: Session) -> None:
    session.record_user("fix it")
    exchange(session, "s1", "shell", {"command": "ls"}, "a.py\n")
    exchange(session, "q1", "ask_user", {"question": "may I edit a.py?"}, "yes, go")
    by_call = {m.tool_call_id: m.content for m in messages(session) if m.role == "tool"}
    assert by_call == {"s1": "a.py\n", "q1": "[turn 6]\nyes, go"}
    assert session.state.is_user_words(6)
    assert not session.state.is_user_words(3)


def test_a_refused_ask_user_carries_no_header(session: Session) -> None:
    """A call that did not run put no words of the user's into the log."""
    session.record_user("fix it")
    session.record_assistant(
        "", [ToolCall(id="q1", name="ask_user", arguments={"question": "?"})]
    )
    session.record_tool("q1", "INVALID_ARGUMENTS{field: question}", ran=False)
    (answer,) = [m for m in messages(session) if m.role == "tool"]
    assert answer.content == "INVALID_ARGUMENTS{field: question}"
    assert not session.state.is_user_words(3)


def test_new_lines_never_change_the_bytes_already_sent(session: Session) -> None:
    """Each request's messages are a prefix of the next one's: citing a turn, answering a
    question, or recording a notice appends and never rewrites."""
    session.record_user("never touch the public API")
    sent = messages(session)

    def still_a_prefix() -> None:
        nonlocal sent
        now = messages(session)
        assert now[: len(sent)] == sent
        sent = now

    assert note_constraint(session, "the public API is frozen", [1], "session") == "C-1"
    still_a_prefix()
    exchange(session, "q1", "ask_user", {"question": "tests too?"}, "yes")
    still_a_prefix()
    assert session.state.is_user_words(5)
    assert note_constraint(session, "tests are frozen too", [1, 5], "session") == "C-2"
    still_a_prefix()
    session.record_user("go on")
    still_a_prefix()
    assert sent[0].content == "[turn 1]\nnever touch the public API"


def test_a_cut_leaves_every_kept_turn_with_its_original_number(
    session: Session, ws: Path
) -> None:
    work_unit(session, "alpha")
    work_unit(session, "beta")
    last = session.record_user("what did we do?")
    session.record_assistant("two things")
    before = [m.content for m in messages(session) if isinstance(m.content, str)]
    done = cut(session, ws)
    assert done.cut is not None
    after = [m.content for m in messages(session) if isinstance(m.content, str)]
    kept = [c for c in before if c.startswith("[turn ") and int(c[6:].split("]")[0]) >= 8]
    assert kept
    assert all(c in after for c in kept)
    assert f"[turn {last.seq}]\nwhat did we do?" in after


def test_replay_numbers_every_turn_the_same_way(session: Session) -> None:
    session.record_user("one")
    exchange(session, "q1", "ask_user", {"question": "?"}, "two")
    session.record_user("three")
    live = messages(session)
    with reopen(session) as again:
        assert messages(again) == live
