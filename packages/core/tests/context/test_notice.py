"""A notice is appended and NEVER edited; an unchanged standing set emits nothing; the
trigger list is closed.

Telling Zeus something changed is an append, and removing bytes is an edit; only the append
leaves the cache intact. So a notice is one line, written once, and nothing later rewrites,
corrects or moves it. Only a later cut takes it out of the context, and never out of the
log. "Check always, tell only on change": a check that finds nothing new says nothing. And
the seven triggers are the whole list, so an eighth is a change to the spec, never a commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from context_helpers import Caller, cut, exchange, reader_for, reopen, work_unit, write_files
from session_lines import raw

from chickadee.client.request import Message
from chickadee.context.concerns import dismiss, note_concern, promote
from chickadee.context.freshness import sweep
from chickadee.context.ledger import Constraint, note_constraint
from chickadee.context.lines import PayloadError
from chickadee.context.request import assemble
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.context.standing import (
    NOTICE_TRIGGERS,
    Notice,
    NoticeTrigger,
    constraint_proposed,
    constraint_recorded,
    run_ended,
)
from chickadee.context.store import LOG_NAME, session_dir

A = "def a():\n    return 1\n"
B = "def b():\n    return 2\n"
FROZEN = "the public API is frozen"


def log_bytes(session: Session) -> bytes:
    return (session_dir(session.root, session.session_id) / LOG_NAME).read_bytes()


def told(session: Session) -> list[tuple[int, NoticeTrigger, str]]:
    """Every notice the session holds: its seq, its trigger, its text."""
    return [
        (t.seq, t.notice.trigger, t.notice.text)
        for t in session.state.turns()
        if t.notice is not None
    ]


def messages(session: Session) -> tuple[Message, ...]:
    return assemble(session.state, session.blob).messages


# -- appended once, never edited ------------------------------------------------------------


def test_a_notice_follows_the_record_it_reports(session: Session) -> None:
    """A notice is speech about a record, so the record is durable first (§5.11.2 PRECOND)."""
    session.record_user("never change the public API")
    assert note_constraint(session, FROZEN, [1], "session") == "C-1"
    assert note_concern(session, "a() may overflow", ["a.py"], "zeus") == "K-1"
    kinds = [session.state.line(seq) for seq in range(2, 6)]
    assert [line[0].kind for line in kinds if line is not None] == [
        "constraint",
        "turn",
        "concern",
        "turn",
    ]
    assert told(session) == [
        (3, NoticeTrigger.CONSTRAINT, f"constraint C-1 recorded · {FROZEN} · from turn 1"),
        (5, NoticeTrigger.CONCERN, "concern K-1 raised · a.py"),
    ]


def test_later_writes_never_touch_a_notice(session: Session, ws: Path) -> None:
    """Every later write appends: the log's earlier bytes are a prefix of its later bytes,
    and the messages already sent are a prefix of every later request's."""
    write_files(ws, {"a.py": A})
    call = Caller(session, reader_for(session, ws))
    session.record_user("never change the public API")
    view = call("read", path="a.py", lines="1-2")
    assert view.handle is not None
    assert note_constraint(session, FROZEN, [1], "session") == "C-1"
    assert note_concern(session, "a() may overflow", ["a.py"], "zeus") == "K-1"
    write_files(ws, {"a.py": A.replace("1", "10")})
    sweep(session, HandleResolver(ws, session.blob))
    notices = told(session)
    assert [(trigger, text) for _seq, trigger, text in notices] == [
        (NoticeTrigger.CONSTRAINT, f"constraint C-1 recorded · {FROZEN} · from turn 1"),
        (NoticeTrigger.CONCERN, "concern K-1 raised · a.py"),
        (NoticeTrigger.STALE_VIEW, f"view {view.handle.id} (a.py 1-2) is stale"),
    ]
    log, sent = log_bytes(session), messages(session)

    later = session.record_user("the tests are frozen too")
    assert note_constraint(session, "tests frozen", [later.seq], "session", "C-1") == "C-2"
    call("read", path="a.py", lines="1-2")
    assert note_concern(session, "b() is untested", ["b.py"], "critic") == "K-2"
    exchange(session, "s1", "shell", {"command": "pytest -q"}, "3 passed\n")

    assert log_bytes(session).startswith(log)
    assert messages(session)[: len(sent)] == sent
    assert told(session)[: len(notices)] == notices
    with reopen(session) as again:
        assert told(again)[: len(notices)] == notices
        assert messages(again)[: len(sent)] == sent


def test_only_a_cut_takes_a_notice_out_of_the_context_and_the_log_keeps_it(
    session: Session, ws: Path
) -> None:
    session.record_user("never change the public API")
    assert note_constraint(session, FROZEN, [1], "session") == "C-1"
    recorded = f"constraint C-1 recorded · {FROZEN} · from turn 1"
    work_unit(session, "alpha")
    work_unit(session, "beta")
    thanks = session.record_user("thanks, and keep answers short")
    assert note_constraint(session, "answers are short", [thanks.seq], "session") == "C-2"
    session.record_assistant("will do")
    after = next(text for _seq, _t, text in told(session) if "C-2" in text)
    log = log_bytes(session)

    done = cut(session, ws)
    assert done.cut is not None
    assert done.cut.boundary < thanks.seq
    assert done.cut.block.splitlines()[1].endswith("· 1 notice elided]")
    context = [str(m.content) for m in messages(session)]
    assert not [c for c in context if recorded in c]
    assert f"C-1  {FROZEN} · from turn 1" in done.cut.standing
    assert f"<harness_notice>\n{after}\n</harness_notice>" in context
    assert log_bytes(session).startswith(log)
    assert (3, NoticeTrigger.CONSTRAINT, recorded) in told(session)


# -- tell only on change --------------------------------------------------------------------


def test_a_check_that_finds_nothing_new_says_nothing(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": A, "b.py": B})
    call = Caller(session, reader_for(session, ws))
    resolver = HandleResolver(ws, session.blob)
    session.record_user("look at both")
    call("read", path="a.py", lines="1-2")
    call("read", path="b.py", lines="1-2")

    for _ in range(3):
        assert sweep(session, resolver) == ()
    assert told(session) == []
    assert note_concern(session, "a() may overflow", ["a.py"], "zeus") == "K-1"
    assert note_concern(session, "a() is slow", ["a.py"], "critic") == "K-2"
    assert isinstance(promote(session, "K-2", "tests/test_a.py::test_slow"), int)
    assert isinstance(dismiss(session, "K-1", by="user"), int)
    assert [text for _seq, _trigger, text in told(session)] == ["concern K-1 raised · a.py"]
    write_files(ws, {"a.py": A.replace("1", "10")})
    assert len(sweep(session, resolver)) == 1
    assert sweep(session, resolver) == ()
    write_files(ws, {"a.py": A.replace("1", "20")})
    assert sweep(session, resolver) == (), "the same view is stale; it was already told"
    assert [trigger for _seq, trigger, _text in told(session)] == [
        NoticeTrigger.CONCERN,
        NoticeTrigger.STALE_VIEW,
    ]


# -- the closed list -------------------------------------------------------------------------


def test_the_trigger_list_is_the_ten_the_spec_names() -> None:
    """Six, then D188 decision 2 added the seventh, the view moving into the worktree, D216
    decision 1 the eighth, a job's exit, D227 decision 4 the ninth, a hook's words, and D240
    §2.10 the tenth, the team's reply to a tell."""
    assert [(t.value, t.name) for t in NOTICE_TRIGGERS] == [
        (1, "CONSTRAINT"),
        (2, "CONCERN"),
        (3, "STALE_VIEW"),
        (4, "RUN_ENDED"),
        (5, "SNAPSHOT"),
        (6, "FORK_EFFECTS"),
        (7, "VIEW_MOVED"),
        (8, "JOB_EXITED"),
        (9, "HOOK"),
        (10, "TEAM_REPLIED"),
    ]
    with pytest.raises(ValueError, match=r"not one of §5\.11\.2's notice triggers"):
        Notice(cast("NoticeTrigger", 11), "an eleventh")


@pytest.mark.parametrize("trigger", [0, 11, 99])
def test_the_fold_refuses_a_notice_no_trigger_names(session: Session, trigger: int) -> None:
    session.record_user("hello")
    line = {"role": "harness", "trigger": trigger, "text": "an eighth", "handle": None}
    why = rf"{trigger} is not one of §5\.11\.2's notice triggers"
    with pytest.raises(PayloadError, match=why):
        session.append(raw("turn", line))
    assert session.state.last_seq == 1
    with reopen(session) as again:
        assert again.state.last_seq == 1


def test_each_trigger_states_an_event_in_the_specs_words() -> None:
    """The table's Emits column. An event is true forever, which is what makes an
    append-only notice safe: nothing it said can later need correcting."""
    c9 = Constraint(
        "C-9", "do not modify payments/", (511,), "user", "design", "q-3", 512, None, None
    )
    assert constraint_recorded(c9).text == (
        "constraint C-9 recorded · do not modify payments/ (your words) · from turn 511"
    )
    assert constraint_recorded(c9, supersedes="C-4").text.endswith(" · supersedes C-4")
    assert constraint_proposed(412, "session").text == (
        "turn 412 may hold an unrecorded constraint (checkpoint) · session"
    )
    assert run_ended("r7", 4).text == "run r7 ended · worktree has 4 commits, not applied"
    assert run_ended("r7", 1).text == "run r7 ended · worktree has 1 commit, not applied"
    assert {
        constraint_recorded(c9).trigger,
        constraint_proposed(412, "session").trigger,
        run_ended("r7", 4).trigger,
    } == {NoticeTrigger.CONSTRAINT, NoticeTrigger.RUN_ENDED}


def test_a_fork_is_told_its_effects_once_on_arrival_and_a_cut_keeps_them(
    session: Session, ws: Path
) -> None:
    """Trigger 6 fires at most once in a session's life: the rate is what admitted it."""
    session.record_user("migrate the webhooks")
    migrate = {
        "command": "curl -X POST https://api.example/migrate",
        "network": True,
    }
    exchange(session, "s1", "shell", migrate, "migrated\n")
    session.record_user("that went badly")
    at = session.record_assistant("let us try another way").seq
    with Session.fork(
        session.root, session.session_id, at, "try another way", "S-fork", code="keep"
    ) as child:
        (arrival,) = told(child)
        seq, trigger, text = arrival
        assert (seq, trigger) == (at + 2, NoticeTrigger.FORK_EFFECTS)
        assert text.splitlines() == [
            f'[forked from S-test at seq {at} · "try another way"]',
            f"[1 effect at or before seq {at} is NOT undone by this fork:",
            f"   seq 2  network     {migrate['command']}",
            " the workspace is fresh; the world is not]",
        ]
        work_unit(child, "alpha")
        work_unit(child, "beta")
        child.record_user("thanks")
        child.record_assistant("welcome")
        done = cut(child, ws)
        assert done.cut is not None
        assert f"[notice · seq {seq} · kept, since nothing restates it]\n{text}" in (
            done.cut.block
        )
        assert [t for _s, t, _x in told(child)].count(NoticeTrigger.FORK_EFFECTS) == 1
