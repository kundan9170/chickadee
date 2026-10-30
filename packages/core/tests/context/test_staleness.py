"""An expanded view that goes stale emits exactly one notice; a dereference of a stale handle
returns CURRENT content, never an error; an unchanged sweep emits nothing.


The sweep and the lazy check cover different failures. The sweep catches Zeus reasoning from
view text already on screen, with no call to check at; the lazy check catches a change the
harness did not make, at the point of use. Neither is allowed to cost anything when nothing
moved.
"""

from __future__ import annotations

import os
from pathlib import Path

from context_helpers import Caller, cut, reader_for, reopen, work_unit, write_files

from chickadee.context.freshness import CHANGED_NOTE, sweep
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import HandleExpired, ToolResult
from chickadee.context.session import Session
from chickadee.context.standing import NoticeTrigger
from chickadee.context.store import LOG_NAME, session_dir

A = "def a():\n    return 1\n"
A2 = "def a():\n    return 2\n"
A3 = "def a():\n    return 3\n"
B = "def b():\n    return 2\n"


def stale_notices(session: Session) -> list[tuple[str | None, str]]:
    """Every staleness notice the session holds: the handle it names, and its text."""
    return [
        (t.notice.handle, t.notice.text)
        for t in session.state.turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.STALE_VIEW
    ]


def log_bytes(session: Session) -> bytes:
    return (session_dir(session.root, session.session_id) / LOG_NAME).read_bytes()


# -- the sweep: one line, on the transition only -------------------------------------------


def test_an_expanded_view_that_goes_stale_is_noticed_exactly_once(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"a.py": A})
    call = Caller(session, reader_for(session, ws))
    resolver = HandleResolver(ws, session.blob)
    session.record_user("what does a return?")
    view = call("read", path="a.py", lines="1-2")
    assert view.handle is not None
    write_files(ws, {"a.py": A2})
    (seq,) = sweep(session, resolver)
    assert stale_notices(session) == [
        (view.handle.id, f"view {view.handle.id} (a.py 1-2) is stale")
    ]
    assert sweep(session, resolver) == ()
    write_files(ws, {"a.py": A3})
    assert sweep(session, resolver) == (), "still the view Zeus was told about"
    write_files(ws, {"a.py": A})
    write_files(ws, {"a.py": A2})
    assert sweep(session, resolver) == (), "moving back and forth is not a new view"
    with reopen(session) as again:
        assert sweep(again, HandleResolver(ws, again.blob)) == (), "the told set is in the log"
        assert [t.seq for t in again.state.turns() if t.notice is not None] == [seq]


def test_re_reading_a_stale_view_is_what_makes_it_worth_telling_again(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"a.py": A})
    call = Caller(session, reader_for(session, ws))
    resolver = HandleResolver(ws, session.blob)
    session.record_user("what does a return?")
    old = call("read", path="a.py", lines="1-2")
    write_files(ws, {"a.py": A2})
    assert len(sweep(session, resolver)) == 1
    new = call("read", path="a.py", lines="1-2")
    assert old.handle is not None
    assert new.handle is not None
    assert new.handle.id != old.handle.id
    assert sweep(session, resolver) == ()
    write_files(ws, {"a.py": A3})
    assert len(sweep(session, resolver)) == 1
    assert [h for h, _text in stale_notices(session)] == [old.handle.id, new.handle.id]


def test_a_view_collapsed_by_a_cut_is_not_swept_and_a_pinned_one_is(
    session: Session, ws: Path
) -> None:
    """A receipt cannot be reasoned from, so only the lazy check guards it; a pinned body
    is re-emitted intact, so it is still on screen and the sweep still owes it a line."""
    write_files(ws, {"a.py": A, "b.py": B})
    call = Caller(session, reader_for(session, ws))
    resolver = HandleResolver(ws, session.blob)
    session.record_user("look at both")
    a = call("read", path="a.py", lines="1-2")
    b = call("read", path="b.py", lines="1-2")
    assert a.handle is not None
    assert b.handle is not None
    session.record_assistant("read them", findings={a.call_id: "a returns 1", b.call_id: "b 2"})
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("thanks")
    session.record_assistant("welcome")
    done = cut(session, ws, pinned=[b.seq])
    assert done.cut is not None
    assert done.cut.boundary > b.seq
    assert done.cut.lifted == (b.seq,)
    assert [v.handle_id for v in session.state.expanded_views()] == [b.handle.id]
    write_files(ws, {"a.py": A2, "b.py": B.replace("2", "20")})
    assert len(sweep(session, resolver)) == 1
    assert [h for h, _text in stale_notices(session)] == [b.handle.id]
    reread = reader_for(session, ws).read(a.handle)
    assert isinstance(reread, ToolResult)
    assert any(CHANGED_NOTE in note for note in reread.notes)


# -- the lazy check: the current content, never an error -----------------------------------


def test_a_stale_dereference_returns_what_the_file_says_now(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": A})
    call = Caller(session, reader_for(session, ws))
    session.record_user("what does a return?")
    first = call("read", path="a.py", lines="1-2")
    assert first.handle is not None
    write_files(ws, {"a.py": A2})
    again = call("read", handle=first.handle.id, lines="1-2")
    assert again.ran
    assert "return 2" in again.text
    assert "return 1" not in again.text
    assert f"a.py {CHANGED_NOTE} — {first.handle.id} named" in again.text
    assert again.handle is not None
    assert again.handle.id != first.handle.id
    found = call("grep_in", handle=first.handle.id, pattern="return")
    assert found.ran
    assert "return 2" in found.text
    assert CHANGED_NOTE in found.text


def test_a_handle_on_a_deleted_file_answers_with_a_typed_result(
    session: Session, ws: Path
) -> None:
    """A text file stores nothing in the session, so once it is gone there is no content
    to offer: the answer says the bytes are absent, as a result the loop renders, and the
    call is recorded as one that did nothing."""
    write_files(ws, {"a.py": A})
    call = Caller(session, reader_for(session, ws))
    session.record_user("what does a return?")
    first = call("read", path="a.py", lines="1-2")
    assert first.handle is not None
    (ws / "a.py").unlink()
    gone = reader_for(session, ws).read(first.handle)
    assert gone == HandleExpired(first.handle.id, "absent")
    line = call("read", handle=first.handle.id)
    assert not line.ran
    assert line.text == gone.render()


# -- nothing moved, nothing said -------------------------------------------------------------


def test_an_unchanged_sweep_writes_nothing(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": A, "b.py": B, "c.py": "x = 1\n"})
    call = Caller(session, reader_for(session, ws))
    resolver = HandleResolver(ws, session.blob)
    session.record_user("look around")
    call("read", path="a.py", lines="1-2")
    call("read", path="b.py", lines="1-1")
    log = log_bytes(session)
    for _ in range(5):
        assert sweep(session, resolver) == ()
    later = (ws / "a.py").stat().st_mtime + 60
    os.utime(ws / "a.py", (later, later))
    write_files(ws, {"b.py": B, "c.py": "x = 2\n"})
    assert sweep(session, resolver) == (), "a touch, a same-bytes write, an unread file"
    assert log_bytes(session) == log
