"""`scopes`: a session's or a run's records past a cursor, the seek every frontend reads with.
A cursor is two numbers; nothing is kept between reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.context.session import Session
from chickadee.context.store import SessionLog
from chickadee.scopes import Cursor, ScopeRecord, run_page, session_page


def _session(root: Path, n: int) -> Session:
    session = Session.create(root, "s1")
    for i in range(n):
        session.record_user(f"line {i}")
    session.sync()
    return session


def test_a_page_reads_past_the_cursor_in_order(tmp_path: Path) -> None:
    _session(tmp_path, 5).close()
    page = session_page(SessionLog.read(tmp_path, "s1"), Cursor(2))
    assert [r.seq for r in page.records] == [3, 4, 5]
    assert page.records[0] == ScopeRecord(
        scope="s1",
        seq=3,
        at=page.records[0].at,
        kind="turn",
        run_id=None,
        payload={"role": "user", "text": "line 2"},
    )
    assert page.cursor == Cursor(5), "a session's cursor is its seq: the log is the holder's"


def test_a_short_page_resumes_from_its_cursor(tmp_path: Path) -> None:
    _session(tmp_path, 5).close()
    first = session_page(SessionLog.read(tmp_path, "s1"), Cursor(), limit=2)
    assert [r.seq for r in first.records] == [1, 2]
    assert first.cursor == Cursor(2, None)
    rest = session_page(SessionLog.read(tmp_path, "s1"), first.cursor, limit=10)
    assert [r.seq for r in rest.records] == [3, 4, 5]


def test_a_log_extended_by_its_holder_pages_on_from_the_cursor(tmp_path: Path) -> None:
    """The engine's fold cache extends a session's log once for every reader; a page reads
    only the lines past its cursor."""
    session = _session(tmp_path, 2)
    log = SessionLog.read(tmp_path, "s1")
    page = session_page(log, Cursor())
    assert session_page(log, page.cursor).records == ()
    session.record_user("later")
    session.sync()
    more = session_page(log.extend(), page.cursor)
    assert [r.seq for r in more.records] == [3]
    session.close()


def test_a_forks_page_reads_through_its_ancestry(tmp_path: Path) -> None:
    parent = _session(tmp_path, 3)
    child = Session.fork(tmp_path, "s1", 3, "try another way", "c1", code="keep")
    child.record_user("child's own")
    child.sync()
    page = session_page(SessionLog.read(tmp_path, "c1"), Cursor())
    assert [r.seq for r in page.records] == [1, 2, 3, 4, 5]
    assert page.records[0].scope == "c1"
    assert page.records[3].kind == "fork"
    child.close()
    parent.close()


def test_a_run_without_a_journal_reads_as_empty(tmp_path: Path) -> None:
    page = run_page(tmp_path / "journal.jsonl", "2026-10-10-0123abcd", Cursor())
    assert page.records == () and page.cursor == Cursor()


def test_a_negative_cursor_is_refused(tmp_path: Path) -> None:
    _session(tmp_path, 1).close()
    with pytest.raises(ValueError, match="cursor"):
        session_page(SessionLog.read(tmp_path, "s1"), Cursor(-1))
