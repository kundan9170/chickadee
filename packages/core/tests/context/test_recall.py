"""`recall` is explicit — cross-session memory is never injected.

Memory is what survived a session: the person's own words, the ledger, the concerns, each
run, and the box's records — found by keyword, as dated pointers to the lines that hold them,
newest first. Nothing runs unless a model calls `recall`, and what the context still shows
verbatim is not recalled.
"""

from __future__ import annotations

import ast
import itertools
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from context_helpers import (
    Caller,
    FakeRecords,
    cut,
    exchange,
    new_session,
    reader_for,
    recall_of,
    run_end,
    run_start,
    sessions_dir,
    work_unit,
)
from session_lines import raw

import chickadee.context.recall as recall_module
import chickadee.context.store as store
from chickadee.context.concerns import note_concern, promote
from chickadee.context.ledger import note_constraint, note_rejected
from chickadee.context.recall import Hit, Recalled, terms
from chickadee.context.request import assemble
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import UnresolvedSearch
from chickadee.context.session import Session


def found(result: object) -> Recalled:
    assert isinstance(result, Recalled), result
    return result


def texts(result: object) -> list[str]:
    return [h.text for h in found(result).hits]


# -- explicit, never injected ------------------------------------------------------------------


def test_nothing_from_another_session_enters_a_context_unasked(
    tmp_path: Path, session: Session, ws: Path
) -> None:
    session.record_user("the retry budget is three attempts")
    other = new_session(tmp_path, "S-next")
    try:
        other.record_user("how many retries did we settle on?")
        assert "three attempts" not in str(assemble(other.state, other.blob).messages)
        answered = Caller(other, reader_for(other, ws))("recall", query="retry budget")
        assert answered.ran
        assert 'user: "the retry budget is three attempts"' in answered.text
    finally:
        other.close()


def test_only_the_recall_tool_reaches_recall() -> None:
    """Checked at the source: no module in the context package imports recall. The tool that
    answers a `recall` call is Zeus's (`zeus/tools/navigate.py`), and nothing that assembles a
    request, cuts one, or restates standing could inject it. (`__init__.py` re-exported it
    until D157 §4 deleted the package's re-exports; it was the one importer allowed.)"""
    package = Path(recall_module.__file__).parent
    importers = []
    for path in sorted(package.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "chickadee.context.recall":
                importers.append(path.name)
    assert sorted(set(importers)) == []


# -- dated pointers, newest first ------------------------------------------------------


def test_a_hit_is_a_dated_pointer_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A line's `at` is second precision, so the clock is pinned: one minute a line, from
    just before midnight, so the older session's line is dated the day before."""
    ticks = itertools.count()
    base = datetime(2026, 9, 25, 23, 59, tzinfo=UTC)
    monkeypatch.setattr(
        store, "_now", lambda: f"{base + timedelta(minutes=next(ticks)):%Y-%m-%dT%H:%M:%SZ}"
    )
    older = new_session(tmp_path, "S-old")
    newer = new_session(tmp_path, "S-new")
    try:
        older.record_user("make the fetcher retry on timeouts")
        newer.record_user("the fetcher retry is too eager")
        newer.record_assistant("noted")
        newer.record_user("cap the fetcher retry at three")
        hits = found(recall_of(sessions_dir(tmp_path))("fetcher retry")).hits
    finally:
        older.close()
        newer.close()
    assert [h.render() for h in hits] == [
        '2026-09-26  user: "cap the fetcher retry at three"  [S-new#3]',
        '2026-09-26  user: "the fetcher retry is too eager"  [S-new#1]',
        '2026-09-25  user: "make the fetcher retry on timeouts"  [S-old#1]',
    ]


def test_every_term_must_match_and_a_quoted_phrase_is_one_term(session: Session) -> None:
    session.record_user("the Retry Loop in fetch.py spins")
    session.record_user("retry the upload later")
    session.record_user("the loop in upload.py spins")
    recall = recall_of(session.root)
    assert terms('spins "the  Retry loop"') == ("spins", "the retry loop")
    assert texts(recall("retry spins")) == ['user: "the Retry Loop in fetch.py spins"']
    assert texts(recall('"retry loop"')) == ['user: "the Retry Loop in fetch.py spins"']
    assert isinstance(recall('"loop retry"'), UnresolvedSearch)


def test_every_record_kind_recall_reaches(session: Session) -> None:
    said = session.record_user("the cache must stay per-user for billing")
    exchange(session, "q1", "ask_user", {"question": "per-user?"}, "yes, for billing")
    assert note_constraint(session, "billing cache is per-user", [said.seq], "session") == "C-1"
    assert (
        note_rejected(
            session, text="a global billing cache", reason="leaks data", by="zeus", at=said.seq
        )
        == "R-1"
    )
    assert note_concern(session, "billing cache may leak", ["cache.py"], "critic") == "K-1"
    session.append(
        raw(
            "design_record", {"id": "D-7", "goal": "per-user billing cache", "outcome": "BUILT"}
        ),
        run_id="2026-09-26-0000beef",
    )
    kinds = [h.kind for h in found(recall_of(session.root)("billing")).hits]
    assert sorted(kinds) == [
        "answer",
        "concern",
        "constraint",
        "design",
        "rejected",
        "turn",
    ]
    rows = texts(recall_of(session.root)("billing"))
    assert 'constraint C-1 (zeus, from turn 1): "billing cache is per-user"' in rows
    assert 'rejected R-1 (zeus): "a global billing cache" — leaks data' in rows
    assert (
        'concern K-1 (a claim, raised by critic) on cache.py: "billing cache may leak"' in rows
    )
    assert 'design D-7 BUILT: "per-user billing cache"' in rows
    assert 'user answered: "yes, for billing"' in rows


def test_a_run_says_how_it_ended_when_its_journal_is_attached(session: Session) -> None:
    alpha = work_unit(session, "alpha")
    beta = work_unit(session, "beta")
    records = FakeRecords({alpha: (run_start(alpha, "build alpha"), run_end(alpha, 2))})
    assert texts(recall_of(session.root, records=records)("run")) == [
        f'run {beta}: "beta" · outcome unknown: the run has no journal · '
        f"result: run {beta} SOLVED",
        # `SOLVED`, the packet loop's terminal; a run ends with an outcome now.
        f'run {alpha}: "build alpha" · COMPLETED · $0.42',
    ]
    assert texts(recall_of(session.root)(beta)) == [
        f'run {beta}: "beta" · result: run {beta} SOLVED'
    ]


# -- what is not recalled, and what is listed rather than searched ---------------------


def test_what_the_context_still_shows_is_not_recalled(session: Session, ws: Path) -> None:
    session.record_user("use exponential backoff for the fetcher")
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("remember the backoff")
    session.record_assistant("I will")
    recall = recall_of(session.root)
    assert isinstance(recall("backoff", current=session.state), UnresolvedSearch)
    assert cut(session, ws).cut is not None
    rows = texts(recall("backoff", current=session.state))
    assert rows == ['user: "use exponential backoff for the fetcher"']


def test_the_current_sessions_own_records_are_recalled(session: Session) -> None:
    """Only turns and tool lines are shown verbatim: a constraint, a rejection or a concern
    recorded in this session is never in messages[], so recall must still find it."""
    said = session.record_user("keep the fetcher's public API stable")
    assert note_constraint(session, "fetcher API is stable", [said.seq], "session") == "C-1"
    alt = note_rejected(
        session, text="rename fetch()", reason="breaks callers", by="zeus", at=said.seq
    )
    assert alt == "R-1"
    assert note_concern(session, "fetch() may be renamed", ["fetcher.py"], "critic") == "K-1"
    recall = recall_of(session.root)
    now = session.state
    assert [h.kind for h in found(recall("rejected", current=now)).hits] == ["rejected"]
    assert [h.kind for h in found(recall("concerns", current=now)).hits] == ["concern"]
    kinds = sorted(h.kind for h in found(recall("fetcher", current=now)).hits)
    assert kinds == ["concern", "constraint"], "the turn is on screen; the records are not"


def test_recall_replays_a_session_again_only_once_its_log_has_changed(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session.record_user("the fetcher retries three times")
    reads: list[str] = []
    real, grow = store.SessionLog.read, store.SessionLog.extend

    def counted(root: Path, session_id: str, *, upto: int | None = None) -> store.SessionLog:
        reads.append(session_id)
        return real(root, session_id, upto=upto)

    def extended(log: store.SessionLog) -> store.SessionLog:
        # A grown log is read on from where it was read: a read all the same.
        reads.append(log.session_id)
        return grow(log)

    monkeypatch.setattr(store.SessionLog, "read", staticmethod(counted))
    monkeypatch.setattr(store.SessionLog, "extend", extended)
    recall = recall_of(session.root)
    assert len(found(recall("fetcher")).hits) == 1
    assert len(found(recall("fetcher")).hits) == 1
    assert reads == [session.session_id], "an unchanged log is not replayed again"
    session.record_user("and the fetcher backs off")
    assert len(found(recall("fetcher")).hits) == 2
    assert len(reads) == 2


def test_two_words_list_every_record_of_their_kind(session: Session) -> None:
    said = session.record_user("look at the pool and the views")
    assert note_concern(session, "the pool leaks", ["db/pool.py"], "zeus") == "K-1"
    assert note_concern(session, "views swallow errors", ["api/views.py"], "critic") == "K-2"
    assert isinstance(promote(session, "K-1", "tests/test_pool.py::test_leak"), int)
    alt = note_rejected(
        session, text="a bigger pool", reason="hides it", by="user", at=said.seq
    )
    assert alt == "R-1"
    recall = recall_of(session.root)
    concerns = texts(recall("concerns"))
    assert len(concerns) == 2
    assert any(t.endswith("· watched by tests/test_pool.py::test_leak") for t in concerns)
    assert [h.kind for h in found(recall("rejected")).hits] == ["rejected"]
    assert isinstance(recall("concerns zebra"), UnresolvedSearch), "two words search, not list"


def test_a_fork_finds_its_parents_lines_once_as_the_parents(
    tmp_path: Path, session: Session
) -> None:
    session.record_user("the parser must accept tabs")
    at = session.record_assistant("agreed").seq
    with Session.fork(
        session.root, session.session_id, at, "try again", "S-child", code="keep"
    ) as child:
        child.record_user("and the parser must accept CRLF")
        hits = found(recall_of(sessions_dir(tmp_path))("parser must")).hits
    assert sorted(h.pointer for h in hits) == ["S-child#4", "S-test#1"]


# -- pointers, not content -------------------------------------------------------------


def test_the_whole_list_sits_behind_a_handle(session: Session, ws: Path) -> None:
    for i in range(12):
        session.record_user(f"deploy note {i}: the deploy waits for the canary")
    listed = found(recall_of(session.root)("deploy canary"))
    result = listed.result(reader_for(session, ws).store, inline_bytes=300)
    assert result.header == "recall('deploy canary') — 12 records in 1 session, newest first"
    shown = (result.content or "").splitlines()
    assert 1 <= len(shown) < 12
    assert result.truncation is not None
    assert result.truncation.dropped == 12 - len(shown)
    assert result.truncation.continue_with == result.handle
    whole = HandleResolver(ws, session.blob).stored_bytes(result.handle)
    assert isinstance(whole, bytes)
    assert whole.decode().splitlines() == list(listed.rows())


def test_a_hit_quotes_one_line_cut_visibly(session: Session) -> None:
    session.record_user("x" * 150 + " marker\nsecond line")
    (hit,) = found(recall_of(session.root)("marker")).hits
    assert isinstance(hit, Hit)
    assert hit.text == f'user: "{"x" * 99}…"'


@pytest.mark.parametrize("query", ["", "   ", '""'])
def test_an_empty_query_is_an_invalid_argument(session: Session, query: str) -> None:
    refused = recall_of(session.root)(query)
    assert not isinstance(refused, Recalled | UnresolvedSearch)
    assert refused.field == "query"
