"""Sessions through the facade: create, open, list, latest, title, close, and turns that
outlive the process that wrote them.

Every failure is a `SessionError`, and its message names the path and the last durable seq:
"session broken" without them leaves the person nothing to inspect.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest
from repos import init_repo
from zeus_turn_helpers import WAIT_S, answer, config_for, said, start

import chickadee.context.store as store
import chickadee.durable.log as durable_log
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.ids import new_session_id
from chickadee.sessions import SESSION_TITLE_WIDTH, Sessions, title_of
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def sessions(tmp_path: Path) -> Iterator[Sessions]:
    held = Sessions(tmp_path / "sessions")
    yield held
    held.close_all()


def log_of(sessions: Sessions, session_id: str) -> Path:
    return sessions.root / session_id / "session.jsonl"


# -- create, open, close --------------------------------------------------------------------


def test_create_mints_an_id_and_an_empty_session_is_listed(sessions: Sessions) -> None:
    sid = sessions.create().session_id
    assert re.fullmatch(r"S-[0-9a-f]{8}", sid)
    assert sessions.ids() == (sid,)
    info = sessions.info(sid)
    assert (info.last_seq, info.last_run_id, info.created_at, info.updated_at) == (
        0,
        None,
        "",
        "",
    )
    assert sessions.title(sid) is None
    assert len({new_session_id() for _ in range(200)}) == 200


def test_turns_outlive_the_process_that_wrote_them(sessions: Sessions) -> None:
    first = sessions.create("S-one")
    first.record_user("fix the fetcher")
    first.record_assistant("on it")
    sessions.close("S-one")

    later = Sessions(sessions.root)
    again = later.open("S-one")
    assert [(t.role, t.text) for t in again.state.turns()] == [
        ("user", "fix the fetcher"),
        ("assistant", "on it"),
    ]
    again.record_user("and the tests")
    later.close_all()
    assert Sessions(sessions.root).info("S-one").last_seq == 3


def test_open_hands_back_the_session_this_process_holds(sessions: Sessions) -> None:
    """A log has one writer, so one process holds one `Session` per id."""
    made = sessions.create("S-one")
    assert sessions.open("S-one") is made
    assert sessions.held("S-one") is made
    sessions.close("S-one")
    sessions.close("S-one")
    assert sessions.held("S-one") is None
    assert sessions.open("S-one") is not made


# -- every failure names the path and the last durable seq ----------------------------------


def test_a_second_writer_is_told_who_holds_it(sessions: Sessions) -> None:
    sessions.create("S-one").record_user("hello")
    other = Sessions(sessions.root)
    with pytest.raises(SessionError) as refused:
        other.open("S-one")
    where = re.escape(str(sessions.root / "S-one"))
    assert re.fullmatch(
        rf"session S-one is open for writing in pid {os.getpid()} on \S+ "
        rf"\({where}; last durable seq 1\)",
        str(refused.value),
    )
    assert other.info("S-one").last_seq == 1, "a held session is still describable"
    sessions.close("S-one")
    assert other.open("S-one").state.last_seq == 1


def test_an_open_during_a_close_waits_for_it_rather_than_failing(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close in progress still holds the file lock. An `open` racing it waits for the
    close to finish and reopens, never reports the session held by its own process."""
    session = sessions.create("S-race")
    closing, proceed = threading.Event(), threading.Event()
    real_close = Session.close

    def slow_close(self: Session) -> None:
        closing.set()
        proceed.wait(5)
        real_close(self)

    monkeypatch.setattr(Session, "close", slow_close)
    closer = threading.Thread(target=sessions.close, args=("S-race",))
    closer.start()
    assert closing.wait(5)
    opened: list[object] = []
    opener = threading.Thread(target=lambda: opened.append(_open_or_error(sessions, "S-race")))
    opener.start()
    time.sleep(0.05)
    proceed.set()
    closer.join(5)
    opener.join(5)
    (again,) = opened
    assert isinstance(again, Session), again
    assert again is not session


def _open_or_error(sessions: Sessions, session_id: str) -> Session | SessionError:
    try:
        return sessions.open(session_id)
    except SessionError as exc:
        return exc


def test_a_missing_session_names_where_it_looked(sessions: Sessions) -> None:
    log = re.escape(str(log_of(sessions, "S-none")))
    with pytest.raises(SessionError, match=f"^session S-none does not exist: no log at {log}$"):
        sessions.open("S-none")
    with pytest.raises(SessionError, match="does not exist"):
        sessions.info("S-none")


def test_an_unreadable_log_is_a_session_error_not_a_raw_os_error(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Describing the failure reads the same log again; that second failure is folded into
    the SessionError, never raised past it."""
    sessions.create("S-locked")
    sessions.close("S-locked")

    def unreadable(*_args: object, **_kwargs: object) -> NoReturn:
        raise PermissionError(13, "Permission denied")

    # The tail repair the store made itself is `durable.AppendLog.open`'s since `T-3.50`.
    monkeypatch.setattr(durable_log, "repair_tail", unreadable)
    monkeypatch.setattr(store, "_first_line", unreadable)
    with pytest.raises(SessionError, match="last durable seq unknown"):
        sessions.open("S-locked")


def test_an_id_that_exists_is_not_created_again(sessions: Sessions) -> None:
    sessions.create("S-one").record_user("hello")
    sessions.close("S-one")
    where = re.escape(str(sessions.root / "S-one"))
    with pytest.raises(
        SessionError, match=rf"^session S-one already exists \({where}; last durable seq 1\)$"
    ):
        sessions.create("S-one")
    assert Sessions(sessions.root).info("S-one").last_seq == 1


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", ".hidden", "-dash", "S one"])
def test_an_unusable_id_is_refused_before_any_path_is_built(
    sessions: Sessions, bad: str
) -> None:
    for attempt in (sessions.create, sessions.open, sessions.info, sessions.title):
        with pytest.raises(SessionError, match="is not a usable session id"):
            attempt(bad)
    assert not sessions.root.exists()


def test_a_corrupt_session_names_the_seq_and_the_last_durable_one(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    for text in ("one", "two", "three"):
        held.record_user(text)
    sessions.close("S-one")
    log = log_of(sessions, "S-one")
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text(lines[0] + "not json\n" + lines[2], encoding="utf-8")
    shown = re.escape(str(log))
    expected = rf"^session S-one is corrupt at seq 2: .+ \({shown}; last durable seq 1\)$"
    with pytest.raises(SessionError, match=expected):
        sessions.open("S-one")
    with pytest.raises(SessionError, match=expected):
        sessions.list()


def test_a_torn_tail_is_repaired_and_the_session_opens(sessions: Sessions) -> None:
    """A crash mid-append leaves a partial last line: the one damage a crash produces, and
    the one repair that is legal."""
    held = sessions.create("S-one")
    held.record_user("one")
    held.record_user("two")
    sessions.close("S-one")
    log = log_of(sessions, "S-one")
    whole = log.read_bytes()
    log.write_bytes(whole + b'{"seq":3,"at":"2026-')
    assert sessions.info("S-one").last_seq == 2
    reopened = sessions.open("S-one")
    assert reopened.state.last_seq == 2
    assert log.read_bytes() == whole
    reopened.record_user("three")
    assert reopened.state.last_seq == 3


# -- the title is a projection -------------------------------------------------------


def test_the_title_is_the_first_user_turns_first_line(sessions: Sessions) -> None:
    held = sessions.create("S-one")
    held.record_user("\n  Fix the flaky fetcher test  \nwith the details below")
    held.record_assistant("looking")
    held.record_user("a later turn renames nothing")
    assert sessions.title("S-one") == "Fix the flaky fetcher test"
    sessions.close("S-one")
    assert Sessions(sessions.root).title("S-one") == "Fix the flaky fetcher test"


def test_a_long_title_is_cut_visibly_and_a_blank_one_is_none() -> None:
    long = "x" * (SESSION_TITLE_WIDTH + 25)
    cut = title_of(long)
    assert cut is not None
    assert len(cut) == SESSION_TITLE_WIDTH
    assert cut.endswith("…")
    exact = "y" * SESSION_TITLE_WIDTH
    assert title_of(exact) == exact
    assert title_of("  \n\t\n") is None


# -- list and latest ---------------------------------------------------------------------------


def test_the_most_recently_written_session_is_latest(sessions: Sessions) -> None:
    """`at` has second precision, so the test pins the logs' modification times: within
    one second those decide, and past it they agree with `at` anyway."""
    assert sessions.latest() is None
    now = time.time_ns()

    def wrote(sid: str, text: str, tick: int) -> None:
        held = sessions.held(sid) or sessions.create(sid)
        held.record_user(text)
        os.utime(log_of(sessions, sid), ns=(now + tick, now + tick))

    wrote("S-aaa", "first", 1)
    wrote("S-bbb", "second", 2)
    sessions.create("S-ccc")
    assert [i.session_id for i in sessions.list()] == ["S-bbb", "S-aaa", "S-ccc"]
    latest = sessions.latest()
    assert latest is not None
    assert latest.session_id == "S-bbb"
    wrote("S-aaa", "third", 3)
    latest = sessions.latest()
    assert latest is not None
    assert (latest.session_id, latest.last_seq) == ("S-aaa", 2)


def test_an_empty_session_is_latest_only_alone(sessions: Sessions) -> None:
    sessions.create("S-empty")
    latest = sessions.latest()
    assert latest is not None
    assert latest.session_id == "S-empty"
    sessions.create("S-old").record_user("something to continue")
    latest = sessions.latest()
    assert latest is not None
    assert latest.session_id == "S-old"


# -- through the engine --------------------------------------------


@pytest.fixture
def facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: dict[str, Any]
) -> Iterator[PacketEngine]:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = init_repo(tmp_path / "repo", commit=None)
    engine, _ = start(monkeypatch, repo, config_for(raw), [answer("hello"), answer("again")])
    yield engine
    engine.close()


def test_the_engine_creates_and_finds_a_session(facade: PacketEngine) -> None:
    assert facade.latest_session() is None
    made = facade.create_session("S-made")
    assert made.session_id == "S-made"
    assert made.last_seq == 0
    latest = facade.latest_session()
    assert latest is not None and latest.session_id == "S-made"


def test_a_session_created_without_an_id_is_named(facade: PacketEngine) -> None:
    assert facade.create_session().session_id


def test_creating_an_existing_session_is_refused(facade: PacketEngine) -> None:
    facade.create_session("S-made")
    with pytest.raises(SessionError, match="already exists"):
        facade.create_session("S-made")


def test_a_turn_opens_a_session_and_never_makes_one(facade: PacketEngine) -> None:
    """Item 14: an id no session has is the person's mistake, not a new session."""
    with pytest.raises(SessionError, match="does not exist"):
        facade.turn("S-nobody", "hi")
    facade.create_session("S-made")
    facade.turn("S-made", "hi").wait(timeout=WAIT_S)


def test_open_session_builds_it_and_names_a_missing_one(facade: PacketEngine) -> None:
    with pytest.raises(SessionError, match="does not exist"):
        facade.open_session("S-nobody")
    facade.create_session("S-made")
    opened = facade.open_session("S-made")
    assert opened.session_id == "S-made"


def test_closing_a_session_releases_its_lock_and_it_opens_again(
    facade: PacketEngine,
) -> None:
    """Item 13: the worker, the client and the log's lock go with `close_session`, so another
    writer can take the log, and a later turn opens it here again."""
    facade.create_session("S-made")
    facade.turn("S-made", "hi").wait(timeout=WAIT_S)
    facade.close_session("S-made")

    root = sessions_root(resolve_roots(facade.repo))
    Session.open(root, "S-made").close()  # the lock is free

    facade.turn("S-made", "again").wait(timeout=WAIT_S)
    assert said(facade, "S-made") == ["hi", "again"]


def test_closing_a_session_not_open_here_does_nothing(facade: PacketEngine) -> None:
    facade.close_session("S-nobody")
