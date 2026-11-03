"""The screen live over a replay: the demo session
and its recorded run served as `chickadee serve --replay` serves them, on a real socket, read by
the screen's own readers into the app under Textual's pilot.

What these pin: the screen reaches what folding the recording directly reaches, so it needed
nothing that was not on the record stream; a route the replay
does not build leaves its answer unknown, never zero; an undo's record makes the screen ask
`/hidden` again; a lost stream is said in the header and the screen goes on; and leaving
stops every reader.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from recordings import Replayed, recording, replayed, run_frames
from textual.content import Content
from tui_frames import SID, WAIT_S, session, until

from chickadee_cli.tui.app import ChickadeeApp, connector
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.live import SessionReader
from chickadee_cli.tui.model import Connection, StaleRoute, TuiState
from chickadee_cli.tui.panes import RoleFrame
from chickadee_serve.frames import Hidden, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.errors import Disconnected
from chickadee_wire.events import Frame
from chickadee_wire.models import Liveness
from chickadee_wire.questions import Questions

pytestmark = pytest.mark.anyio

SIZE = (104, 33)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def replay(tmp_path: Path) -> Iterator[Replayed]:
    with replayed(recording(tmp_path / "recording")) as served:
        yield served


def _app(served: Replayed) -> ChickadeeApp:
    return ChickadeeApp(
        feed=Feed(served.session_id),
        connect=connector(served.client, served.session_id),
        depth="truecolor",
        encoding="utf-8",
    )


def _answer_the_recorded_question(served: Replayed) -> None:
    """The replay waits at its question as live; the screen answers nothing until Part 5,
    so the test answers it, as a person would at the REPL."""
    open_ = Questions()
    open_.feed_all(served.client.session_journal(served.session_id))
    (asked,) = open_.open
    assert asked.question is not None
    served.client.answer(served.session_id, asked.id, asked.question.default or "")


def _direct(served: Replayed) -> TuiState:
    """The recording folded without a screen: the session's records as served, then its run."""
    feed = Feed(served.session_id)
    for frame in served.client.session_journal(served.session_id):
        feed.apply(frame)
    for frame in run_frames():
        feed.apply(frame)
    return feed.state


def _title(app: ChickadeeApp, role: str) -> str:
    title = app.query_one(f"#{role}", RoleFrame).border_title
    return Content.from_markup(str(title)).plain if title is not None else ""


# -- the replay reads as the recording -------------------------------------------------------


async def test_the_screen_reaches_what_folding_the_recording_reaches(replay: Replayed) -> None:
    app = _app(replay)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: app.state.stops.open, "the recorded question")
        _answer_the_recorded_question(replay)
        await until(pilot, lambda: app.state.worker.end is not None, "the run's end")
        await until(
            pilot, lambda: any(e.kind == "marker" for e in app.state.zeus.entries[-1:]), "undo"
        )
        direct = _direct(replay)
        state = app.state
        assert state.athena.board == direct.athena.board
        assert state.worker.tasks == direct.worker.tasks
        recorded = [e.kind for e in state.zeus.entries if e.seq is not None]
        assert recorded == [e.kind for e in direct.zeus.entries], "the records, as folded"
        said = "\n".join(e.text or "" for e in state.zeus.entries if e.seq is None)
        assert "answer by key or number" in said, "the question, shown as the REPL shows it"
        assert _title(app, "diff").startswith("CODE · T2 · last task")
        assert _title(app, "athena").startswith("PLANNER")


async def test_a_route_the_replay_does_not_build_stays_unknown(replay: Replayed) -> None:
    app = _app(replay)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: app.state.info is not None, "the session's info")
        await pilot.pause(0.3)
        assert app.state.header.spend.total_usd is None, "/budget is NOT_BUILT: unknown, not $0"
        assert app.state.liveness is None and app.state.run_header is None
        assert app.state.lost == {}, "a route not built is not a failure"


async def test_an_undo_makes_the_screen_ask_hidden_again(
    replay: Replayed, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    hidden = Client.hidden

    def counting(self: Client, sid: str) -> Hidden:
        asked.append(sid)
        return hidden(self, sid)

    app = _app(replay)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: app.state.stops.open, "the recorded question")
        monkeypatch.setattr(Client, "hidden", counting)  # only reads after this one count
        _answer_the_recorded_question(replay)
        await until(pilot, lambda: asked, "a /hidden read after the undo's record")


# -- failure and leaving ---------------------------------------------------------------------


async def test_a_lost_stream_is_said_and_the_screen_goes_on(
    replay: Replayed, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(*_args: object, **_kwargs: object) -> Iterator[Frame]:
        raise Disconnected("the stream dropped more times than its budget allows")
        yield  # pragma: no cover

    monkeypatch.setattr(Client, "session_events", gone)
    app = _app(replay)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: app.state.lost, "the loss said")
        assert app.state.connection is Connection.LOST
        assert "the session's stream" in app.state.lost
        assert app.is_running


async def test_leaving_stops_every_reader(replay: Replayed) -> None:
    app = _app(replay)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: app.state.info is not None, "the session's info")
        await pilot.press("ctrl+q")
    replay.client.close()  # what the command does once the screen has gone
    deadline = time.monotonic() + WAIT_S
    while _readers():
        assert time.monotonic() < deadline, f"still reading: {_readers()}"
        time.sleep(0.05)


def _readers() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name in ("tui-session", "tui-routes")]


class _TeamWorks:
    """A server whose session's team works on `run`, whose journal is one `run_end`."""

    def __init__(self, run: str) -> None:
        self.run = run
        self.followed: list[str] = []

    def liveness(self, sid: str) -> Liveness:
        return Liveness(kind="working", last_seq=3, team_run=self.run)

    def run_events(self, run_id: str, after: int = 0, **_kw: object) -> Iterator[Frame]:
        self.followed.append(run_id)
        yield RecordFrame(scope=run_id, seq=1, at="t", kind="run_end", payload={})


def test_the_run_the_team_works_on_is_followed_whether_or_not_an_episode_named_it() -> None:
    """A run Zeus's `coder` resumed writes no `episode` of its own; liveness names it,
    and the reader follows it, after handing the liveness on."""
    run = "2026-10-29-0000cafe"
    server = _TeamWorks(run)
    seen: list[object] = []
    reader = SessionReader(
        cast(Client, server),
        "S-1",
        on_frame=seen.append,
        on_snapshot=seen.append,
        on_lost=lambda *_a: None,
    )
    reader._read_one(StaleRoute.LIVENESS)  # pyright: ignore[reportPrivateUsage]
    deadline = time.monotonic() + WAIT_S
    while len(seen) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    reader.close()
    assert server.followed == [run]
    assert isinstance(seen[0], Liveness), "the liveness first, so the screen shows the run"
    assert isinstance(seen[1], RecordFrame) and seen[1].scope == run


class _FailsOnce:
    """A server whose `/budget` gets no answer once, then answers."""

    def __init__(self) -> None:
        self.asked = 0

    def budget(self, sid: str) -> object:
        self.asked += 1
        if self.asked == 1:
            raise Disconnected("GET /session/S-1/budget: no answer came within 30 s")
        return "the budget"


def test_a_route_that_answers_after_failing_is_said_recovered() -> None:
    """The reader hands on that a failed route answers again, after its answer, so the
    screen stops saying it is lost; a route that never failed says nothing more."""
    from chickadee_cli.tui.feed import Recovered

    seen: list[object] = []
    lost: list[tuple[str, str]] = []
    reader = SessionReader(
        cast(Client, _FailsOnce()),
        "S-1",
        on_frame=seen.append,
        on_snapshot=seen.append,
        on_lost=lambda what, why, _stream: lost.append((what, why)),
    )
    reader._read_one(StaleRoute.BUDGET)  # pyright: ignore[reportPrivateUsage]
    assert [w for w, _ in lost] == ["/budget"] and seen == []
    reader._read_one(StaleRoute.BUDGET)  # pyright: ignore[reportPrivateUsage]
    assert seen == ["the budget", Recovered("/budget")]
    reader._read_one(StaleRoute.BUDGET)  # pyright: ignore[reportPrivateUsage]
    assert seen[-1] == "the budget", "said once"
    reader.close()


# -- a reopened session's history is drawn once it is read, not once per record ---------------


def _turns(n: int) -> list[RecordFrame]:
    frames: list[RecordFrame] = []
    for i in range(n):
        frames.append(session("turn", {"role": "user", "text": f"question {i}"}, 2 * i + 1))
        frames.append(session("turn", {"role": "assistant", "text": f"answer {i}"}, 2 * i + 2))
    return frames


async def test_records_waiting_together_are_folded_in_order_and_drawn_once() -> None:
    """The reader runs ahead of the screen while it reads a session's history: what waits is
    folded in order and drawn once, to the state folding the records directly reaches. One
    draw per record made a long session's reopening take minutes (229 s for 1000 records)."""
    frames = _turns(150)
    app = ChickadeeApp(feed=Feed(SID), depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        drawn: list[object] = []
        show = app.show

        def counted(*args: Any, **kwargs: Any) -> None:
            drawn.append(args)
            show(*args, **kwargs)

        app.show = counted  # type: ignore[method-assign]
        for frame in frames:  # all waiting before the loop gets to them
            app.deliver(lambda a, f=frame: a.take_frame(f))
        await pilot.pause()
        direct = Feed(SID)
        for frame in frames:
            direct.apply(frame)
        assert [e.kind for e in app.state.zeus.entries] == [
            e.kind for e in direct.state.zeus.entries
        ]
        assert len(app.state.zeus.entries) == len(frames)
        assert len(drawn) == 1, "one draw for the batch"
        views = list(app.zeus.transcript.children)
        assert len([v for v in views if v.display]) >= len(frames) - 1, "every entry mounted"


async def test_the_zeus_pane_mounts_what_a_draw_adds_in_one_call() -> None:
    """Each mount walks every view already there: one per entry was quadratic in a session's
    length."""
    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        transcript = app.zeus.transcript
        mounts: list[int] = []
        mount = transcript.mount

        def counted(*widgets: Any, **kwargs: Any) -> Any:
            mounts.append(len(widgets))
            return mount(*widgets, **kwargs)

        transcript.mount = counted  # type: ignore[method-assign]
        for frame in _turns(20):
            feed.apply(frame)
        app.show(app.state)
        await pilot.pause()
        assert mounts == [40]
