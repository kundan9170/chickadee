"""The failure and edge states, through the pilot and from the plain
functions.

What these pin: a lost wire puts a banner across the header naming what was lost and why,
freezes the panes under a dimmed veil, and keeps what is typed in the box, saying it was not
sent; the cap crossed puts a banner saying
what the run's own `hard_cap` record says (stopped there, or not) and offering `/budget`; a
record the client cannot read is said in the pane it would have fed, each of them; a failed
run's report opens in the worker pane, its terminal and why lines word for word, offering
`/resume` while its journal has not ended and `/handoff` once it has; a run that did not fail
shows no report; and with no terminal there is no screen (`test_tui_command.py`'s exit 6).
"""

from __future__ import annotations

from typing import Any

import pytest
from recordings import run_frames
from tui_frames import RUN_ID, SID, episode, run_record, session

from chickadee_cli.render.stream import Verbosity
from chickadee_cli.tui.app import NOT_SENT_LOST, ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.header import banner_line
from chickadee_cli.tui.model import ZeusKind
from chickadee_cli.tui.panes.athena import athena_cards
from chickadee_cli.tui.panes.worker import (
    WorkerFrame,
    failure_lines,
    worker_after,
    worker_cards,
)
from chickadee_cli.tui.panes.zeus import entry_text, tool_mode
from chickadee_cli.tui.theme import UNICODE, palette
from chickadee_serve.frames import RecordFrame

pytestmark = pytest.mark.anyio

SIZE = (104, 33)
COLOURS = palette("truecolor")
WHY = "the run stopped on a harness error: KeyError: 'T3'"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _feed(upto: int = 30) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


def _cap(terminal: bool) -> RecordFrame:
    payload: dict[str, Any] = {
        "cap_usd": 1.0, "kind": "hard_cap", "scope": "run", "spent_usd": 1.02,
        "terminal": terminal,
    }  # fmt: skip
    return run_record("budget", payload, 900)


def _coder_result(terminal: str, seq: int = 950, run_id: str | None = RUN_ID) -> RecordFrame:
    outcome: dict[str, Any] = {"terminal": terminal, "why": [WHY]}
    payload: dict[str, Any] = {
        "call_id": "c-coder", "ran": True, "text": f"{terminal}{{…}}", "outcome": outcome,
    }  # fmt: skip
    return RecordFrame(
        scope=SID, seq=seq, at="2026-10-10T10:05:00Z", kind="tool", run_id=run_id,
        payload=payload,
    )  # fmt: skip


# -- a lost wire -------------------------------------------------------------------------------


def test_nothing_is_bannered_while_all_is_well() -> None:
    assert banner_line(_feed().state, COLOURS) is None


async def test_a_lost_wire_is_bannered_and_freezes_the_panes() -> None:
    feed = _feed()
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        app.take_lost("the session's stream", "connection reset", True)
        await pilot.pause()
        banner = app.base.query_one("#banner")
        assert banner.display
        text = str(banner.render())
        assert "LOST" in text and "the session's stream: connection reset" in text
        assert "chickadee tui --continue" in text
        assert app.base.query_one("#panes").has_class("-frozen")


async def test_words_typed_while_the_wire_is_lost_are_kept_and_not_sent() -> None:
    feed = _feed()
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        app.take_lost("the session's stream", "connection reset", True)
        app.zeus.composer.load_text("keep this")
        await pilot.press("enter")
        await pilot.pause()
        assert app.zeus.composer.text == "keep this", "D7.11: kept, never queued"
        assert not app.state.zeus.pending
        said = [e.text or "" for e in app.state.zeus.entries if e.kind is ZeusKind.OUTPUT]
        assert any(NOT_SENT_LOST.split(".")[0] in line for line in said)


def test_a_route_that_fails_is_not_a_lost_wire() -> None:
    feed = _feed()
    feed.lost("/budget", "refused", stream=False)
    assert banner_line(feed.state, COLOURS) is None, "said beside where it runs, not bannered"


# -- the cap -----------------------------------------------------------------------------------


def test_the_cap_s_banner_says_what_its_record_says() -> None:
    stopped = _feed()
    stopped.apply(_cap(terminal=True))
    line = banner_line(stopped.state, COLOURS)
    assert line is not None
    assert (
        "$1.02 of $1.00: the run stopped there and kept a handoff you can resume" in line.plain
    )
    assert "/budget raises it, then /resume continues" in line.plain
    going = _feed()
    going.apply(_cap(terminal=False))
    line = banner_line(going.state, COLOURS)
    assert line is not None and "the run goes on to its next boundary" in line.plain
    assert "/resume" not in line.plain


async def test_the_cap_s_banner_is_drawn_across_the_header() -> None:
    feed = _feed()
    feed.apply(_cap(terminal=True))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        banner = app.base.query_one("#banner")
        assert banner.display and "CAP" in str(banner.render())
        assert banner.region.y == app.base.query_one("BrandHeader").region.bottom


# -- a record the client cannot read -----------------------------------------------------------


def test_an_unreadable_record_is_said_in_the_pane_it_would_have_fed() -> None:
    feed = _feed()
    feed.apply(session("turn", {"role": "user"}, 500))  # no text: not a turn this client reads
    zeus = [e for e in feed.state.zeus.entries if e.kind is ZeusKind.UNREADABLE]
    assert zeus, "Zeus's pane"
    mode = tool_mode(Verbosity.NORMAL, expanded=False)
    drawn = entry_text(zeus[-1], mode, Verbosity.NORMAL, COLOURS, feed.state.zeus)
    assert drawn is not None and "cannot read" in drawn.plain, "the REPL's words for it"
    feed.apply(run_record("task_reviewed", {"task_id": "T1"}, 901))
    her = athena_cards(feed.state, COLOURS)[-1]
    assert her.body.plain == "task_reviewed: a record this client cannot read"
    feed.apply(run_record("exec", {"label": 3}, 902, task_id="T2"))
    feed.state.worker.current = "T2"
    its = worker_cards(feed.state, COLOURS)[-1]
    assert its.body.plain == "exec: a record this client cannot read"


# -- a failed run -----------------------------------------------------------------------


def test_a_failed_run_s_report_is_its_own_words_and_offers_resume() -> None:
    feed = _feed()
    feed.apply(_coder_result("BLOCKED"))
    lines = [line.plain for line in failure_lines(feed.state, COLOURS)]
    assert lines == ["RUN BLOCKED", WHY, "its journal did not end: /resume continues it"]
    shown = [line.plain for line in worker_after(feed.state, UNICODE, COLOURS)]
    assert WHY in shown, "opens in the worker pane, under its cards"


def test_a_failed_run_that_ended_offers_its_handoff() -> None:
    feed = _feed(upto=10_000)  # the recording ends its run
    feed.apply(_coder_result("ABORTED", seq=10_001))
    end = feed.state.worker.end
    assert end is not None
    lines = [line.plain for line in failure_lines(feed.state, COLOURS)]
    assert lines[-1] == f"it ended {end.outcome}: /handoff shows what it kept"


def test_a_run_that_did_not_fail_shows_no_report() -> None:
    feed = _feed(upto=10_000)
    feed.apply(_coder_result("PARTIAL", seq=10_001))
    assert failure_lines(feed.state, COLOURS) == []


async def test_the_report_is_in_view_above_the_pinned_facts() -> None:
    """Found in Part 7b's screenshots: docked inside the scroll, the facts covered the newest
    lines, the report among them; the stream now scrolls above them and follows its end."""
    feed = _feed(upto=40)
    feed.apply(_coder_result("BLOCKED"))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        frame = app.base.query_one("#worker", WorkerFrame)
        stream, facts = frame.stream, frame.facts
        assert stream.region.bottom <= facts.region.y, "the facts sit below, not over"
        assert stream.max_scroll_y > 0, "the stream overflows here"
        assert stream.scroll_y == stream.max_scroll_y, "and shows its newest line, the report"


def test_a_coder_answer_with_no_run_is_never_the_run_on_screen_s_failure() -> None:
    """A `coder` call answered with no run (a stop that approved nothing, a work already
    complete) names none: it is Zeus's answer alone, never `RUN BLOCKED` on the run shown."""
    feed = _feed()
    feed.apply(_coder_result("BLOCKED", run_id=None))
    assert feed.state.worker.outcome is None
    assert failure_lines(feed.state, COLOURS) == []
