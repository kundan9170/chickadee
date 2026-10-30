"""`Enter` on a task and on a command (the owner's D7.10), through the
pilot.

What these pin: with Athena's pane focused, `↑`/`↓` choose a task (the one in hand first), its
row marked and the pane's hint naming the keys; `Enter` opens what the records hold of it, round
by round, each round's facts and Athena's answer, the brief and the report's words said not on
the wire; with the worker's pane focused, the latest command is chosen and `Enter` opens
its record, its output said not on the wire with its blobs named; unfocused, a pane marks
nothing; and the keys do nothing under a modal.
"""

from __future__ import annotations

import pytest
from overlay_fakes import Wire, hosted
from recordings import run_frames
from textual.widgets import Static
from tui_frames import SID, approval_asked, episode

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.model import WorkerKind
from chickadee_cli.tui.overlays.command import CommandDetail
from chickadee_cli.tui.overlays.task import NONE_GIVEN, TaskDetail
from chickadee_cli.tui.panes import RoleFrame
from chickadee_cli.tui.panes.cards import Card
from chickadee_wire.models import ExecP

pytestmark = pytest.mark.anyio

SIZE = (120, 40)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def armed_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.0)


def _feed(upto: int = 10_000) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


def _marked(body: str) -> list[str]:
    """The rows marked chosen: `▸ ` before a task's glyph or a command's line (a call's own
    glyph is `▸` too, after `athena `)."""
    return [line for line in body.splitlines() if line.startswith("▸ ")]


def _app(feed: Feed | None = None) -> ChickadeeApp:
    return ChickadeeApp(feed=feed or _feed(), depth="truecolor", encoding="utf-8")


def _body(app: ChickadeeApp, role: str) -> str:
    return str(app.base.query_one(f"#{role}", RoleFrame).body.content)


def _hint(app: ChickadeeApp, role: str) -> str:
    """The pane's keys, under its cards while it has the focus."""
    return str(app.base.query_one(f"#{role}", RoleFrame).query_one(".after", Static).render())


def _chosen_cards(app: ChickadeeApp, role: str) -> list[Card]:
    return [
        c
        for c in app.base.query_one(f"#{role}", RoleFrame).query(Card)
        if c.has_class("-chosen")
    ]


async def test_enter_on_a_task_opens_what_the_records_hold_of_it() -> None:
    app = _app()
    wire = Wire()
    hosted(app, wire)
    async with app.run_test(size=SIZE) as pilot:
        # The run's blobs, as `/run/{id}/blob/{sha}` answers them: the brief and each
        # round's words; one the fake does not hold is refused, and said so.
        worker = app.state.worker.tasks["T1"]
        assert worker.brief_sha is not None
        wire.blobs[worker.brief_sha] = "## T1\nRaise the retry limit."
        for report in worker.reports:
            if report.report_sha is not None:
                wire.blobs[report.report_sha] = f"round {report.round}: done"
        assert _marked(_body(app, "athena")) == [], "unfocused, nothing is marked"
        await pilot.press("f2")
        await pilot.pause()
        body = _body(app, "athena")
        assert len(_marked(body)) == 1 and "↑↓ choose · Enter details" in _hint(app, "athena")
        athena = app.base.query_one("#athena", RoleFrame)
        assert athena.chosen() == "T1", "no task in hand: the first"
        await pilot.press("down")
        assert athena.chosen() == "T2"
        await pilot.press("up", "enter")
        await pilot.pause()
        assert isinstance(app.overlay, TaskDetail) and app.overlay.task_id == "T1"
        detail = str(app.overlay.content.render())
        worker = app.state.worker.tasks["T1"]
        assert len(worker.reports) == 2, "the recording: T1 fails round 1, passes round 2"
        assert "round 1" in detail and "round 2" in detail
        for review in worker.reviews:
            assert f"Planner → {review.answer.replace('_', ' ')}" in detail
        assert "BRIEF\n  ## T1\n  Raise the retry limit." in detail
        assert "│ round 2: done" in detail
        if any(r.text_sha is not None for r in worker.reviews):
            assert "not read: refused" in detail, "a review's words the fake does not hold"
        if any(r.text_sha is None for r in worker.reviews):
            assert NONE_GIVEN in detail
        assert "committed: " in detail


async def test_enter_on_a_command_opens_its_record() -> None:
    app = _app(_feed(upto=40))  # T1 under way, its two commands run
    hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f3")
        await pilot.pause()
        worker = app.base.query_one("#worker", RoleFrame)
        task = app.state.worker.task
        assert task is not None
        commands = [e for e in task.entries if e.kind is WorkerKind.EXEC]
        assert len(commands) == 2
        assert worker.chosen() == str(commands[-1].seq), "the latest command first"
        (marked,) = _chosen_cards(app, "worker")
        assert commands[-1].seq in marked.text.marks, "the card holding the chosen command"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.overlay, CommandDetail)
        ran = commands[-1].record
        assert isinstance(ran, ExecP)
        detail = str(app.overlay.content.render())
        assert ran.label in detail and f"exit {ran.exit_code}" in detail
        assert "STDOUT" in detail and "STDERR" in detail
        # Its output is the run's blob; the fake holds none of it, so it is refused, and said.
        assert "not read: refused" in detail


async def test_a_pane_s_keys_do_nothing_under_a_modal() -> None:
    feed = _feed()
    app = _app(feed)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f2")
        app.take(approval_asked("shell-9", 990))
        await pilot.pause()
        athena = app.base.query_one("#athena", RoleFrame)
        assert not athena.check_action("details", ())
        assert app.overlay is None
