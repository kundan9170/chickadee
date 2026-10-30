"""Undo, through the pilot.

What these pin: `^U` opens it, lit before any run since a turn can be taken back; it lists the
turns `/undoable` offers, newest first, then each task of the run on screen and its steps,
newest step first, each step by its effect's operation id; `Space` steps through only the
halves the engine offers for a turn; `Enter` sends the REPL's own line (`/undo <turn>
<halves>`, `/undo task <run> <task>`, `/undo <operation> <run>`) and closes, so what was
restored is said in Zeus's pane; "what happens" names only what the records say, and says a
clash is not known before it runs; a refused read is said, the tasks still offered; and closing
the screen while `/undo` asks which halves takes nothing back.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from overlay_fakes import Wire, hosted
from recordings import recorded_run_id, run_frames
from tui_frames import SID, episode

from chickadee_cli.interaction import Interaction
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.keys import BAR, lit
from chickadee_cli.tui.overlays.undo import UNKNOWN_CLASH, Undo
from chickadee_cli.tui.panes.zeus import TALK
from chickadee_cli.tui.talk import Screen as TalkScreen
from chickadee_cli.tui.talk import Talk
from chickadee_wire.client import Client
from chickadee_wire.models import EffectCompletedP, Undoable

pytestmark = pytest.mark.anyio

SIZE = (120, 50)
RUN = recorded_run_id()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _feed() -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        feed.apply(frame)
    return feed


def _wire() -> Wire:
    return Wire(
        undoables=(
            Undoable(turn=1, prompt="first", options=["conversation"]),
            Undoable(turn=2, prompt="add retries", options=["both", "conversation", "code"]),
        )
    )


def _undo(app: ChickadeeApp) -> Undo:
    assert isinstance(app.overlay, Undo)
    return app.overlay


def test_ctrl_u_is_lit_before_any_run() -> None:
    key = next(k for g in BAR for k in g if k.key == "ctrl+u")
    assert key.action is not None and lit(key, Feed(SID).state)


async def test_it_lists_turns_then_each_task_and_its_steps_newest_first() -> None:
    app = ChickadeeApp(feed=_feed(), depth="truecolor", encoding="utf-8")
    hosted(app, _wire())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+u")
        await pilot.pause()
        offered = _undo(app).offered()
        assert [c.kind for c in offered[:2]] == ["turn", "turn"]
        assert offered[0].line == "/undo 2" and offered[1].line == "/undo 1", "newest first"
        board = app.state.athena.board
        assert board is not None
        tasks = [c for c in offered if c.kind == "task"]
        assert [c.line for c in tasks] == [f"/undo task {RUN} {t}" for t in board.tasks]
        worker = app.state.worker.tasks["T1"]
        effects = [
            (e.step, e.record)
            for e in worker.entries
            if e.step is not None and isinstance(e.record, EffectCompletedP)
        ]
        at = offered.index(tasks[0])
        steps = offered[at + 1 : at + 1 + len(effects)]
        assert [c.line for c in steps] == [
            f"/undo {effect.operation_id} {RUN}" for _step, effect in reversed(effects)
        ]
        assert all(c.kind == "step" for c in steps)


async def test_space_steps_through_the_offered_halves_and_enter_sends_one_line() -> None:
    app = ChickadeeApp(feed=_feed(), depth="truecolor", encoding="utf-8")
    _reader, sent = hosted(app, _wire())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+u")
        await pilot.pause()
        body = str(_undo(app).content.render())
        assert "(*) code and conversation" in body and UNKNOWN_CLASH in body
        await pilot.press("space")
        assert "(*) conversation only" in str(_undo(app).content.render())
        await pilot.press("enter")
        await pilot.pause()
        assert sent.commands == ["/undo 2 conversation"] and app.overlay is None
        await pilot.press("ctrl+u")
        await pilot.press("down", "space")
        assert _undo(app).halves == {}, "turn 1 offers one half: nothing to step through"
        await pilot.press("down", "enter")
        await pilot.pause()
        assert sent.commands[-1].startswith(f"/undo task {RUN} ")


async def test_a_refused_read_is_said_and_the_tasks_still_offered() -> None:
    app = ChickadeeApp(feed=_feed(), depth="truecolor", encoding="utf-8")
    wire = _wire()
    wire.undoables = ()

    def refused(_sid: str) -> tuple[Undoable, ...]:
        raise KeyError("not served here")

    wire.undoable = refused  # type: ignore[method-assign]
    hosted(app, wire)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+u")
        await pilot.pause()
        undo = _undo(app)
        assert "the turns: refused" in str(undo.content.render())
        assert undo.offered() and undo.offered()[0].kind == "task"


# -- closing while it asks -----------------------------------------------------------------


class _AskedScreen:
    """A `talk.Screen` that notes when a command asks the person something."""

    def __init__(self) -> None:
        self.asked = threading.Event()
        self.said: list[str] = []

    def prompt(self, text: str | None) -> None:
        if text is not None:
            self.asked.set()

    def lines(self, lines: Sequence[str]) -> None:
        self.said.extend(lines)

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


class _UndoClient:
    server = None

    def __init__(self) -> None:
        self.undone: list[int] = []

    def undoable(self, _sid: str) -> tuple[Undoable, ...]:
        halves: list[Literal["both", "conversation", "code"]] = ["both", "conversation", "code"]
        return (Undoable(turn=2, prompt="add retries", options=halves),)

    def undo(self, _sid: str, turn: int, **_kw: object) -> None:
        self.undone.append(turn)


def test_closing_the_screen_while_undo_asks_which_halves_undoes_nothing() -> None:
    """An empty line takes the question's default (`both`); closing the screen is no line at
    all, so the command stops unanswered and nothing is taken back."""
    client = _UndoClient()
    screen = _AskedScreen()
    talk = Talk(
        cast(Client, client),
        SID,
        cast(TalkScreen, screen),
        cast(Interaction, SimpleNamespace(current=None, typed=None)),
        follow=lambda _run, _after: None,
    )
    talk.start()
    talk.submit("/undo 2")
    assert screen.asked.wait(5), "it asks which halves"
    talk.close()
    talk._thread.join(5)  # pyright: ignore[reportPrivateUsage]
    assert not talk._thread.is_alive()  # pyright: ignore[reportPrivateUsage]
    assert client.undone == []
    assert not any("undone" in line for line in screen.said)


async def test_once_its_question_is_answered_the_composer_is_zeus_s_again() -> None:
    """`/undo` asks which halves in the composer; answered, the composer's placeholder is back
    to Zeus's, not the question's."""
    app = ChickadeeApp(feed=_feed(), depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        app.zeus.asking("undo both / conversation / code? [both] ")
        await pilot.pause()
        assert str(app.zeus.composer.placeholder).startswith("undo both")
        app.zeus.asking(None)
        await pilot.pause()
        assert str(app.zeus.composer.placeholder) == TALK
