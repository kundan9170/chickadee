"""Talking with Zeus on the screen, against a hosted engine whose
model answers from a script (`testing/hosting.py`, as the REPL's own tests run).

What these pin: Enter sends what is typed, it is pending until its record lands, and his reply
is drawn; Alt+Enter and Ctrl+J add a line; Up recalls what was said; `/` lists the REPL's
commands, Tab completes one, and a command prints the REPL's lines; a line typed while a turn
runs is queued and says so; Esc stops the turn in flight; a refused turn is no longer pending
and says why; a question is answered by the next line; a character typed in another pane lands
in the composer; and a replay takes answers only.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from engines import Step, answer, call, calls
from hosting import SID, hosted
from recordings import recording, replayed
from textual.pilot import Pilot
from tui_frames import until

from chickadee.client.request import ChatRequest
from chickadee.schema.adapter import Completion
from chickadee_cli.tui.app import ChickadeeApp, connector
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import ZeusKind
from chickadee_cli.tui.panes.cards import Card
from chickadee_cli.tui.talk import QUEUED, READ_ONLY
from chickadee_serve.frames import ErrorBody, ErrorCode, TurnAccepted
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.models import AssistantTurnP

pytestmark = pytest.mark.anyio

SIZE = (104, 33)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _app(client: Client, sid: str = SID) -> ChickadeeApp:
    return ChickadeeApp(
        feed=Feed(sid), connect=connector(client, sid), depth="truecolor", encoding="utf-8"
    )


async def _type(app: ChickadeeApp, pilot: Pilot[None], text: str) -> None:
    app.zeus.composer.load_text(text)
    await pilot.press("enter")


def _said(app: ChickadeeApp) -> str:
    """Everything the screen said in Zeus's pane, as text."""
    return "\n".join(e.text or "" for e in app.state.zeus.entries if e.kind is ZeusKind.OUTPUT)


def _replies(app: ChickadeeApp) -> list[str]:
    return [
        e.record.text
        for e in app.state.zeus.entries
        if e.kind is ZeusKind.ZEUS and isinstance(e.record, AssistantTurnP)
    ]


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[Client]:
    script: list[Step] = getattr(request, "param", [answer("all good")])
    with hosted(monkeypatch, script) as host, host.client() as client:
        yield client


# -- words to Zeus ---------------------------------------------------------------------------


async def test_enter_sends_and_his_reply_is_drawn(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await _type(app, pilot, "is the readme fine?")
        await until(pilot, lambda: _replies(app), "his reply")
        assert app.state.zeus.pending == []
        kinds = [e.kind for e in app.state.zeus.entries]
        assert kinds[:2] == [ZeusKind.YOU, ZeusKind.ZEUS]
        assert _replies(app) == ["all good"]


async def test_alt_enter_and_ctrl_j_add_a_line(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a", "alt+enter", "b", "ctrl+j", "c")
        assert app.zeus.composer.text == "a\nb\nc"
        assert app.state.zeus.entries == [], "nothing was sent"


async def test_up_recalls_what_was_said(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await _type(app, pilot, "first words")
        await until(pilot, lambda: _replies(app), "his reply")
        await pilot.press("up")
        assert app.zeus.composer.text == "first words"
        await pilot.press("down")
        assert app.zeus.composer.text == "", "back to the draft"


# -- commands --------------------------------------------------------------------------------


async def test_slash_lists_the_commands_and_tab_completes_one(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f2", "slash")
        assert app.focused is app.zeus.composer, "/ from any pane goes to the composer"
        await pilot.press("s", "t", "a")
        await pilot.pause()
        assert app.zeus.completions.has_class("-open")
        assert app.zeus.completions.option_count == 1
        await pilot.press("tab")
        assert app.zeus.composer.text == "/status "
        await pilot.press("escape")
        assert not app.zeus.completions.has_class("-open"), "Esc closes the list first"


async def test_a_command_prints_the_repl_s_lines(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        # A bare `/help` is the help overlay now (Part 7); `/verbose` is still the REPL's.
        await _type(app, pilot, "/verbose")
        await until(pilot, lambda: "verbosity normal" in _said(app), "/verbose's line")
        await _type(app, pilot, "/nope")
        await until(pilot, lambda: "no command /nope" in _said(app), "the REPL's refusal")
        await _type(app, pilot, "!ls")
        await until(pilot, lambda: "shell lines are not run" in _said(app), "the shell line")


# -- a turn in flight ------------------------------------------------------------------------


@pytest.fixture
def slow(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Client, threading.Event]]:
    started, release = threading.Event(), threading.Event()

    def slowly(_request: ChatRequest) -> Completion:
        started.set()
        release.wait(10)
        return answer("first done")

    script: list[Step] = [slowly, answer("second done")]
    with hosted(monkeypatch, script) as host, host.client() as client:
        try:
            yield client, started
        finally:
            release.set()


async def test_a_line_typed_while_a_turn_runs_is_queued(
    slow: tuple[Client, threading.Event],
) -> None:
    client, started = slow
    app = _app(client)
    async with app.run_test(size=SIZE) as pilot:
        await _type(app, pilot, "first")
        await until(pilot, started.is_set, "the first turn running")
        await _type(app, pilot, "second")
        await until(pilot, lambda: QUEUED in _said(app), "the queued note")


async def test_esc_stops_the_turn_in_flight(
    slow: tuple[Client, threading.Event], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, started = slow
    asked: list[bool] = []
    interrupt = Client.interrupt

    def recording(self: Client, sid: str, *, hard: bool = False) -> None:
        asked.append(hard)
        interrupt(self, sid, hard=hard)

    monkeypatch.setattr(Client, "interrupt", recording)
    app = _app(client)
    async with app.run_test(size=SIZE) as pilot:
        await _type(app, pilot, "first")
        await until(pilot, started.is_set, "the turn running")
        await pilot.press("escape")
        await until(pilot, lambda: asked, "the interrupt sent")
        assert asked == [False], "one press: the next boundary"
        await until(pilot, lambda: "stopping at the next boundary" in _said(app), "said")


async def test_a_refused_turn_is_no_longer_pending_and_says_why(
    engine: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(_self: Client, _sid: str, _text: str, **_sent: object) -> TurnAccepted:
        raise WireError(
            409, ErrorBody(code=ErrorCode.CONFLICT, message="the session is closing")
        )

    monkeypatch.setattr(Client, "turn", refused)
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await _type(app, pilot, "hello")
        await until(pilot, lambda: "the session is closing" in _said(app), "the refusal")
        assert app.state.zeus.pending == []


# -- questions and panes ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine",
    [[calls(call("ask_user", "q1", question="which branch?")), answer("on it")]],
    indirect=True,
)
async def test_a_question_is_answered_by_the_next_line(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await _type(app, pilot, "go")

        def asked() -> list[Card]:
            return [c for c in app.zeus.query(Card) if c.text.title.plain == "question"]

        await until(pilot, asked, "the question's card")
        (card,) = asked()
        assert card.text.body.plain == "which branch?" and card.text.tick.plain == (
            "waiting for you"
        ), "the question is in its call's card"
        assert app.zeus.composer.placeholder == "Answer Chickadee…"
        await _type(app, pilot, "main, please")
        await until(pilot, lambda: _replies(app)[-1:] == ["on it"], "his reply after it")
        await pilot.pause()
        assert card.text.tick.plain == "✔ answered" and card.text.foot is not None
        assert card.text.foot.plain == "main, please"
        assert "which branch" not in _said(app), "no prose: the question is not printed"
        assert "answer in words" not in _said(app)
        markers = [e for e in app.state.zeus.entries if e.kind is ZeusKind.MARKER]
        assert markers, "the answer's record is kept"
        assert app.state.zeus.busy is None, "the reply is in: not still answering"


async def test_a_character_typed_in_another_pane_lands_in_the_composer(engine: Client) -> None:
    app = _app(engine)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f3", "h", "i")
        assert app.focused is app.zeus.composer
        assert app.zeus.composer.text == "hi"


async def test_a_replay_takes_answers_only(tmp_path: Path) -> None:
    with replayed(recording(tmp_path / "recording")) as served:
        app = _app(served.client, served.session_id)
        async with app.run_test(size=SIZE) as pilot:
            await until(pilot, lambda: app.state.stops.open, "the recorded question")
            await _type(app, pilot, "/run something")
            await until(pilot, lambda: READ_ONLY in _said(app), "read-only, said")
            await _type(app, pilot, "no")
            await until(pilot, lambda: not app.state.stops.open, "the question answered")
