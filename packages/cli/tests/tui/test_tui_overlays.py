"""The overlays over the screen: the command
palette, help, the budget, the models and the sessions, through the pilot.

What these pin: `^P` opens the palette over the REPL's whole command list in its order, a
command no route serves dimmed and named so, `Enter` running a command as typed and completing
one that needs an argument, the filter by the composer's rule; `?` opens help, unless the
composer holds words, and help lists every key and command, a key not built naming what builds
it; under an overlay the screen's own keys do nothing and `Esc` closes it; the budget shows the
route's total, rungs and the board's tasks, unknown never $0, and its keys send `/budget`'s own
lines (`l` only when pressed twice), the warning levels dimmed; the models list rungs with
WINDOW and SERVED BY unknown, roles only where their calls name the rung, and `Enter` sends
`/rung`; the sessions list each session's spend and state from its routes, preview its last run
by the board's fold, `Enter` on this one sends `/resume` and on another reopens the screen on
it, and fork is dimmed; and a bare `/budget`, `/rung`, `/resume`, `/undo`, `/handoff`
or `/help` opens its overlay, while the line an overlay sends is the REPL's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest
from overlay_fakes import Wire, hosted
from recordings import recorded_run_id, run_frames
from textual.pilot import Pilot
from tui_frames import SID, episode

from chickadee_cli.commands import COMMANDS
from chickadee_cli.interaction import Interaction
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.overlays.budget import LIFT_AGAIN, Budget
from chickadee_cli.tui.overlays.help import Help
from chickadee_cli.tui.overlays.models import Models, used_by
from chickadee_cli.tui.overlays.palette import NOT_SERVED, Palette, needs_argument
from chickadee_cli.tui.overlays.sessions import SessionList
from chickadee_cli.tui.talk import OVERLAYS, Talk
from chickadee_cli.tui.talk import Screen as TalkScreen
from chickadee_cli.tui.theme import UNKNOWN
from chickadee_serve.frames import Resumed
from chickadee_wire.client import Client
from chickadee_wire.models import (
    CatalogModel,
    Liveness,
    ModelWiring,
    RungBudget,
    SessionBudget,
    SessionInfo,
)

pytestmark = pytest.mark.anyio

SIZE = (120, 40)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def armed_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.0)


def _feed(upto: int = 60) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


def _app(feed: Feed | None = None) -> ChickadeeApp:
    return ChickadeeApp(feed=feed or _feed(), depth="truecolor", encoding="utf-8")


def _body(app: ChickadeeApp) -> str:
    overlay = app.overlay
    assert overlay is not None
    return str(overlay.content.render())


def _foot(app: ChickadeeApp) -> str:
    assert app.overlay is not None
    return str(app.overlay.foot.render())


async def _open(pilot: Pilot[None], *keys: str) -> None:
    await pilot.press(*keys)
    await pilot.pause()


# -- the palette (05) ------------------------------------------------------------------------


def test_a_command_needs_an_argument_only_outside_its_optional_parts() -> None:
    assert needs_argument(COMMANDS["run"]) and needs_argument(COMMANDS["ask"])
    assert not needs_argument(COMMANDS["journal"]) and not needs_argument(COMMANDS["undo"])
    assert not needs_argument(COMMANDS["budget"])


async def test_ctrl_p_lists_the_repl_s_commands_in_order_and_dims_the_unserved() -> None:
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await _open(pilot, "ctrl+p")
        assert isinstance(app.overlay, Palette) and app.screen is app.overlay
        body = _body(app).splitlines()
        assert [line.split()[0] for line in body] == [f"/{n}" for n in COMMANDS], "D7.2"
        config = next(line for line in body if line.startswith("/config"))
        assert NOT_SERVED in config, "no route serves /config (D7.5)"
        diff = next(line for line in body if line.startswith("/diff"))
        assert diff.rstrip().endswith("^D"), "the bar key that does the same"
        assert "COMMANDS" in str(app.overlay.query_one("#box").border_title)


async def test_the_palette_filters_runs_as_typed_and_completes_an_argument() -> None:
    app = _app()
    _reader, sent = hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        await _open(pilot, "ctrl+p", "b", "u")
        assert _body(app).split()[0] == "/budget"
        await _open(pilot, "enter")
        assert sent.submitted == ["/budget"], "as typed: a bare /budget opens its overlay"
        assert app.overlay is None
        await _open(pilot, "ctrl+p", "r", "u", "n")
        await _open(pilot, "enter")
        assert app.zeus.composer.text == "/run ", "/run <goal> cannot run without its goal"
        assert sent.submitted == ["/budget"]


async def test_under_an_overlay_the_screen_s_keys_do_nothing_and_esc_closes_it() -> None:
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await _open(pilot, "ctrl+p", "f2", "ctrl+d")
        assert app.focus_role == "zeus" and isinstance(app.overlay, Palette)
        await _open(pilot, "escape")
        assert app.overlay is None and app.screen is app.base and app.is_running


# -- help --------------------------------------------------------------------------------------


async def test_question_mark_opens_help_unless_the_composer_holds_words() -> None:
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await _open(pilot, "w", "h", "y", "question_mark")
        assert app.overlay is None and app.zeus.composer.text == "why?", "D7.4"
        app.zeus.composer.load_text("")
        await _open(pilot, "question_mark")
        assert isinstance(app.overlay, Help)


async def test_help_names_every_key_and_command_and_what_builds_an_unbuilt_key() -> None:
    from chickadee_cli.tui.keys import BAR, DIFF_BAR, OFF_BAR

    app = _app()
    async with app.run_test(size=(120, 80)) as pilot:
        await _open(pilot, "question_mark")
        body = _body(app)
        for key in (*(k for g in BAR for k in g), *OFF_BAR, *(k for g in DIFF_BAR for k in g)):
            assert key.label in body, key.key
        assert "undo hunk  (not built: a later release)" in body
        for command in COMMANDS.values():
            assert command.usage.split()[0] in body, command.name


# -- the budget (08) -------------------------------------------------------------------------


def _budget(cap: float | None = 5.0) -> SessionBudget:
    high = RungBudget(cap_seq=None, cap_source="none", cap_usd=None, spent_usd=0.53)
    low = RungBudget(cap_seq=4, cap_source="session", cap_usd=1.5, spent_usd=0.19)
    return SessionBudget(
        cap_seq=None, cap_source="config", cap_usd=cap, spent_usd=0.72,
        rungs={"high-param": high, "low-param": low},
        roles={"zeus": 0.31, "athena": 0.22, "worker": 0.19},
        warning_fractions=[0.5, 0.8],
    )  # fmt: skip


async def test_the_budget_says_unknown_until_the_route_answers_never_zero() -> None:
    app = _app()
    reader, _sent = hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("budget")
        await pilot.pause()
        assert "budget" in reader.refreshed, "read again as it opens"
        total = _body(app).splitlines()[0]
        assert "reading /budget…" in total and "$" not in total


async def test_the_budget_shows_the_route_s_total_rungs_and_the_board_s_tasks() -> None:
    feed = _feed()
    feed.snapshot(_budget())
    app = _app(feed)
    hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("budget")
        await pilot.pause()
        body = _body(app)
        assert "$0.72 of $5.00 hard cap (from config)" in body
        assert "high-param" in body and "none" in body and "$1.50" in body
        board = app.state.athena.board
        assert board is not None
        for task in board.tasks.values():
            assert f"{task.task_id} {task.title}" in body and f"${task.cost_usd:.2f}" in body
        # The warning levels marked under the bar, and the spend by role.
        assert "50%" in body and "80%" in body
        assert "BY ROLE" in body and "planning, reviews" in body and "$0.22" in body
        assert isinstance(app.overlay, Budget)
        assert "w warn levels" in str(app.overlay.actions.render())


async def test_the_budget_s_keys_send_budget_s_own_lines() -> None:
    feed = _feed()
    feed.snapshot(_budget())
    app = _app(feed)
    _reader, sent = hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("budget")
        await _open(pilot, "c", "7", "enter")
        assert sent.commands == ["/budget 7"]
        app.open_overlay("budget")
        await _open(pilot, "r", *"low-param 2", "enter")
        assert sent.commands[-1] == "/budget low-param 2"
        app.open_overlay("budget")
        await _open(pilot, "l")
        assert isinstance(app.overlay, Budget)
        assert LIFT_AGAIN in str(app.overlay.actions.render())
        assert sent.commands[-1] == "/budget low-param 2", "one press lifts nothing"
        await _open(pilot, "l")
        assert sent.commands[-1] == "/budget off" and app.overlay is None
        app.open_overlay("budget")
        await _open(pilot, "w")
        assert sent.commands[-1] == "/budget off" and isinstance(app.overlay, Budget), "W16"


async def test_a_lifted_cap_cannot_be_lifted_again() -> None:
    feed = _feed()
    feed.snapshot(_budget(cap=None))
    app = _app(feed)
    _reader, sent = hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("budget")
        await _open(pilot, "l", "l")
        assert sent.commands == [] and "lifted for this session family" in _body(app)


# -- the models (06) -------------------------------------------------------------------------


def _wired(feed: Feed) -> Feed:
    feed.snapshot(
        ModelWiring(
            catalog=["big", "mid", "small"],
            rungs={"high-param": "big", "low-param": "small"},
            switches=[],
        )
    )
    return feed


def test_a_role_is_listed_under_a_rung_only_where_its_calls_name_it() -> None:
    assert used_by({"athena": "high-param", "worker": None}, "high-param") == ("athena",)
    assert used_by({"athena": None, "worker": None}, "high-param") == ()


async def test_the_models_list_rungs_with_what_the_wire_lacks_unknown() -> None:
    app = _app(_wired(_feed()))
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("rung")
        await pilot.pause()
        assert isinstance(app.overlay, Models)
        lines = _body(app).splitlines()
        assert lines[0].split()[:4] == ["RUNG", "MODEL", "WINDOW", "SERVED"]
        high = next(line for line in lines if line.startswith("high-param"))
        assert high.split()[1:4] == ["big", UNKNOWN, UNKNOWN], "W17"
        rung = app.state.athena.rung
        if rung == "high-param":
            assert "athena" in high, "her calls name the rung"
        assert "locked" not in _body(app).lower(), "never invented"


async def test_the_models_name_each_window_provider_and_the_configured_roles() -> None:
    """What a server that sends the assignment and the catalog's models
    shows; the test above is one that sends neither."""
    feed = _feed()
    big = CatalogModel(context_window=200_000, max_output_tokens=8_000, provider="anthropic")
    small = CatalogModel(context_window=1_000_000, max_output_tokens=8_000, provider="local")
    feed.snapshot(
        ModelWiring(
            catalog=["big", "small"],
            rungs={"high-param": "big", "low-param": "small"},
            switches=[],
            roles={"zeus": "high-param", "athena": "high-param", "worker": "low-param"},
            models={"big": big, "small": small},
        )
    )
    app = _app(feed)
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("rung")
        await pilot.pause()
        lines = _body(app).splitlines()
        high = next(line for line in lines if line.startswith("high-param"))
        assert high.split()[1:4] == ["big", "200K", "anthropic"]
        assert high.rstrip().endswith("zeus, athena")
        low = next(line for line in lines if line.startswith("low-param"))
        assert low.split()[1:4] == ["small", "1M", "local"] and low.rstrip().endswith("worker")


async def test_enter_in_the_catalog_sends_rung_and_the_current_model_sends_nothing() -> None:
    app = _app(_wired(_feed()))
    reader, sent = hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("rung")
        await _open(pilot, "right", "enter")
        assert sent.commands == [], "the rung's own model: nothing to switch"
        await _open(pilot, "down", "enter")
        assert sent.commands == ["/rung high-param mid"] and app.overlay is None
        app.open_overlay("rung")
        await _open(pilot, "r")
        assert "rung" in reader.refreshed


# -- the sessions (09) -----------------------------------------------------------------------


def _sessions() -> Wire:
    run = recorded_run_id()
    here = SessionInfo(
        session_id=SID, last_seq=9, updated_at="2026-10-23T10:04:00Z", title="this one"
    )
    other = SessionInfo(
        session_id="S-other", last_seq=4, updated_at="2026-10-22T09:00:00Z",
        title="Port the journal", last_run_id=run,
    )  # fmt: skip
    spent = {
        sid: SessionBudget(cap_seq=None, cap_source="config", cap_usd=5.0, spent_usd=usd)
        for sid, usd in ((SID, 0.72), ("S-other", 3.4))
    }
    return Wire(
        listed=(other, here),
        budgets=spent,
        lives={
            SID: Liveness(kind="working", last_seq=9),
            "S-other": Liveness(
                kind="idle", last_seq=4, outcome="BUDGET_CAP", why="paused at the cap"
            ),
        },
    )  # fmt: skip


async def test_the_sessions_show_each_one_s_spend_state_and_last_run() -> None:
    app = _app()
    hosted(app, _sessions())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("resume")
        await pilot.pause()
        assert isinstance(app.overlay, SessionList)
        body = _body(app)
        assert body.index(SID) < body.index("S-other"), "newest first"
        assert "2026-10-23 10:04 · $0.72 · working" in body, "the record's time, not an age"
        assert "$3.40 · idle: BUDGET_CAP" in body
        await _open(pilot, "down")
        body = _body(app)
        assert "paused at the cap" in body
        board = app.state.athena.board
        assert board is not None
        for task in board.tasks.values():
            assert f"{task.task_id} {task.title}" in body, "its last run, by the board's fold"
        assert "f fork" in _foot(app)


async def test_enter_resumes_this_session_and_reopens_the_screen_on_another() -> None:
    app = _app()
    _reader, sent = hosted(app, _sessions())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("resume")
        await _open(pilot, "f")
        assert isinstance(app.overlay, SessionList), "fork is dimmed (D7.7)"
        await _open(pilot, "enter")
        assert sent.commands == ["/resume"], "past the overlay: the REPL's resume"
        app.open_overlay("resume")
        await _open(pilot, "down", "enter")
        assert app.next_session == "S-other"
    assert not app.is_running, "the screen closed; `run` opens it on S-other (D7.6)"


# -- the commands that open them -------------------------------------------------------------


class _OverlayScreen:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.said: list[str] = []

    def open_overlay(self, name: str) -> None:
        self.opened.append(name)

    def lines(self, lines: Sequence[str]) -> None:
        self.said.extend(lines)


class _ResumingClient:
    server = None

    def liveness(self, _sid: str) -> Liveness:
        return Liveness(kind="idle", last_seq=1)

    def resume(self, _sid: str) -> Resumed:
        return Resumed(session_id=SID, reconciled=(), in_flight="R-1")


def test_a_bare_command_opens_its_overlay_and_an_overlay_s_line_is_the_repl_s() -> None:
    screen = _OverlayScreen()
    asked = SimpleNamespace(current=None, typed=None)
    followed: list[tuple[str, int]] = []
    talk = Talk(
        cast(Client, _ResumingClient()),
        SID,
        cast(TalkScreen, screen),
        cast(Interaction, asked),
        follow=lambda run, after: followed.append((run, after)),
    )
    for name in OVERLAYS:
        talk.handle(f"/{name}")
    assert screen.opened == list(OVERLAYS)
    talk.handle("/resume", typed=False)
    assert screen.opened == list(OVERLAYS), "past the screen's own commands"
    assert any("continuing run R-1" in line for line in screen.said), "the REPL's /resume"
    # A resumed run writes no `episode`: the screen is told to follow it, from its first
    # record, since the panes fold the whole run.
    assert followed == [("R-1", 0)]


# -- where they sit ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["palette", "help", "budget", "rung", "resume", "undo", "handoff"]
)
async def test_an_overlay_sits_below_the_header_at_its_picture_s_width(name: str) -> None:
    """The header's stage strip and spend stay in view over every
    overlay, and each box is as wide as its picture draws it."""
    app = _app()
    hosted(app, _sessions())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay(name)
        await pilot.pause()
        overlay = app.overlay
        assert overlay is not None
        box = overlay.query_one("#box").region
        header = app.base.query_one("BrandHeader").region
        assert box.y >= header.bottom, "the header is not covered"
        assert box.bottom <= SIZE[1] - 2, "the composer's last line and the bar stay in view"
        assert abs(box.width - SIZE[0] * overlay.WIDTH // 100) <= 1
        assert overlay.styles.padding.totals == (0, 0), (
            "spaced by the box's margin: a padded screen draws the panes beneath it shifted"
        )


async def test_an_overlay_still_fits_a_narrow_terminal() -> None:
    app = _app()
    async with app.run_test(size=(70, 30)) as pilot:
        app.open_overlay("palette")
        await pilot.pause()
        assert app.overlay is not None
        box = app.overlay.query_one("#box").region
        assert 58 <= box.width <= 70, "never narrower than its rows can be read"


async def test_the_sessions_filter_row_takes_the_keys_on_slash() -> None:
    """the mockup draws `> filter` always; the list has the keys until `/`."""
    app = _app()
    hosted(app, _sessions())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("resume")
        await pilot.pause()
        assert isinstance(app.overlay, SessionList)
        filter_box = app.overlay.query_one("#filter")
        assert filter_box.display and app.focused is not filter_box
        await pilot.press("slash", *"port")
        await pilot.pause()
        assert app.focused is filter_box
        assert [s.session_id for s in app.overlay.shown()] == ["S-other"]
