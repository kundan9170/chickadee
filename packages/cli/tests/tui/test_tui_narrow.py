"""The screen in narrow terminals and zoomed (the owner's D6.3), from the plain layout rules and
through the pilot.

What these pin: 100 columns or more is layout E, 80 to 99 stacked with Zeus over one tab that
follows the focus, below 80 one pane at a time, below 60 by 20 a sentence saying the screen is
too small and where the line view is; whatever a layout hides, the ticker names its latest
entry, each drawn by its own pane; the header drops the bird below 100 and the spend's bar
below 80, and its stage strip is cut with `…`, never silently; the bar keeps every key at any
width, dropping labels first; a letter typed while another pane shows brings Zeus back;
`Ctrl+Up` zooms the focused pane and `Ctrl+Down` restores it; a resize keeps the role in
focus; and the modals still fit at 70 columns.
"""

from __future__ import annotations

import pytest
from recordings import run_frames
from tui_frames import SID, approval_asked, episode

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.keys import BAR
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.modals.confirm import ConfirmModal
from chickadee_cli.tui.narrow import TOO_SMALL, Mode, layout, mode_for, ticker_line
from chickadee_cli.tui.statusbar import bar_line
from chickadee_cli.tui.theme import UNICODE, palette

pytestmark = pytest.mark.anyio

COLOURS = palette("truecolor")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def armed_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.0)


def _feed(upto: int = 30) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


def _app(feed: Feed | None = None) -> ChickadeeApp:
    return ChickadeeApp(feed=feed or _feed(), depth="truecolor", encoding="utf-8")


def _shown(app: ChickadeeApp) -> set[str]:
    return {
        r for r in ("zeus", "diff", "athena", "worker") if app.base.query_one(f"#{r}").display
    }


# -- the rules -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "mode"),
    [((104, 33), Mode.WIDE), ((100, 20), Mode.WIDE), ((99, 33), Mode.STACKED),
     ((80, 24), Mode.STACKED), ((79, 24), Mode.SINGLE), ((60, 20), Mode.SINGLE),
     ((59, 30), Mode.TOO_SMALL), ((120, 19), Mode.TOO_SMALL)],
)  # fmt: skip
def test_the_mode_follows_the_terminal_s_size(size: tuple[int, int], mode: Mode) -> None:
    assert mode_for(*size) is mode


def test_each_layout_shows_what_section_5_says_and_tickers_what_it_hides() -> None:
    wide = layout(Mode.WIDE, "zeus", None)
    # BACKGROUND rides with CODE in every layout; the app hides it while nothing runs.
    assert wide.panes == {"zeus", "diff", "background", "athena", "worker"} and not wide.ticker
    stacked = layout(Mode.STACKED, "zeus", None)
    assert stacked.panes == {"zeus", "diff", "background"} and stacked.tab == "diff"
    assert stacked.ticker == ("athena", "worker")
    assert layout(Mode.STACKED, "athena", None).panes == {"zeus", "athena"}
    single = layout(Mode.SINGLE, "worker", None)
    assert single.panes == {"worker"} and single.ticker == ("zeus", "athena")
    assert layout(Mode.SINGLE, "diff", None).panes == {"diff", "background"}, "F4 below 80"
    zoomed = layout(Mode.WIDE, "zeus", "athena")
    assert zoomed.panes == {"athena"} and zoomed.ticker == ("zeus", "worker")
    assert layout(Mode.TOO_SMALL, "zeus", None).panes == set()


def test_the_ticker_draws_each_hidden_role_s_latest_entry_as_its_pane_does() -> None:
    line = ticker_line(_feed().state, ("athena", "worker"), UNICODE, COLOURS).plain
    assert line.startswith("athena ") and "  ·  worker " in line
    assert "T1 → feedback" in line, "Athena's last verdict, as her pane draws it"


def test_the_bar_keeps_every_key_at_any_width() -> None:
    state = _feed().state
    narrow = bar_line(state, COLOURS, width=60).plain
    for key in (k for g in BAR for k in g if k.drawn(state)):
        assert f" {key.shown} " in narrow, key.key
    assert "commands" not in narrow, "labels go first"
    assert "commands" in bar_line(state, COLOURS, width=200).plain


# -- through the pilot -----------------------------------------------------------------------


async def test_104_columns_is_layout_e() -> None:
    app = _app()
    async with app.run_test(size=(104, 33)) as pilot:
        await pilot.pause()
        assert _shown(app) == {"zeus", "diff", "athena", "worker"}
        assert not app.base.query_one("#tabs").display
        assert app.base.query_one("#bird").display


async def test_90_columns_stacks_zeus_over_a_tab_that_follows_the_focus() -> None:
    app = _app()
    async with app.run_test(size=(90, 33)) as pilot:
        await pilot.pause()
        assert _shown(app) == {"zeus", "diff"}
        assert app.base.query_one("#zeus").region.width == 90, "full width"
        assert not app.base.query_one("#bird").display, "the bird goes below 100"
        await pilot.press("f2")
        assert _shown(app) == {"zeus", "athena"}
        assert "[PLANNER]" in str(app.base.query_one("#tabs").render())
        await pilot.press("f3")
        assert _shown(app) == {"zeus", "worker"}
        assert "athena " in str(app.base.query_one("#ticker").render())


async def test_70_columns_shows_one_pane_and_a_letter_brings_zeus_back() -> None:
    app = _app()
    async with app.run_test(size=(70, 30)) as pilot:
        await pilot.pause()
        assert _shown(app) == {"zeus"}
        await pilot.press("f2")
        assert _shown(app) == {"athena"}
        await pilot.press("h", "i")
        assert _shown(app) == {"zeus"} and app.zeus.composer.text == "hi"


async def test_below_60_by_20_it_says_it_is_too_small_and_where_the_line_view_is() -> None:
    app = _app()
    async with app.run_test(size=(50, 18)) as pilot:
        await pilot.pause()
        assert _shown(app) == set()
        notice = app.base.query_one("#too-small")
        assert notice.display and str(notice.render()) == TOO_SMALL
        assert "chickadee" in TOO_SMALL


async def test_the_header_cuts_its_stage_strip_with_an_ellipsis_never_silently() -> None:
    app = _app()
    async with app.run_test(size=(70, 30)) as pilot:
        await pilot.pause()
        strip = app.base.query_one("#stages")
        assert strip.styles.text_overflow == "ellipsis" and strip.styles.text_wrap == "nowrap"


async def test_ctrl_up_zooms_the_focused_pane_and_ctrl_down_restores_it() -> None:
    app = _app()
    async with app.run_test(size=(104, 33)) as pilot:
        await pilot.press("f3", "ctrl+up")
        assert _shown(app) == {"worker"} and app.zoomed == "worker"
        ticker = str(app.base.query_one("#ticker").render())
        assert ticker.startswith("athena "), "the hidden roles' latest, so nothing is missed"
        await pilot.press("ctrl+down")
        assert _shown(app) == {"zeus", "diff", "athena", "worker"} and app.zoomed is None
        await pilot.press("f1", "ctrl+up")
        assert _shown(app) == {"zeus"}, "from the composer, Zeus's pane"


async def test_a_resize_keeps_the_role_in_focus() -> None:
    app = _app()
    async with app.run_test(size=(104, 33)) as pilot:
        await pilot.press("f2")
        await pilot.resize_terminal(90, 33)
        await pilot.pause()
        assert _shown(app) == {"zeus", "athena"}, "the tab is the role in focus"
        await pilot.resize_terminal(104, 33)
        await pilot.pause()
        assert _shown(app) == {"zeus", "diff", "athena", "worker"}


async def test_the_modals_still_fit_at_70_columns() -> None:
    feed = _feed()
    feed.apply(approval_asked("shell-1", 900))
    app = _app(feed)
    async with app.run_test(size=(70, 30)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ConfirmModal)
        box = app.screen.query_one("#box")
        assert 0 < box.region.width <= 70
        assert "Deny" in str(app.screen.answers.render()), "every answer still drawn"
