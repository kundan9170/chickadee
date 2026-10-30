"""The header's rows (`tui/header.py`): the session, where it runs, each role's model,
and how it gives way on a short screen; and each pane's label (`panes.spend_label`): what its
role spent and how full its latest prompt was, the model being the header's.

What the wire does not carry is left out, never filled in: a role no record or wiring names
reads `—`, a pane with no call names no spend and no context.
"""

from __future__ import annotations

import threading

import pytest
from overlay_fakes import Wire, hosted
from rich.cells import cell_len
from textual.content import Content
from textual.widgets import Static
from tui_frames import SID, run_record, session, spend, user, workspace

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.header import (
    SHORT_BELOW,
    BrandHeader,
    full_screen_line,
    header_rows,
    models_line,
    session_line,
    where_lines,
)
from chickadee_cli.tui.keys import ScreenView
from chickadee_cli.tui.model import StaleRoute, TuiState
from chickadee_cli.tui.panes import border_title, spend_label
from chickadee_cli.tui.statusbar import StatusBar, bar_line
from chickadee_cli.tui.talk import SCREEN_COMMANDS
from chickadee_cli.tui.theme import ASCII, UNICODE, glyphs_for, palette
from chickadee_serve.frames import JsonValue, RecordFrame
from chickadee_wire.models import (
    CatalogModel,
    IndexProgress,
    Liveness,
    ModelWiring,
    RunHeader,
    SessionInfo,
)

pytestmark = pytest.mark.anyio

COLOURS = palette("truecolor")
GLYPHS = glyphs_for("utf-8")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _wiring() -> ModelWiring:
    window = CatalogModel(context_window=200_000, max_output_tokens=8_000, provider="p")
    return ModelWiring(
        catalog=["big", "small", "scout"],
        rungs={"high-param": "big", "low-param": "small", "locate": "scout"},
        switches=[],
        roles={"zeus": "high-param", "athena": "high-param", "worker": "low-param"},
        models={"big": window, "small": window},
    )


def _call(role: str, rung: str, model: str, tokens: int) -> dict[str, JsonValue]:
    return {
        "role": role, "rung": rung, "model": model, "cost_usd": 0.5, "request": "r",
        "stop_reason": "end_turn", "tokens_in": tokens, "tokens_out": 1,
    }  # fmt: skip


def _spend(
    seq: int, rung: str, model: str, tokens: int, role: str | None = None
) -> RecordFrame:
    payload: dict[str, JsonValue] = {
        "cost_usd": 0.1, "cache_writes": 0, "cached_in": 0, "estimated": False,
        "model": model, "rung": rung, "tokens_in": tokens, "tokens_out": 1,
    }  # fmt: skip
    if role is not None:
        payload["role"] = role
    return session("spend", payload, seq)


# -- the rows ---------------------------------------------------------------------------------


def test_the_session_row_is_its_title_and_id_once_read() -> None:
    feed = Feed(SID)
    assert session_line(feed.state, COLOURS).plain.split() == ["session", "—"]
    feed.snapshot(SessionInfo(session_id=SID, last_seq=1, title="Range sums"))
    assert session_line(feed.state, COLOURS).plain.split() == ["session", "Range", "sums", SID]
    feed.snapshot(SessionInfo(session_id=SID, last_seq=1))
    assert "(untitled)" in session_line(feed.state, COLOURS).plain


def test_the_first_turn_makes_the_title_stale_until_one_is_read() -> None:
    """The screen read the session's info as it opened, before any turn, so the title
    (the first turn's first line) stayed `(untitled)`; the first turn reads it again."""
    feed = Feed(SID)
    feed.snapshot(SessionInfo(session_id=SID, last_seq=0))
    assert StaleRoute.INFO in feed.apply(user("Add range-sum queries", 1)).refresh
    feed.snapshot(SessionInfo(session_id=SID, last_seq=1, title="Add range-sum queries"))
    assert StaleRoute.INFO not in feed.apply(user("and tests", 2)).refresh, "titled: once"


def test_a_title_record_reads_the_title_again() -> None:
    """The title role's name, or the person's `/rename`, lands as a `title` record; the
    header reads the session's info again to show it."""
    feed = Feed(SID)
    feed.snapshot(SessionInfo(session_id=SID, last_seq=1, title="Can you look at pr 670"))
    named = feed.apply(session("title", {"text": "PR 670 UI review", "by": "model"}, 2))
    assert StaleRoute.INFO in named.refresh


def test_the_where_row_says_the_tree_and_the_sandbox_once_the_header_is_read() -> None:
    """The checkout's uncommitted paths and the level a command runs at."""
    feed = Feed(SID)
    assert where_lines(feed.state, None, UNICODE, COLOURS).plain.split() == ["repo", "—"]
    feed.apply(workspace("pending", 1))
    feed.snapshot(RunHeader(session_id=SID, lines=[], sandbox="landlock", uncommitted=["a.py"]))
    row = where_lines(feed.state, "project-chickadee", UNICODE, COLOURS).plain
    assert row.startswith("repo     project-chickadee  wt/s-1  branch ccr-1")
    assert row.endswith(" · 1 uncommitted path · sandbox landlock")
    feed.snapshot(RunHeader(session_id=SID, lines=[], sandbox="", uncommitted=[]))
    assert where_lines(feed.state, "r", UNICODE, COLOURS).plain.endswith(" · clean")


def test_each_role_s_model_is_the_wired_one_until_a_call_names_who_served_it() -> None:
    feed = Feed(SID)
    words = ["models", "designer", "—", "·", "planner", "—", "·", "worker", "—"]
    assert models_line(feed.state, COLOURS).plain.split() == words, "unknown, never guessed"
    feed.snapshot(_wiring())
    assert models_line(feed.state, COLOURS).plain.split()[1:] == [
        "designer", "big", "·", "planner", "big", "·", "worker", "small",
    ]  # fmt: skip
    feed.apply(_spend(2, "high-param", "big-fallback", 10, role="zeus"))
    feed.apply(run_record("model_call", _call("athena", "high-param", "big-2", 10), 3))
    worker = _call("worker", "low-param", "small-2", 10)
    feed.apply(run_record("model_call", worker, 4, task_id="T1"))
    feed.apply(_spend(5, "high-param", "scout", 10, role="locate"))
    feed.apply(_spend(6, "high-param", "old", 10))
    assert models_line(feed.state, COLOURS).plain.split()[1:] == [
        "designer", "big-fallback", "·", "planner", "big-2", "·", "worker", "small-2",
    ], "who served each role; another role's line, or one naming none, is not his"  # fmt: skip


def test_a_model_is_named_as_its_provider_names_it_not_by_its_catalog_name() -> None:
    """Wire 4.11: the catalog's `frontier` is the config's word; the header says the model its
    provider serves (`model_id`), and the catalog name where the server does not say."""
    feed = Feed(SID)
    wiring = _wiring()
    named = {
        "big": CatalogModel(
            context_window=200_000, max_output_tokens=8_000, provider="p",
            model_id="claude-opus-4-1",
        ),
        "small": CatalogModel(context_window=200_000, max_output_tokens=8_000, provider="p"),
    }  # fmt: skip
    feed.snapshot(wiring.model_copy(update={"models": named}))
    assert models_line(feed.state, COLOURS).plain.split()[1:] == [
        "designer", "claude-opus-4-1", "·",
        "planner", "claude-opus-4-1", "·",
        "worker", "small",
    ]  # fmt: skip


def test_a_role_s_context_is_its_latest_prompt_against_its_model_s_window() -> None:
    feed = Feed(SID)
    feed.apply(_spend(1, "high-param", "served-model", 10, role="zeus"))
    assert feed.state.context("zeus") == (10, None), "no wiring: no window to read"
    feed.snapshot(_wiring())
    assert feed.state.context("zeus") == (10, None), "a model the catalog lacks: no window"
    feed.apply(_spend(2, "high-param", "big", 50_000, role="zeus"))
    feed.apply(_spend(3, "locate", "scout", 9, role="locate"))
    assert feed.state.context("zeus") == (50_000, 200_000), "the scout's prompt is not his"
    assert feed.state.context("athena") is None and feed.state.context("worker") is None
    feed.apply(run_record("model_call", _call("athena", "high-param", "big", 20_000), 4))
    assert feed.state.context("athena") == (20_000, 200_000)


def test_another_role_s_call_on_his_rung_is_not_the_designer_s() -> None:
    """Wire 4.9: a spend line names its role. `residue` defaults to Zeus's rung; its small
    prompt is neither his context nor his model, and his own line after it is."""
    feed = Feed(SID)
    feed.snapshot(_wiring())
    feed.apply(_spend(1, "high-param", "big", 150_000, role="zeus"))
    feed.apply(_spend(2, "high-param", "big-mini", 4_000, role="residue"))
    feed.apply(_spend(3, "high-param", "big-mini", 4_000, role="locate"))
    assert feed.state.context("zeus") == (150_000, 200_000)
    assert feed.state.role_model("zeus") == "big"
    assert feed.state.zeus.cost_usd == pytest.approx(0.3), "every session charge is his pane's"
    feed.apply(_spend(4, "high-param", "big", 160_000, role="zeus"))
    assert feed.state.context("zeus") == (160_000, 200_000)


def test_a_line_from_before_roles_were_named_is_never_read_as_his() -> None:
    """Before wire 4.9 `residue` shared his rung and a line said no role: its prompt could be
    either's, so it is his pane's spend and not his context or model, which stay unknown."""
    feed = Feed(SID)
    feed.snapshot(_wiring())
    feed.apply(spend(0.1, 1))  # a line naming no role, on high-param
    assert feed.state.context("zeus") is None and feed.state.zeus.cost_usd == pytest.approx(0.1)
    assert feed.state.role_model("zeus") == "big", "the wired model, not the line's"
    feed.apply(_spend(2, "high-param", "big", 50_000, role="zeus"))
    feed.apply(spend(0.1, 3))
    assert feed.state.context("zeus") == (50_000, 200_000), "his own line, and only his"


# -- the panes' labels ------------------------------------------------------------------------


def _forms(cost: float | None, context: tuple[int, int | None] | None) -> list[str]:
    labels = spend_label(cost, context, COLOURS["athena"], GLYPHS, COLOURS)
    return [label.plain for label in labels]


def test_a_label_says_the_spend_then_the_context_widest_first() -> None:
    assert _forms(None, None) == [""], "no call yet: no $0, no 0%"
    assert _forms(0.31, None) == ["$0.31"]
    assert _forms(0.31, (50_000, 200_000)) == [
        "$0.31 · ctx ██░░░░░░ 25%", "$0.31 · ctx 25%", "$0.31",
    ]  # fmt: skip
    assert _forms(0.31, (34_000, None)) == ["$0.31 · ctx 34K", "$0.31"], "no window: tokens"
    assert _forms(None, (50_000, 200_000))[0] == "ctx ██░░░░░░ 25%"


def test_a_full_context_is_drawn_red() -> None:
    label = spend_label(1.0, (190_000, 200_000), COLOURS["athena"], GLYPHS, COLOURS)[0]
    reds = {span.style for span in label.spans if str(span.style) == COLOURS["bad"]}
    assert label.plain.endswith("95%") and reds


def test_a_title_takes_the_widest_form_of_its_label_that_fits_whole() -> None:
    forms = (Content("$0.31 · ctx ██░░░░░░ 25%"), Content("$0.31 · ctx 25%"), Content("$0.31"))
    left = Content("PLANNER")
    assert border_title(left, forms, 60, double=False).plain.endswith("ctx ██░░░░░░ 25%")
    assert border_title(left, forms, 34, double=False).plain.endswith("$0.31 · ctx 25%")
    assert border_title(left, forms, 28, double=False).plain.endswith("─ $0.31")


# -- the screen -------------------------------------------------------------------------------


def _shown(app: ChickadeeApp, name: str) -> bool:
    return app.base.query_one(f"BrandHeader #{name}", Static).display


@pytest.mark.parametrize(("height", "tall"), [(34, True), (SHORT_BELOW - 1, False)])
async def test_a_short_screen_keeps_three_rows_of_the_header(height: int, tall: bool) -> None:
    app = ChickadeeApp(feed=Feed(SID), repo="r", depth="truecolor", encoding="utf-8")
    hosted(app, Wire())
    async with app.run_test(size=(120, height)) as pilot:
        await pilot.pause()
        header = app.base.query_one(BrandHeader).region
        assert header.height == header_rows(height) == (6 if tall else 4)
        assert _shown(app, "session") is tall and _shown(app, "models") is tall
        assert app.base.query_one("BrandHeader #bird").display is tall
        assert _shown(app, "where") and _shown(app, "stages"), "where and the stages stay"
        app.open_overlay("help")
        await pilot.pause()
        overlay = app.overlay
        assert overlay is not None
        assert overlay.query_one("#box").region.y == header.bottom, "an overlay starts under it"


# -- full screen -------------------------------------------------------------------


def test_full_screen_says_where_the_stage_and_what_is_uncommitted_on_one_line() -> None:
    feed = Feed(SID)
    feed.apply(workspace("pending", 1))
    line = full_screen_line(feed.state, "r", GLYPHS, COLOURS).plain
    assert line == f"CHICKADEE · r  wt/s-1  branch ccr-1 ·  {GLYPHS.running} DESIGN "
    feed.snapshot(RunHeader(session_id=SID, lines=[], sandbox="", uncommitted=["a.py"]))
    line = full_screen_line(feed.state, "r", GLYPHS, COLOURS).plain
    assert " · 1 uncommitted path · " in line, "said only when there is something to say"


def _rows(app: ChickadeeApp) -> dict[str, bool]:
    header = app.base.query_one(BrandHeader)
    return {
        "line": header.query_one("#line", Static).display,
        "bird": header.query_one("#bird").display,
        "rows": any(s.display for s in header.query(".row")),
    }


async def test_f_makes_the_header_one_line_and_remembers_it() -> None:
    kept: list[bool] = []
    app = ChickadeeApp(
        feed=Feed(SID), repo="r", depth="truecolor", encoding="utf-8", remember=kept.append
    )
    hosted(app, Wire())
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        assert app.base.query_one(BrandHeader).region.height == 6
        await pilot.press("ctrl+f")
        await pilot.pause()
        header = app.base.query_one(BrandHeader).region
        assert header.height == app.header_height == 2
        assert _rows(app) == {"line": True, "bird": False, "rows": False}
        spend = str(app.base.query_one("BrandHeader #spend", Static).render())
        assert spend.splitlines() == ["—"], "the spend's numbers, beside the one line"
        app.open_overlay("help")
        await pilot.pause()
        assert app.overlay is not None
        assert app.overlay.query_one("#box").region.y == header.bottom
        app.overlay.dismiss()
        await pilot.pause()
        await pilot.press("ctrl+f")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.base.query_one(BrandHeader).region.height == 6
        assert _rows(app) == {"line": False, "bird": True, "rows": True}
    assert kept == [True, False]


async def test_a_screen_opens_as_the_person_left_it_and_says_when_it_cannot_remember() -> None:
    def broken(_on: bool) -> None:
        raise OSError("read-only file system")

    app = ChickadeeApp(
        feed=Feed(SID), repo="r", depth="truecolor", encoding="utf-8",
        full_screen=True, remember=broken,
    )  # fmt: skip
    hosted(app, Wire())
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        assert app.base.query_one(BrandHeader).region.height == 2, "opened as left"
        await pilot.press("ctrl+f")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.full_screen, "toggled all the same"
        said = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
        assert any(m.startswith("full screen off for now, not remembered") for m in said)


async def test_the_bar_s_lit_keys_are_buttons() -> None:
    """A lit key's chip runs its action when clicked; a dimmed one (`^D` before a run) has no
    action to run."""
    line = bar_line(TuiState(), COLOURS)
    actions = {str(span.style).partition("@click=")[2] for span in line.spans}
    assert "app.toggle_fullscreen" in actions and "app.overlay('help')" in actions
    assert "app.toggle_diff" not in actions, "dimmed: not a button"
    app = ChickadeeApp(feed=Feed(SID), repo="r", depth="truecolor", encoding="utf-8")
    hosted(app, Wire())
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        bar = app.base.query_one(StatusBar)
        x = cell_len(str(bar.render()).split(" ^F ")[0]) + 2
        await pilot.click(StatusBar, offset=(x, 0))
        await pilot.pause()
        assert app.full_screen and app.base.query_one(BrandHeader).region.height == 2


async def test_two_quick_toggles_keep_the_last_choice() -> None:
    """The writes run on threads; they land one at a time, so the file says the last choice
    however slow the first write is."""
    kept: list[bool] = []
    first = threading.Event()

    def slow(on: bool) -> None:
        if not kept:
            first.wait(1.0)  # the first write stalls until the second toggle has happened
        kept.append(on)

    app = ChickadeeApp(
        feed=Feed(SID), repo="r", depth="truecolor", encoding="utf-8", remember=slow
    )
    hosted(app, Wire())
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+f", "ctrl+f")
        first.set()
        await app.workers.wait_for_complete()
        assert not app.full_screen
    assert kept[-1] is False, "the last choice is what is kept"


def test_the_diff_screen_s_hunk_chip_is_two_buttons() -> None:
    """`[ ]` is drawn once for two keys: `[` the previous hunk, `]` the next (off the bar)."""
    line = bar_line(TuiState(), COLOURS, view=ScreenView.DIFF, hunks=True)
    clicks = {
        span.style.partition("@click=")[2]
        for span in line.spans
        if isinstance(span.style, str) and "@click=" in span.style
    }
    assert "app.on_diff('hunk(-1)')" in clicks and "app.on_diff('hunk(1)')" in clicks


def test_a_bare_fullscreen_is_the_screen_s_own() -> None:
    toggled: list[str] = []

    class Screen:
        def toggle_fullscreen(self) -> None:
            toggled.append("^F")

    SCREEN_COMMANDS["/fullscreen"](Screen())  # pyright: ignore[reportArgumentType]
    assert toggled == ["^F"]


def test_the_where_row_says_the_index_is_being_built_until_it_is() -> None:
    """The index built beside the person is said on the repo row as a bar,
    with the stage's word either side of the reading, then gone; a build that failed (said,
    with no progress) is said too, in red, until the person runs `chickadee init`."""
    feed = Feed(SID)
    building = "building this repository's index"

    def row(progress: IndexProgress | None, glyphs: object = UNICODE) -> str:
        feed.snapshot(
            Liveness(kind="idle", last_seq=0, indexing=building, index_progress=progress)
        )
        return where_lines(feed.state, "r", glyphs, COLOURS).plain  # pyright: ignore[reportArgumentType]

    assert row(IndexProgress(stage="preparing")).endswith(" · indexing · preparing…")
    assert row(IndexProgress(stage="reading", done=820, total=1830)).endswith(
        " · indexing ████░░░░░░ 45% · 820/1830 files"
    )
    assert row(IndexProgress(stage="reading", done=0, total=0)).endswith(
        " · indexing ░░░░░░░░░░ 0% · 0/0 files"
    )
    assert row(IndexProgress(stage="reading", done=3, total=4), ASCII).endswith(
        " · indexing ########-- 75% · 3/4 files"
    ), "a terminal without the blocks"
    assert row(IndexProgress(stage="linking", done=4, total=4)).endswith(
        " · indexing · linking 4 files…"
    )
    feed.snapshot(Liveness(kind="idle", last_seq=0))
    assert "index" not in where_lines(feed.state, "r", UNICODE, COLOURS).plain
    failed = "the index could not be built (x); `chickadee init` says why"
    feed.snapshot(Liveness(kind="idle", last_seq=0, indexing=failed))
    line = where_lines(feed.state, "r", UNICODE, COLOURS)
    assert line.plain.endswith(f" · {failed}")
    assert any(
        span.style == COLOURS["bad"] and line.plain[span.start : span.end] == failed
        for span in line.spans
    ), "in red: told by the missing progress, not by the sentence's words"


def test_liveness_is_read_again_while_a_build_runs_and_not_after_it_failed() -> None:
    """The poll follows the build's progress; a failure, said with no progress, is
    read no more until something else moves the screen."""
    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    asked: list[set[str]] = []

    class Live:
        def refresh(self, routes: set[str]) -> None:
            asked.append(routes)

    app.live = Live()  # type: ignore[assignment]
    feed.snapshot(
        Liveness(
            kind="idle", last_seq=0, indexing="b", index_progress=IndexProgress(stage="reading")
        )
    )
    app._indexing_poll()  # pyright: ignore[reportPrivateUsage]
    assert asked == [{StaleRoute.LIVENESS}]
    feed.snapshot(Liveness(kind="idle", last_seq=0, indexing="the index could not be built"))
    app._indexing_poll()  # pyright: ignore[reportPrivateUsage]
    assert len(asked) == 1, "failed: nothing more to read"


def test_the_header_s_tree_is_read_again_when_the_terminal_gives_the_focus_back() -> None:
    """A commit the person makes in another window says nothing on the wire; coming
    back to the screen reads the header's tree again, as the slow poll does."""
    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    asked: list[set[str]] = []

    class Live:
        def refresh(self, routes: set[str]) -> None:
            asked.append(routes)

    app.live = Live()  # type: ignore[assignment]
    from textual import events

    app.on_app_focus(events.AppFocus())
    app._header_poll()  # pyright: ignore[reportPrivateUsage]
    assert asked == [{StaleRoute.HEADER}, {StaleRoute.HEADER}]
